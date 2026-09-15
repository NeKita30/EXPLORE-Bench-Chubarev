import os
import argparse
import torch
from tqdm import tqdm
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import spacy
import multiprocessing as mp

from utils.evaluator import BaseEvaluator
from dataset.bench_dataset import BenchData
from utils.aggregate_res import aggregate_mutil_woker_results, aggregate_mutil_time_results

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()


def build_scorer(model, device):
    if 'Qwen' in model:
        from utils.metric.qwen3 import Qwen
        scorer = Qwen(model, device)
    else:
        raise ValueError(f"{model} is not avalible")
    return scorer


class Evaluator(BaseEvaluator):
    def __init__(self, args):
        super(Evaluator, self).__init__(args)

        self.scorer = build_scorer(args.llm, args.device)
        self.nlp = spacy.load('en_core_web_lg')
        self.bert = SentenceTransformer(args.bert, device=args.device)

        self.data = BenchData(args.data_root, args.anno)
        self.len_data = len(self.data)
        self.num_data = min(args.data_num, self.len_data) if args.data_num > 0 else self.len_data

        self.soft_coverage = args.soft_coverage
        
        self.eval_mode = args.eval_mode
        self.which_scene = args.which_scene

    def get_key_words(self, sub_captions):
        words = []
        word2caption = {}
        for i, text in enumerate(sub_captions):
            doc = self.nlp(text)
            candidates = []
            for token in doc:
                if token.tag_ and token.tag_[0] == 'N':
                    candidates.append(str(token.lemma_.lower()))
            if len(candidates) == 0:
                continue

            start_id = len(words)
            words = words + candidates
            for w in range(start_id, len(words)):
                word2caption[w] = i
        return words, word2caption

   
    def start(self, indices=None):
        if indices is None:
            indices = range(self.num_data)

        eval_mode = self.eval_mode

        with torch.no_grad():
            for index in tqdm(list(indices)):
                start_frame, gt_objects, gt_description, gt_relations = self.data.get_data(index)

                caption = self.long_caption.get(start_frame, None)
                if not caption:
                    print('No predicted final-scene description for the initial-scene image:', start_frame)
                    continue

                if eval_mode == 'single-scene':
                    which_scene = self.which_scene
                    if not gt_objects[which_scene]:
                        continue

                    cur_caption = caption
                    if isinstance(cur_caption, list):
                        if which_scene == 'final':
                            cur_caption = cur_caption[-1]
                        else:
                            raise ValueError(f"unsupported scene: {which_scene}")

                    cur_gt_objects = gt_objects[which_scene]
                    cur_gt_description = gt_description[which_scene]
                    cur_gt_relations = gt_relations[which_scene]

                    self._eval_one_scene(
                        start_frame=start_frame,
                        scene=which_scene,
                        caption_text=cur_caption,
                        gt_objects=cur_gt_objects,
                        gt_description=cur_gt_description,
                        gt_relations=cur_gt_relations
                    )

                else:
                    raise ValueError(f"Unknown eval_mode: {eval_mode}")
                    
    def _eval_one_scene(self, start_frame, scene, caption_text, gt_objects, gt_description, gt_relations):
        sub_captions = self.split_caption(caption_text)
        words, word2caption = self.get_key_words(sub_captions)

        no_overlap_gt_objects = list(set(gt_objects))
        match_object, object_score = self.get_coverage(no_overlap_gt_objects, words, scene=scene)

        des_score, relation_score = self.get_accuracy(
            gt_objects, no_overlap_gt_objects, gt_description, gt_relations,
            sub_captions, word2caption, match_object, scene=scene
        )

        result = {'object': {}, 'attribute': {}, 'relation': {}}
        for k, gt_object in enumerate(no_overlap_gt_objects):
            result['object'][gt_object] = float(object_score[k])

        for k, des in enumerate(gt_description):
            result['attribute'][des] = float(des_score[k])

        relations = []
        for r in gt_relations:
            if isinstance(r, list):
                relations += r
            elif isinstance(r, str):
                relations.append(r)
        assert len(relations) == len(relation_score)
        for k, relation in enumerate(relations):
            result['relation'][relation] = float(relation_score[k])

        self.log_results(start_frame, scene, result)
    
    def get_coverage(self, no_overlap_gt_objects, words, scene=None):
        with torch.no_grad():
            gts_embed = self.bert.encode(no_overlap_gt_objects, convert_to_tensor=True, device=self.bert.device)
            preds_embed = self.bert.encode(words, convert_to_tensor=True, device=self.bert.device)
            cosine_scores = self.bert.similarity(gts_embed, preds_embed)

        max_sim = cosine_scores.max(dim=0)[0]
        gt2preds_max_sim = cosine_scores.max(dim=1)[0]

        match_object = (cosine_scores == max_sim.unsqueeze(0)) * (cosine_scores == gt2preds_max_sim.unsqueeze(-1))
        object_score = match_object.int().sum(dim=-1) > 0
        contained_object_idx = torch.where(object_score)[0]

        if self.soft_coverage:
            contained_object_score = [gt2preds_max_sim[i].item() for i in contained_object_idx]
            object_score = object_score * gt2preds_max_sim
            cur_coverage = sum(contained_object_score) / len(no_overlap_gt_objects)
        else:
            cur_coverage = len(contained_object_idx) / len(no_overlap_gt_objects)

        if self.eval_mode == "multi-scene":
            assert scene is not None
            self.update_object_coverage_scene(scene, cur_coverage)
        else:
            self.update_object_coverage(cur_coverage)

        return match_object, object_score

    def get_accuracy(self, gt_objects, no_overlap_gt_objects, gt_description, gt_relations,
                     sub_captions, word2caption, match_object, scene=None):
        _match_object = []
        for gt in gt_objects:
            object_id = no_overlap_gt_objects.index(gt)
            _match_object.append(match_object[object_id])
        match_object = torch.stack(_match_object, dim=0)

        preds, gts, rels, rels_preds = [], [], [], []
        for k in range(len(gt_description)):
            cur_match = list(match_object[k, :].int().nonzero().view(-1))
            gts.append(gt_description[k])

            object_captions = sorted([word2caption[int(cur_)] for cur_ in cur_match])
            pred_caption = [sub_captions[subcaption_id] for subcaption_id in object_captions]
            preds.append('. '.join(pred_caption))

            if len(gt_relations[k]) > 0:
                for cur_obj_gt_relation in gt_relations[k]:
                    for k_obj, gt_object in enumerate(gt_objects):
                        if k_obj == k:
                            continue
                        if gt_object in cur_obj_gt_relation:
                            cur_match_2 = list(match_object[k_obj, :].int().nonzero().view(-1))
                            object_captions = object_captions + sorted([word2caption[int(cur_)] for cur_ in cur_match_2])

                    relation_pred_caption = [sub_captions[subcaption_id] for subcaption_id in sorted(list(set(object_captions)))]
                    rels_preds.append('. '.join(relation_pred_caption))
                    rels.append(cur_obj_gt_relation)
            else:
                rels_preds.append('')

        object_score, relation_score = self.scorer.evaluate(gts, preds, rels, rels_preds)

        attr_avg = sum(object_score) / len(gt_description)
        rel_avg = sum(relation_score) / len(relation_score) if len(relation_score) > 0 else 0.0

        if self.eval_mode == "multi-scene":
            assert scene is not None
            self.update_attribute_score_scene(scene, attr_avg)
            self.update_relation_score_scene(scene, rel_avg)
        else:
            self.update_attribute_score(attr_avg)
            assert len(relation_score) > 0
            self.update_relation_score(rel_avg)

        return object_score, relation_score


def split_indices(total, num_chunks):
    chunk_size = total // num_chunks
    chunks = [list(range(i * chunk_size, (i + 1) * chunk_size)) for i in range(num_chunks)]
    if total % num_chunks != 0:
        chunks[-1].extend(list(range(num_chunks * chunk_size, total)))
    return chunks


def worker_process(rank, gpu_id, indices, args_dict):
    # 子进程内重建 args
    args = argparse.Namespace(**args_dict)

    args.device = f"mps"

    args.output_dir = os.path.join(args.output_dir, f"worker_{rank}")
    os.makedirs(args.output_dir, exist_ok=True)

    evaluator = Evaluator(args)

    for i in range(args.k):
        print(f"[worker {rank} | gpu {gpu_id}] Start eval. Epoch={i}, num={len(indices)}")
        evaluator.reset_record()
        evaluator.save_index = i
        evaluator.start(indices=indices)
        if args.eval_mode == "single-scene" and evaluator.count < len(indices):
            print(f"[worker {rank}] Only {evaluator.count}/{len(indices)} used for evaluation.")
        elif args.eval_mode == "multi-scene" and evaluator.scene_count['final'] < len(indices):
            print(f"[worker {rank}] Only {evaluator.scene_count['final']}/{len(indices)} used for evaluation.")
        evaluator.get_results()

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SceneDescription-Evaluator", add_help=True)
    parser.add_argument("--data_root", type=str, required=True, help="path to benchmark dataset")
    parser.add_argument("--anno", type=str, required=True, help="path to annotation file")
    parser.add_argument("--description_file", type=str, required=True, help="path to scene descriptions to be evaluted")
    parser.add_argument("--llm", type=str, help="path to llm scorer")
    parser.add_argument("--bert", type=str, help="path to sbert")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--soft_coverage", action="store_true", help="use word similarity as coverage")
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument("--k", type=int, default=1, help="eval time")
    parser.add_argument("--data_num", type=int, default=-1, help="the number of data to be used for eval")
    
    # infer 参数（用于命名 output_dir）
    parser.add_argument('--infer_strategy', type=str, default='single-step', choices=['single-step', 'multi-step'])
    parser.add_argument('--window_size', type=int, default=0)
    parser.add_argument('--segment_num', type=int, default=1)
    parser.add_argument('--rollout', type=str, default='single-rollout', choices=['single-rollout', 'multi-rollout'])
    
    # eval 参数
    parser.add_argument('--eval_mode', type=str, default='single-scene', choices=['single-scene', 'multi-scene'])
    parser.add_argument('--which_scene', type=str, default='final', choices=['mid_1', 'mid_2', 'final'])  # available when eval_mode is single-scene
    parser.add_argument('--dataset_type', type=str, default='full', choices=['short_seq', 'medium_seq', 'long_seq', 'full', 'self_recorded', 'tiny_short', 'tiny_medium', 'tiny_long', 'tiny_full'])

    # 多进程参数
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--gpu_ids", type=str, default="", help="e.g. '0,1,2,3'; 0..num_processes-1 if is '' ")

    args = parser.parse_args()
    
    if args.infer_strategy == 'multi-step':
        if args.eval_mode == 'single-scene':
            if args.window_size > 0:
                args.output_dir = os.path.join(args.output_dir, args.infer_strategy, f"window_{args.window_size}", args.rollout, args.eval_mode, args.which_scene, args.dataset_type)
            else:
                args.output_dir = os.path.join(args.output_dir, args.infer_strategy, f"{args.segment_num}_segments", args.rollout, args.eval_mode, args.which_scene, args.dataset_type)
        elif args.eval_mode == 'multi-scene':
            args.segment_num = 3
            args.output_dir = os.path.join(args.output_dir, args.infer_strategy, f"{args.segment_num}_segments", args.rollout, args.eval_mode, args.dataset_type)
    elif args.infer_strategy == 'single-step':
        if args.eval_mode == 'single-scene':
            args.output_dir = os.path.join(args.output_dir, args.infer_strategy, args.eval_mode, args.which_scene, args.dataset_type)
        elif args.eval_mode == 'multi-scene':
            args.output_dir = os.path.join(args.output_dir, args.infer_strategy, args.eval_mode, args.dataset_type)
    print(f'final ouput dir path: {args.output_dir}')
    os.makedirs(args.output_dir, exist_ok=True)
    
    mp.set_start_method("spawn", force=True)

    if args.gpu_ids.strip():
        gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
        assert len(gpu_ids) == args.num_processes, "The length of `gpu_ids` must be equal to `num_processes`."
    else:
        gpu_ids = [i for i in range(args.num_processes)]

    tmp_data = BenchData(args.data_root, args.anno)
    len_data = len(tmp_data)
    total = min(args.data_num, len_data) if args.data_num > 0 else len_data

    chunks = split_indices(total, args.num_processes)

    args_dict = vars(args).copy()

    with mp.Pool(processes=args.num_processes) as pool:
        process_args = [(rank, gpu_ids[rank], chunks[rank], args_dict) for rank in range(args.num_processes)]
        _ = pool.starmap(worker_process, process_args)

    for i in range(args.k):
        aggregate_mutil_woker_results(args.output_dir, target_name=f"result_{i}.json")
    aggregate_mutil_time_results(args.output_dir)
    
