import torch
import os
from tqdm import tqdm
import re
import math
import json
import gc
import time
from prompts import *


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def save_json(filename, ds):
    with open(filename, 'w', encoding="utf-8") as f:
        json.dump(ds, f, indent=2, ensure_ascii=False)


def parse_list_response(text):
    code_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if code_match:
        list_text = code_match.group(1)
    else:
        list_match = re.search(r'\[[\s\S]*?\]', text)
        if list_match:
            list_text = list_match.group(0)
        else:
            list_text = text

    try:
        parsed_data = json.loads(list_text)
        if isinstance(parsed_data, list):
            return parsed_data
        else:
            print(f"The returned text is not in list format: {parsed_data}")
            return str(text)
    except json.JSONDecodeError as e:
        print(f"List parsing failed: {e}")
        print("Raw text around:", repr(text[max(0, e.pos - 100): e.pos + 100]))
        return str(text)


def _split_atomic_actions_into_segments(atomic_actions: str, segment_num: int):
    atomic_action_list = atomic_actions.split('|')
    num_actions = len(atomic_action_list)
    segment_size = num_actions // segment_num

    segments = []
    for i in range(segment_num):
        if i != segment_num - 1:
            seg_list = atomic_action_list[i * segment_size:(i + 1) * segment_size]
        else:
            seg_list = atomic_action_list[i * segment_size:]
        segments.append('|'.join(seg_list))
    return segments


# split by fixed window size
def _split_atomic_actions_by_window(atomic_actions: str, window_size: int):
    if window_size <= 0:
        raise ValueError("window_size must be > 0 in _split_atomic_actions_by_window")

    atomic_action_list = atomic_actions.split('|')
    segments = []
    for i in range(0, len(atomic_action_list), window_size):
        seg_list = atomic_action_list[i:i + window_size]  # last one may be shorter, keep whole
        segments.append('|'.join(seg_list))
    return segments


def construct_messages(sample, infer_strategy, segment_num, rollout="single-rollout", window_size: int = 0):
    """
    - single-step: one prompt
    - multi-step + single-rollout: one prompt (expects JSON list)
    - multi-step + multi-rollout: per-segment prompt in rollout loop

    window_size behavior:
      - window_size == 0: old segmentation by segment_num
      - window_size > 0: segment by fixed window_size, segment_num becomes dynamic = len(segments)
    """
    start_frame = sample["start_frame"]
    atomic_actions = sample["atomic_actions"]
    dataset_path = sample["dataset_path"]

    start_frame_path = os.path.join(dataset_path, 'images', start_frame)

    messages = {
        "start_frame_path": start_frame_path,
        "infer_strategy": infer_strategy,
        "segment_num": segment_num,
        "rollout": rollout,
        "window_size": window_size,  
    }

    if infer_strategy == 'single-step':
        messages["prompt"] = final_scene_prediction_prompt.format(atomic_actions=atomic_actions)

    elif infer_strategy == 'multi-step':
        # choose splitting method
        if window_size and window_size > 0:
            segments = _split_atomic_actions_by_window(atomic_actions, window_size)
        else:
            segments = _split_atomic_actions_into_segments(atomic_actions, segment_num)

        # update dynamic segment_num
        messages["segment_num"] = len(segments)
        sample["atomic_action_segments_list"] = segments  # for multi-rollout usage

        if rollout == "single-rollout":
            atomic_action_segments = ""
            for i, seg in enumerate(segments):
                atomic_action_segments += f"Segment {i + 1}: {seg}\n"
            atomic_action_segments = atomic_action_segments.strip()
            messages["prompt"] = mutil_scene_prediction_prompt.format(
                segment_num=len(segments),
                atomic_action_segments=atomic_action_segments
            )
        elif rollout == "multi-rollout":
            # prompt built per step in run_model
            pass
        else:
            raise ValueError(f"unsupported rollout: {rollout}")

    else:
        raise ValueError(f"unsupported infer_strategy: {infer_strategy}")

    sample["messages"] = messages
    return sample


def load_dataset(dataset_path, ann_file):
    dataset = []
    json_path = os.path.join(dataset_path, ann_file)
    with open(json_path, "r", encoding="utf-8") as f:
        datas = json.load(f)

    for data in datas:
        sample = dict()
        sample['start_frame'] = data['start_frame']
        sample['atomic_actions'] = data['atomic_actions']
        sample["dataset_path"] = dataset_path
        dataset.append(sample)

    chunks_num = int(os.environ.get("num_chunks", 1))
    chunk_idx = int(os.environ.get("chunk_idx", 0))
    print(f"chunks_num: {chunks_num}, chunk_idx: {chunk_idx}")
    chunk = get_chunk(dataset, chunks_num, chunk_idx)
    return chunk


def _call_model_once(model, prompt: str, start_frame_path: str):
    messages = {
        "prompt": prompt,
        "start_frame_path": start_frame_path,
        "infer_strategy": "multi-step",
        "segment_num": 1,
    }
    return model.generate_outputs([messages])[0]


def run_model(samples, model, save_path):
    out_samples = []

    with open(save_path, "w", encoding="utf-8") as f, torch.no_grad():
        for sample in tqdm(samples):
            messages = sample["messages"]
            infer_strategy = messages["infer_strategy"]
            rollout = messages.get("rollout", "single-rollout")
            segment_num = messages["segment_num"]
            start_frame_path = messages["start_frame_path"]

            success = True
            response = None

            try:
                if infer_strategy == "single-step":
                    response = model.generate_outputs([messages])[0]

                elif infer_strategy == "multi-step" and rollout == "single-rollout":
                    response = model.generate_outputs([messages])[0]
                    response = parse_list_response(response)
                    if isinstance(response, str):
                        success = False

                elif infer_strategy == "multi-step" and rollout == "multi-rollout":
                    segments = sample.get("atomic_action_segments_list", None)
                    if not segments or len(segments) != segment_num:
                        raise ValueError("atomic_action_segments_list missing or invalid")

                    rollout_scenes = []
                    prev_scene = None
                    for i in range(segment_num):
                        seg_actions = segments[i]
                        if i == 0:
                            prompt = final_scene_prediction_prompt.format(
                                atomic_actions=seg_actions
                            )
                        else:
                            prompt = multi_rollout_scene_prediction_prompt_next.format(
                                previous_scene=prev_scene,
                                atomic_actions=seg_actions
                            )

                        out_text = _call_model_once(model, prompt, start_frame_path)
                        scene_text = out_text.strip()
                        rollout_scenes.append(scene_text)
                        prev_scene = scene_text

                    response = rollout_scenes
                else:
                    raise ValueError(f"unsupported combination: {infer_strategy} + {rollout}")

            except Exception as e:
                print("ОШИБКА")
                print(e)
                response = "response error"
                success = False
                
            if response is None:
                response = "response error"
                success = False

            out_sample = dict(sample)
            out_sample.pop("messages", None)
            out_sample.pop("atomic_actions", None)
            out_sample.pop("dataset_path", None)
            out_sample.pop("atomic_action_segments_list", None)

            out_sample["response"] = response
            out_sample["success"] = success

            out_samples.append(out_sample)

            f.write(json.dumps(out_sample, ensure_ascii=False) + "\n")
            f.flush()

            gc.collect()

    return out_samples


def model_inference(model, dataset_path, ann_file, output_path, infer_strategy, segment_num, rollout="single-rollout", window_size: int = 0):
    dataset = load_dataset(dataset_path, ann_file)
    samples = []
    for sample in dataset:
        sample = construct_messages(sample, infer_strategy, segment_num, rollout=rollout, window_size=window_size)
        samples.append(sample)

    chunk_idx = int(os.environ.get("chunk_idx", 0))
    num_chunks = int(os.environ.get("num_chunks", 1))

    results_all_ready = False
    
    if num_chunks > 1:
        results_path = os.path.join(output_path, f"results_{chunk_idx}.json")
        t0 = time.time()
        _ = run_model(samples, model, results_path)
        t1 = time.time()

        total_results_path = os.listdir(output_path)
        total_results_path = [result for result in total_results_path if result.startswith("results_")]
        if len(total_results_path) == num_chunks:
            results_all_ready = True
            total_results = []
            for result in total_results_path:
                rp = os.path.join(output_path, result)
                with open(rp, "r", encoding="utf-8") as f:
                    extend_datas = f.readlines()
                    extend_datas = [json.loads(data) for data in extend_datas]
                    total_results.extend(extend_datas)
            with open(os.path.join(output_path, "results.json"), "w", encoding="utf-8") as f:
                json.dump(total_results, f, indent=2, ensure_ascii=False)

    elif num_chunks == 1:
        results_path = os.path.join(output_path, "results.json")
        t0 = time.time()
        out_samples = run_model(samples, model, results_path)
        t1 = time.time()
        save_json(results_path, out_samples)
        results_all_ready = True

    else:
        raise ValueError("num_chunks must be greater than 0")
    
    # write per-chunk time
    stats = {
        "chunk_idx": chunk_idx,
        "num_chunks": num_chunks,
        "elapsed_seconds": t1 - t0,
        "num_samples_in_chunk": len(samples),
        "infer_strategy": infer_strategy,
        "rollout": rollout
    }
    stats_path = os.path.join(output_path, f"chunk_time_{chunk_idx}.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
        
    if results_all_ready:
        ok = try_aggregate_chunk_times(output_path, num_chunks, out_name="chunk_time_all.json")
        if ok:
            print(f"chunk_time_all.json saved to {os.path.join(output_path, 'chunk_time_all.json')}")

    print(f'chunk {chunk_idx} inference finished')


def load_list_of_dicts(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON list[dict]")
    return [x for x in data if isinstance(x, dict)]


def after_think(text):
        tag = "</think>"
        i = text.find(tag)
        return text[i + len(tag):].lstrip() if i != -1 else text
    
    
def remove_tags(text):
    text = text.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "")
    text = text.replace("<answer>", "").replace("</answer>", "").replace(r"\boxed{", "").replace(r"}", "")
    
    return text


def convert_list_to_dict(
    input_json: str,
    output_json: str,
    start_key: str = "start_frame",
    response_key: str = "response",
):
    data = load_list_of_dicts(input_json)

    out = {}
    for d in data:
        if start_key in d and response_key in d:
            org_response = d[response_key]
            if isinstance(org_response, str):
                response = remove_tags(after_think(org_response))
            elif isinstance(org_response, list):
                response = [remove_tags(after_think(org_res)) for org_res in org_response]
            out[str(d[start_key])] = response

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f'num of saved records: {len(out)}')
    return out


def try_aggregate_chunk_times(output_path: str, num_chunks: int, out_name: str = "chunk_time_all.json"):
    """
    Aggregate chunk_time_{i}.json into one file when all are ready.
    Records per-chunk elapsed, and min/max/avg/sum elapsed (seconds).
    """
    all_paths = [os.path.join(output_path, f"chunk_time_{i}.json") for i in range(num_chunks)]
    if not all(os.path.exists(p) for p in all_paths):
        return False

    items = []
    for p in all_paths:
        with open(p, "r", encoding="utf-8") as f:
            items.append(json.load(f))

    items = sorted(items, key=lambda x: x.get("chunk_idx", -1))

    # per-chunk elapsed
    per_chunk_elapsed = {}
    elapsed_list = []
    for x in items:
        idx = x.get("chunk_idx")
        el = float(x.get("elapsed_seconds", 0.0))
        per_chunk_elapsed[str(idx)] = el
        elapsed_list.append(el)

    if len(elapsed_list) == 0:
        min_elapsed = max_elapsed = avg_elapsed = sum_elapsed = 0.0
    else:
        sum_elapsed = sum(elapsed_list)
        min_elapsed = min(elapsed_list)
        max_elapsed = max(elapsed_list)
        avg_elapsed = sum_elapsed / len(elapsed_list)

    out = {
        "num_chunks": num_chunks,
        "per_chunk_elapsed_seconds": per_chunk_elapsed,
        "min_elapsed_seconds": min_elapsed,
        "max_elapsed_seconds": max_elapsed,
        "avg_elapsed_seconds": avg_elapsed,
        "sum_elapsed_seconds": sum_elapsed,  # sum across chunks (not wall time)
        "chunks": items,  # keep full metadata per chunk
    }

    out_path = os.path.join(output_path, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    return True
