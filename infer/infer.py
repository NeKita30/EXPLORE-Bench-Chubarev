import torch
import os
import random
import numpy as np
from argparse import ArgumentParser
from LLMs import init_llm
from infer_utils import model_inference, convert_list_to_dict
import gc


def set_seed(seed_value):
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default="../../EXPLORE-Dataset/",
                        help='path of the eval dataset')
    parser.add_argument('--anno_file', type=str, default='anno.json',
                        help='file name of the annotation file')
    parser.add_argument('--output_path', type=str, default='infer_results/Qwen3-VL-8B-Instruct',
                        help='dir path of saved files')
    parser.add_argument('--model_name', type=str, default='qwen3-vl',
                        help='name of model')
    parser.add_argument('--model_path', type=str, default="Qwen/Qwen3-VL-8B-Instruct")

    parser.add_argument('--infer_strategy', type=str, default='single-step',
                        choices=['single-step', 'multi-step'])
    parser.add_argument('--rollout', type=str, default='single-rollout',
                        choices=['single-rollout', 'multi-rollout'])
    parser.add_argument('--window_size', type=int, default=0,
                        help='when >0 and infer_strategy=multi-step, split atomic actions by fixed window size')
    parser.add_argument('--segment_num', type=int, default=1)
    parser.add_argument('--enable_thinking', type=int, default=0,
                        choices=[0, 1])

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_chunks', type=str, default="1")
    parser.add_argument('--chunk_idx', type=str, default="0")
    parser.add_argument('--cuda_visible_devices', type=str, default=None)

    args = parser.parse_args()

    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        os.environ["tensor_parallel_size"] = str(len(args.cuda_visible_devices.split(",")))

    os.environ["num_chunks"] = args.num_chunks
    os.environ["chunk_idx"] = args.chunk_idx
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    eval_dataset_path = args.dataset_path
    if not os.path.exists(eval_dataset_path):
        raise RuntimeError(f"{eval_dataset_path} dose not exist")

    anno_file_path = args.anno_file
    infer_strategy = args.infer_strategy
    segment_num = args.segment_num
    rollout = args.rollout
    window_size = args.window_size  

    if infer_strategy == 'single-step':
        segment_num = 1
        rollout = "single-rollout"
        eval_output_path = os.path.join(args.output_path, infer_strategy)

    elif infer_strategy == 'multi-step':
        # window_size=0: require segment_num>1
        if window_size == 0:
            assert segment_num > 1, "segment_num must be greater than 1 when infer_strategy is multi-step and window_size=0"
            eval_output_path = os.path.join(args.output_path, infer_strategy, f"{segment_num}_segments", rollout)
        else:
            if window_size < 0:
                raise ValueError("window_size must be >= 0")
            # window_size>0: segment count is dynamic
            eval_output_path = os.path.join(args.output_path, infer_strategy, f"window_{window_size}", rollout)
    elif infer_strategy == "my-strategy":
        eval_output_path = os.path.join(args.output_path, infer_strategy)
    else:
        raise ValueError(f"unsupported infer_strategy: {infer_strategy}")

    print(f'final output dir path: {eval_output_path}')
    os.makedirs(eval_output_path, exist_ok=True)

    print('initializing LLM...')
    model = init_llm(args.model_name, args.model_path, bool(args.enable_thinking))
    set_seed(args.seed)

    model_inference(
        model=model,
        dataset_path=eval_dataset_path,
        ann_file=anno_file_path,
        output_path=eval_output_path,
        infer_strategy=infer_strategy,
        segment_num=segment_num,
        rollout=rollout,
        window_size=window_size,
    )

    convert_list_to_dict(
        os.path.join(eval_output_path, "results.json"),
        os.path.join(eval_output_path, f"{args.output_path.split('/')[-1]}.json")
    )

    print(f'final results saved to {eval_output_path}')
    gc.collect()


if __name__ == '__main__':
    main()