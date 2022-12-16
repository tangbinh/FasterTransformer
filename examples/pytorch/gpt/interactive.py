import argparse
import os
import sys
import torch
import torch.distributed as dist
import time

from transformers import AutoTokenizer, AutoConfig

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../..")
from examples.pytorch.gpt.utils.parallel_gpt import ParallelGPT


def main(args):
    dist.init_process_group(backend="mpi")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    config = vars(AutoConfig.from_pretrained(args.model_path))

    gpt = ParallelGPT(
        head_num=config["num_attention_heads"],
        size_per_head=config["hidden_size"] // config["num_attention_heads"],
        vocab_size=config["vocab_size"],
        start_id=config["bos_token_id"],
        end_id=config["bos_token_id"],
        layer_num=config["num_hidden_layers"],
        max_seq_len=config["max_position_embeddings"],
        tensor_para_size=args.model_parallel,
        pipeline_para_size=args.pipeline_parallel,
        int8_mode=args.int8,
        layernorm_eps=1e-5,
        layernorm_type="pre_layernorm",
        activation_type="Relu",
        has_post_decoder_layernorm=True,
        weights_data_type=args.weight_type,
        lib_path=args.lib_path,
        # inference_data_type=args.data_type,
    )
    gpt.weights.load(
        os.path.join(args.model_path, f"c-model/{args.model_parallel}-gpu"),
        tensor_para_rank=gpt.tensor_para_rank,
        pipeline_para_rank=gpt.pipeline_para_rank,
    )
    if args.data_type == "fp16":
        gpt.weights._map(lambda w: w.half())
    elif args.data_type == "bf16":
        gpt.weights._map(lambda w: w.bfloat16())
    gpt.cuda()

    @torch.inference_mode()
    def generate(tokens):
        output, seq_length = gpt(
            start_ids=tokens,
            start_lengths=torch.tensor([len(tokens[0])], dtype=torch.int32),
            output_len=args.output_len,
            beam_width=args.beam_width,
            top_k=torch.full((args.batch_size,), args.top_k, dtype=torch.int32),
            top_p=torch.full((args.batch_size,), args.top_p, dtype=torch.float32),
            beam_search_diversity_rate=torch.full(
                (args.batch_size,), args.diversity_rate, dtype=torch.float32
            ),
            temperature=torch.full(
                (args.batch_size,), args.temperature, dtype=torch.float32
            ),
            len_penalty=torch.full(
                (args.batch_size,), args.len_penalty, dtype=torch.float32
            ),
            repetition_penalty=torch.full(
                (args.batch_size,), args.repetition_penalty, dtype=torch.float32
            ),
            random_seed=torch.full((args.batch_size,), 0, dtype=torch.int64),
            return_output_length=True,
            return_cum_log_probs=0,
        )
        output_lines = tokenizer.decode(output[0][0][len(tokens[0]) : seq_length[0]])
        output_lines = ".".join(output_lines.split(".")[:4]) + "."
        return output_lines, output, seq_length

    object = [None]
    while True:
        if torch.distributed.get_rank() == 0:
            prompt = input("\033[32mPrompt: \033[0;1m").rstrip()
            tokens = tokenizer.encode(prompt, return_tensors="pt")
            tokens = tokens[:, -923:]
            tokens = tokens.type(torch.int32)
            object = [tokens]

        dist.broadcast_object_list(object, src=0)
        start = time.monotonic()
        output, output_token, seq_length= generate(object[0])
        end = time.monotonic()

        if torch.distributed.get_rank() == 0:
            print(f"\033[0mOutput: {output}")
            print(f"Took {1000 * (end - start):4f} ms")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="opt-125m")
    parser.add_argument("--data-type", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--max-ite", type=int, default=20)
    parser.add_argument("--lib-path", type=str, default="./lib/libth_parallel_gpt.so")
    parser.add_argument("--model-parallel", type=int, default=1)
    parser.add_argument("--pipeline-parallel", type=int, default=1)
    parser.add_argument("--weight-type", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--int8", type=int, default=0)

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--beam-width", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--len-penalty", type=float, default=0.0)
    parser.add_argument("--diversity-rate", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
