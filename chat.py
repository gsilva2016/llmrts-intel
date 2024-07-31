import argparse
from typing import List, Tuple
from threading import Thread
import torch
from optimum.intel.openvino import OVModelForCausalLM
from transformers import (AutoTokenizer, AutoConfig,
                          TextIteratorStreamer, StoppingCriteriaList, StoppingCriteria)
import time
device = ""

class StopOnTokens(StoppingCriteria):
    def __init__(self, token_ids):
        self.token_ids = token_ids

    def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for stop_id in self.token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h',
                        '--help',
                        action='help',
                        help='Show this help message and exit.')
    parser.add_argument('-m',
                        '--model_path',
                        required=True,
                        type=str,
                        help='Required. model path')
    parser.add_argument('-l',
                        '--max_sequence_length',
                        default=256,
                        required=False,
                        type=int,
                        help='Required. maximun length of output')
    parser.add_argument('-d',
                        '--device',
                        default='GPU.0',
                        required=False,
                        type=str,
                        help='Required. device for inference')
    args = parser.parse_args()
    model_dir = args.model_path

    ov_config = {"PERFORMANCE_HINT": "LATENCY",
                 "NUM_STREAMS": "1", "CACHE_DIR": ""}

    device = args.device

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    ov_model = OVModelForCausalLM.from_pretrained(
        model_dir,
        device=args.device,
        ov_config=ov_config,
        config=AutoConfig.from_pretrained(model_dir),
        trust_remote_code=True,
    )

    prompt = "Does MLC LLM support Intel hardware?"
    messages = [
            {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
                ]

    text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
            )

    model_inputs = tokenizer([text], return_tensors="pt")

    # warm up
    start_time = time.time()
    generated_ids = ov_model.generate(
            model_inputs.input_ids,
                max_new_tokens=512
                )

    tokens_len = len(generated_ids[0])
    elapsed_time = time.time() - start_time
    print(f"tps: { tokens_len / elapsed_time }")

    start_time = time.time()
    generated_ids = ov_model.generate(
            model_inputs.input_ids,
                max_new_tokens=512
                )
    tokens_len = len(generated_ids[0])


    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)
    elapsed_time = time.time() - start_time
    print(f"tps: { tokens_len / elapsed_time }")
