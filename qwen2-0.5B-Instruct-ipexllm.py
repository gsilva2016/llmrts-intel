import torch
from transformers import AutoTokenizer #, AutoModelForCausalLM
#from intel_extension_for_transformers.transformers import AutoModelForCausalLM
from ipex_llm.transformers import AutoModelForCausalLM
import argparse

# auto completed in ipex-llm
#import intel_extension_for_pytorch as ipex
import time

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-d',
                        '--device',
                        default='GPU.0',
                        required=False,
                        type=str,
                        help='Required. device for inference')
args = parser.parse_args()
device = args.device
#device = "xpu:0" # the device to load the model onto
device_map = device

model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2-0.5B-Instruct",
            torch_dtype="auto",
            device_map="auto"
        )

# auto done with ipex-llm
#if device == "xpu":
#    model = ipex.optimize_transformers(model, inplace=True, dtype=torch.float16, device=device)

model = model.half().to(device)
#print(model.dtype)
#print(model)
start_time = time.time()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

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
model_inputs = tokenizer([text], return_tensors="pt").to(device)


print("Calling generate...")
start_time = time.time()
generated_ids = model.generate(
            model_inputs.input_ids,
                max_new_tokens=512
                )
tokens_len = len(generated_ids[0])
print(tokens_len, " tokens")
elapsed_time = time.time() - start_time
print(f"tps: { tokens_len / elapsed_time }")
print("Generate completed...throwing away and doing another")


start_time = time.time()
generated_ids = model.generate(
            model_inputs.input_ids,
                max_new_tokens=512
                )
tokens_len = len(generated_ids[0])

generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

elapsed_time = time.time() - start_time
print(f"tps: { tokens_len / elapsed_time }")
print(response)
