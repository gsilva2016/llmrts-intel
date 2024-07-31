from mlc_llm import MLCEngine
import time

# Create engine
model = "HF://mlc-ai/Qwen2-0.5B-Instruct-q4f16_1-MLC"
engine = MLCEngine(model, device="vulkan:0")

# Run chat completion in OpenAI API.
start_time = time.time()
tokens_len = 0
for response in engine.chat.completions.create(
    messages=[{"role": "user", "content": "Does MLC LLM support Intel hardware?"}],
    model=model,
    stream=True,
):
    for choice in response.choices:
        tokens_len = tokens_len + 1
        print(choice.delta.content , end="", flush=True)
    
elapsed_time = time.time() - start_time
print(f"\ntps: { tokens_len / elapsed_time }")
print("\n")


start_time = time.time()
tokens_len = 0
for response in engine.chat.completions.create(
    messages=[{"role": "user", "content": "Does MLC LLM support Intel hardware?"}],
    model=model,
    stream=True,
):
    for choice in response.choices:
        tokens_len = tokens_len + 1
        print(choice.delta.content , end="", flush=True)

elapsed_time = time.time() - start_time
print(f"\ntps: { tokens_len / elapsed_time }")
print("\n")

engine.terminate()
