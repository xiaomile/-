from lmdeploy import turbomind as tm
import time
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# load model
if len(sys.argv)==2:
    model_path = sys.argv[1]
    engine = "turbomind"
elif len(sys.argv)==3:
    model_path = sys.argv[1]
    engine = sys.argv[2]
else:
    model_path = "./workspace"
    engine = "turbomind"

if engine not in ['turbomind','hf']:
    raise("engine must in ('turbomind','hf')")

def get_prompt(query: str) -> str:
    prompt = f"""<|System|>:You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.

<|User|>:{query}
<|Bot|>:"""
    return prompt



def gen_lmdeploy(query: str, tokenizer, model) -> str:
    prompt = get_prompt(query)
    input_ids = tokenizer.encode(prompt)
    for outputs in model.stream_infer(
            session_id=0,
            input_ids=[input_ids],
            request_output_len=512,
            temperature=0.0,
        ):
        ...
    res, _tokens = outputs[0]
    output = tokenizer.decode(res)
    return output



def gen_transformers(query: str, tokenizer, model) -> str:
    prompt = get_prompt(query)
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    res = model.generate(
        **inputs,
        max_new_tokens=512,
    )
    output = tokenizer.decode(res[0])
    output = output.replace(prompt, "").replace("<eoa>", "").strip()
    return output

device = "cuda:0"




if engine == "hf":
    model_path = "/root/share/temp/model_repos/internlm-chat-7b/"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True)
    model.to(device).eval();
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    gen_func = gen_transformers
else:
    tm_model = tm.TurboMind.from_pretrained(model_path)
    model = tm_model.create_instance()
    tokenizer = tm_model.tokenizer
    gen_func = gen_lmdeploy


count = 0
start = time.time()
query = "你好啊兄嘚"
for i in range(100):
    # inference
    response = gen_func(query, tokenizer, model)
    count += len(response)
end = time.time()
cost = (end - start)
throughput = round(count / cost)

print(f"推理引擎{engine} 耗时 {cost:.2f}秒  吞吐量 {throughput} 字/秒")
