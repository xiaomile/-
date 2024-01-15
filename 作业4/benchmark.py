import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from lmdeploy import turbomind as tm


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

import sys
import time

device = "cuda:0"

model_path = "/root/share/temp/model_repos/internlm-chat-7b/"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

engine = sys.argv[1]

tm_model = tm.TurboMind.from_pretrained("./workspace/")
model = tm_model.create_instance()
gen_func = gen_lmdeploy


count = 0
start = time.time()
for i in range(100):
    q = "写一篇关于四季的300字小作文。"
    output = gen_func(q, tokenizer, model)
    count += len(output)
end = time.time()
cost = (end - start)
throughput = round(count / cost)

print(f"{engine} 耗时 {cost:.2f}秒  {throughput} 字/秒")
