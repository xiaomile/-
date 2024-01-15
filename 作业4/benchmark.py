from lmdeploy import turbomind as tm

# load model
model_path = "/root/share/temp/model_repos/internlm-chat-7b/"
tm_model = tm.TurboMind.from_pretrained(model_path, model_name='internlm-chat-7b')
generator = tm_model.create_instance()

# process query
query = "你好啊兄嘚"
prompt = tm_model.model.get_prompt(query)
input_ids = tm_model.tokenizer.encode(prompt)


count = 0
start = time.time()
for i in range(100):
    # inference
    for outputs in generator.stream_infer(session_id=0,input_ids=[input_ids]):
        res, tokens = outputs[0]
    response = tm_model.tokenizer.decode(res)
    count += len(response)
end = time.time()
cost = (end - start)
throughput = round(count / cost)

print(f"耗时 {cost:.2f}秒  吞吐量 {throughput} 字/秒")
