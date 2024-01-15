基础作业：

- 使用 LMDeploy 以本地对话、网页Gradio、API服务中的一种方式部署 InternLM-Chat-7B 模型，生成 300 字的小故事（需截图）

离线转化模型为turbomind格式
![image](https://github.com/xiaomile/InternLM-homework/assets/14927720/6d1a7b6a-4516-4020-af0c-6d6cee5b6daf)

网页 Demo 演示(TurboMind 服务作为后端)

turbomind服务端启动！
![image](https://github.com/xiaomile/InternLM-homework/assets/14927720/35307a9d-da06-4db2-8658-b5454edab1b8)

gradio前端启动！
![image](https://github.com/xiaomile/InternLM-homework/assets/14927720/092432d6-3d02-4de5-ad62-06d50d9daaa3)

demo演示
![image](https://github.com/xiaomile/InternLM-homework/assets/14927720/b6658e67-3740-46a3-a4bc-e7c8bffc675a)





进阶作业（可选做）

将第四节课训练自我认知小助手模型使用 LMDeploy 量化部署到 OpenXLab 平台。

对internlm-chat-7b模型进行量化，并同时使用KV Cache量化，使用量化后的模型完成API服务的部署，分别对比模型量化前后和 KV Cache 量化前后的显存大小（将 bs设置为 1 和 max len 设置为512）。

打开config.ini 修改max_batch_size=1,cache_block_seq_len = 512
![image](https://github.com/xiaomile/InternLM-homework/assets/14927720/71d9881e-a940-4dc5-a184-d709ac8f0555)

量化前

启动API服务并监控显存占用情况
![image](https://github.com/xiaomile/InternLM-homework/assets/14927720/1b763187-773d-46c4-92e8-3db4388918da)
开始推理


![image](https://github.com/xiaomile/InternLM-homework/assets/14927720/df189b5e-de80-4662-84bb-5856c60f1a99)
显存占用 14822MB

开启kv cache量化并推理
![image](https://github.com/xiaomile/InternLM-homework/assets/14927720/9bf94628-c52f-418c-b993-8e2f2334f5a5)
显存占用 14726MB，比量化前少了一点

模型本身量化4bit并推理
![image](https://github.com/xiaomile/InternLM-homework/assets/14927720/74da7a7d-f7cd-42f0-9bdf-454bdf2e4ddb)
显存占用5856MB，比量化前大下降

模型本身量化4bit+kv cache量化后推理



并修改turbomind/weights/config.ini里的quant_policy为4（开启kv量化）


API服务启动！

在自己的任务数据集上任取若干条进行Benchmark测试，测试方向包括：
（1）TurboMind推理+Python代码集成
（2）在（1）的基础上采用W4A16量化
（3）在（1）的基础上开启KV Cache量化
（4）在（2）的基础上开启KV Cache量化
（5）使用Huggingface推理
