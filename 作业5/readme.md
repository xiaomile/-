基础作业

使用 OpenCompass 评测 InternLM2-Chat-7B 模型在 C-Eval 数据集上的性能

使用huggingface-cli下载模型时报错
![image](https://github.com/xiaomile/InternLM-homework/assets/14927720/e86acbdc-c293-47ac-a2b8-5152ec146813)

改用huggingface_hub的snapshot_download就能正常下载
```python
import os

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 下载模型
from huggingface_hub import snapshot_download
snapshot_download(repo_id="internlm/internlm2-chat-7b", local_dir="./model/internlm2-chat-7b")
```
![image](https://github.com/xiaomile/InternLM-homework/assets/14927720/fb988667-d6ee-4df1-9606-b2884de105ed)

数据集和opencompass的安装按照教程上来做即可

启动评测
```bash
python run.py --datasets ceval_gen --hf-path ./model/internlm2-chat-7b/ --tokenizer-path ./model/internlm2-chat-7b/ --tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True --model-kwargs trust_remote_code=True device_map='auto' --max-seq-len 2048 --max-out-len 16 --batch-size 4 --num-gpus 1
```
第一次评测爆了显存，升一下配置再来
![image](https://github.com/xiaomile/InternLM-homework/assets/14927720/3f765781-1909-4168-9324-2532513df303)



进阶作业

使用 OpenCompass 评测 InternLM2-Chat-7B 模型使用 LMDeploy 0.2.0 部署后在 C-Eval 数据集上的性能

