基础作业

使用 OpenCompass 评测 InternLM2-Chat-7B 模型在 C-Eval 数据集上的性能

使用huggingface-cli下载模型时报错
![image](https://github.com/xiaomile/InternLM-homework/assets/14927720/e86acbdc-c293-47ac-a2b8-5152ec146813)

改用huggingface_hub的python包来下载
```python
import os

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 下载模型
from huggingface_hub import snapshot_download
snapshot_download(repo_id="internlm/internlm2-chat-7b", local_dir="./model/internlm2-chat-7b")
```

