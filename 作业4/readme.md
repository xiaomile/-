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

- 将第四节课训练自我认知小助手模型使用 LMDeploy 量化部署到 OpenXLab 平台。

  量化模型
  ```
  cp /root/share/temp/datasets/c4/calib_dataloader.py  /root/.conda/envs/lmdeploy/lib/python3.10/site-packages/lmdeploy/lite/utils/
  cp -r /root/share/temp/datasets/c4/ /root/.cache/huggingface/datasets/
  # 第一步统计minmax校准
  lmdeploy lite calibrate \
    --model  /root/personal_assistant/config/work_dirs/hf_merge/ \
    --calib_dataset "c4" \
    --calib_samples 128 \
    --calib_seqlen 2048 \
    --work_dir ./quant_output_hfmerge
  # 第二步量化权重模型
  lmdeploy lite auto_awq \
    --model  /root/personal_assistant/config/work_dirs/hf_merge/ \
    --w_bits 4 \
    --w_group_size 128 \
    --work_dir ./quant_output_hfmerge
  # 第三步 转换模型的layout，转换成 TurboMind 格式
  lmdeploy convert  internlm-chat-7b ./quant_output_hfmerge \
      --model-format awq \
      --group-size 128 \
      --dst_path ./workspace_hfmerge
  ```
  测试一下模型量化后的效果
  ```
   lmdeploy chat turbomind ./workspace_hfmerge
  ```
  ![image](https://github.com/xiaomile/InternLM-homework/assets/14927720/5449bf11-c46e-4142-9938-3c92368c100c)
  可以看到显存占用只有5856MB，大幅低于原模型
  
  [模型上到传到openxlab](https://openxlab.org.cn/models/detail/xiaomile/personal_assistant_4bit)
  
  [openxlab应用地址](https://openxlab.org.cn/apps/detail/xiaomile/personal_assistant_4bit)
  
  应用发布不了，一直排队中。。。
  ![image](https://github.com/xiaomile/InternLM-homework/assets/14927720/6369302d-ecbc-4a96-bb80-281e79f26fd2)





- 对internlm-chat-7b模型进行量化，并同时使用KV Cache量化，使用量化后的模型完成API服务的部署，分别对比模型量化前后和 KV Cache 量化前后的显存大小（将 bs设置为 1 和 max len 设置为512）。

  打开config.ini 修改max_batch_size=1,cache_block_seq_len = 512
  
  ![image](https://github.com/xiaomile/InternLM-homework/assets/14927720/71d9881e-a940-4dc5-a184-d709ac8f0555)
  
  量化前
  
  启动API服务并监控显存占用情况
  ![image](https://github.com/xiaomile/InternLM-homework/assets/14927720/1b763187-773d-46c4-92e8-3db4388918da)
  开始推理
  ![image](https://github.com/xiaomile/InternLM-homework/assets/14927720/04c91e26-362c-4ab9-b997-fd118e4431b8)
  ![image](https://github.com/xiaomile/InternLM-homework/assets/14927720/a5f95ef1-df01-46e7-87af-90c373805b7b)
  
  显存占用 14918MB
  
  开启kv cache量化并推理(quant_policy=4,use_context_fmha=0)
  ![image](https://github.com/xiaomile/InternLM-homework/assets/14927720/608a855f-5c6e-4ff0-9553-92a3153b26b0)
  显存占用 14790MB，比量化前少了一点
  
  模型本身量化4bit并推理
  ![image](https://github.com/xiaomile/InternLM-homework/assets/14927720/20df6fe9-8a0f-4a39-a4fc-08d1cce0a3f6)
  显存占用5952MB，比量化前大幅下降
  
  模型本身量化4bit+kv cache量化后推理(quant_policy=4,use_context_fmha=0)
  ![image](https://github.com/xiaomile/InternLM-homework/assets/14927720/8a6ca8f0-82fc-4b0e-8213-ce56c1f6b5b4)
  显存占用5828MB，比4bit量化要少一点

- 在自己的任务数据集上任取若干条进行Benchmark测试（[测试脚本](benchmark.py)），测试方向包括：
  
  （1）TurboMind推理+Python代码集成
  ```
  python benchmark.py
  ```
  推理引擎turbomind 耗时 11.49秒  吞吐量 131 字/秒
  ![image](https://github.com/xiaomile/InternLM-homework/assets/14927720/e3a94f2a-9b25-4186-8e1c-866e04f1b976)
  
  （2）在（1）的基础上采用W4A16量化
  ```
  python benchmark.py ./workspace_quant
  ```
  推理引擎turbomind 耗时 7.62秒  吞吐量 236 字/秒
  ![image](https://github.com/xiaomile/InternLM-homework/assets/14927720/a92a1a42-9f8b-4402-9d2e-974d0c630649)
  
  （3）在（1）的基础上开启KV Cache量化
  
  (quant_policy=4,use_context_fmha=0)
  ```
  python benchmark.py
  ```
  推理引擎turbomind 耗时 11.83秒  吞吐量 127 字/秒
  ![image](https://github.com/xiaomile/InternLM-homework/assets/14927720/74281b71-4422-4985-87a1-71c1ed9005ed)
  
  （4）在（2）的基础上开启KV Cache量化
  
  (quant_policy=4,use_context_fmha=0)
  ```
  python benchmark.py ./workspace_quant
  ```
  推理引擎turbomind 耗时 7.78秒  吞吐量 231 字/秒
  ![image](https://github.com/xiaomile/InternLM-homework/assets/14927720/9c473213-30e2-4304-963c-8a21640eb87a)
  
  （5）使用Huggingface推理
  ```
  python benchmark.py /root/share/temp/model_repos/internlm-chat-7b/ hf
  ```
  推理引擎hf 耗时 55.54秒  吞吐量 50 字/秒
  ![image](https://github.com/xiaomile/InternLM-homework/assets/14927720/be08ef99-7446-4b6e-89dc-5027e56f5d9f)
  
  | 推理引擎| 量化 | 耗时(秒) | 吞吐量(字/秒) |
  | :------:| :---:|:----:|:----:|
  | turbomind | 否 | 11.49 |131 |
  | turbomind | w4A16量化 | 7.62 | 236 |
  | turbomind | k/v cache量化 | 11.83 | 127 |
  | turbomind | w4A16+k/v cache量化 | 7.78 | 231 |
  | huggingface | 否 | 55.54 | 50 |
  
  结论，4bit最快，其次是turbomind 的原模型， 原模型的hf最慢。 kv都会比原来基础上慢一些。
