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

评测结果
```

dataset                                         version    metric         mode      opencompass.models.huggingface.HuggingFace_model_internlm2-chat-7b
----------------------------------------------  ---------  -------------  ------  --------------------------------------------------------------------
ceval-computer_network                          db9ce2     accuracy       gen                                                                    47.37
ceval-operating_system                          1c2571     accuracy       gen                                                                    57.89
ceval-computer_architecture                     a74dad     accuracy       gen                                                                    42.86
ceval-college_programming                       4ca32a     accuracy       gen                                                                    51.35
ceval-college_physics                           963fa8     accuracy       gen                                                                    36.84
ceval-college_chemistry                         e78857     accuracy       gen                                                                    33.33
ceval-advanced_mathematics                      ce03e2     accuracy       gen                                                                    15.79
ceval-probability_and_statistics                65e812     accuracy       gen                                                                    27.78
ceval-discrete_mathematics                      e894ae     accuracy       gen                                                                    18.75
ceval-electrical_engineer                       ae42b9     accuracy       gen                                                                    40.54
ceval-metrology_engineer                        ee34ea     accuracy       gen                                                                    58.33
ceval-high_school_mathematics                   1dc5bf     accuracy       gen                                                                    44.44
ceval-high_school_physics                       adf25f     accuracy       gen                                                                    47.37
ceval-high_school_chemistry                     2ed27f     accuracy       gen                                                                    52.63
ceval-high_school_biology                       8e2b9a     accuracy       gen                                                                    26.32
ceval-middle_school_mathematics                 bee8d5     accuracy       gen                                                                    26.32
ceval-middle_school_biology                     86817c     accuracy       gen                                                                    66.67
ceval-middle_school_physics                     8accf6     accuracy       gen                                                                    57.89
ceval-middle_school_chemistry                   167a15     accuracy       gen                                                                    95
ceval-veterinary_medicine                       b4e08d     accuracy       gen                                                                    39.13
ceval-college_economics                         f3f4e6     accuracy       gen                                                                    47.27
ceval-business_administration                   c1614e     accuracy       gen                                                                    51.52
ceval-marxism                                   cf874c     accuracy       gen                                                                    84.21
ceval-mao_zedong_thought                        51c7a4     accuracy       gen                                                                    70.83
ceval-education_science                         591fee     accuracy       gen                                                                    72.41
ceval-teacher_qualification                     4e4ced     accuracy       gen                                                                    79.55
ceval-high_school_politics                      5c0de2     accuracy       gen                                                                    21.05
ceval-high_school_geography                     865461     accuracy       gen                                                                    47.37
ceval-middle_school_politics                    5be3e7     accuracy       gen                                                                    42.86
ceval-middle_school_geography                   8a63be     accuracy       gen                                                                    58.33
ceval-modern_chinese_history                    fc01af     accuracy       gen                                                                    65.22
ceval-ideological_and_moral_cultivation         a2aa4a     accuracy       gen                                                                    89.47
ceval-logic                                     f5b022     accuracy       gen                                                                    54.55
ceval-law                                       a110a1     accuracy       gen                                                                    41.67
ceval-chinese_language_and_literature           0f8b68     accuracy       gen                                                                    56.52
ceval-art_studies                               2a1300     accuracy       gen                                                                    69.7
ceval-professional_tour_guide                   4e673e     accuracy       gen                                                                    86.21
ceval-legal_professional                        ce8787     accuracy       gen                                                                    43.48
ceval-high_school_chinese                       315705     accuracy       gen                                                                    68.42
ceval-high_school_history                       7eb30a     accuracy       gen                                                                    75
ceval-middle_school_history                     48ab4a     accuracy       gen                                                                    68.18
ceval-civil_servant                             87d061     accuracy       gen                                                                    55.32
ceval-sports_science                            70f27b     accuracy       gen                                                                    73.68
ceval-plant_protection                          8941f9     accuracy       gen                                                                    77.27
ceval-basic_medicine                            c409d6     accuracy       gen                                                                    63.16
ceval-clinical_medicine                         49e82d     accuracy       gen                                                                    45.45
ceval-urban_and_rural_planner                   95b885     accuracy       gen                                                                    58.7
ceval-accountant                                002837     accuracy       gen                                                                    44.9
ceval-fire_engineer                             bc23f5     accuracy       gen                                                                    38.71
ceval-environmental_impact_assessment_engineer  c64e2d     accuracy       gen                                                                    45.16
ceval-tax_accountant                            3a5e3c     accuracy       gen                                                                    51.02
ceval-physician                                 6e277d     accuracy       gen                                                                    51.02
ceval-stem                                      -          naive_average  gen                                                                    44.33
ceval-social-science                            -          naive_average  gen                                                                    57.54
ceval-humanities                                -          naive_average  gen                                                                    65.31
ceval-other                                     -          naive_average  gen                                                                    54.94
ceval-hard                                      -          naive_average  gen                                                                    34.62
ceval                                           -          naive_average  gen                                                                    53.55
```

进阶作业

使用 OpenCompass 评测 InternLM2-Chat-7B 模型使用 LMDeploy 0.2.0 部署后在 C-Eval 数据集上的性能

安装lmdeploy 0.2.0
```
pip install lmdeploy==0.2.0
```

转化模型为turbomind格式
```
cd /root/code/opencompass/model
lmdeploy convert internlm2-chat-7b ./model/internlm2-chat-7b
```

编写评测demo
在configs目录下新建eval_internlm2_chat_turbomind_api.py,可参考eval_internlm_chat_turbomind_api.py
```python
### eval_internlm2_chat_turbomind_api.py
from mmengine.config import read_base
from opencompass.models.turbomind_api import TurboMindAPIModel

with read_base():
    # choose a list of datasets
    from .datasets.ceval.ceval_gen_5f30c7 import ceval_datasets
    # and output the results in a choosen format
    from .summarizers.medium import summarizer

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])


meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<|User|>:', end='\n'),
        dict(role='BOT', begin='<|Bot|>:', end='<eoa>\n', generate=True),
    ],
    eos_token_id=103028)

models = [
    dict(
        type=TurboMindAPIModel,
        abbr='internlm2-chat-7b-turbomind',
        path="./model/workspace",
        api_addr='http://0.0.0.0:23333',
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        meta_template=meta_template,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]

```

用lmdeploy部署internlm2-chat-7b，并作为服务端启动
```
lmdeploy serve api_server /root/code/opencompass/model/workspace --server-name 0.0.0.0 --server-port 23333 --max-batch-size 64 --tp 1
```


开始评测
```
cd /root/code/opencompass
python run.py configs/eval_internlm2_chat_turbomind_api.py
```

![image](https://github.com/xiaomile/InternLM-homework/assets/14927720/00a0c0ac-5ce8-42a2-a3d3-e9520159de44)

评测完成
![image](https://github.com/xiaomile/InternLM-homework/assets/14927720/7641332f-cca2-429f-a3b5-b8226b78c607)

评测结果
```
tabulate format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
dataset                                 version    metric         mode    internlm2-chat-7b-turbomind
--------------------------------------  ---------  -------------  ------  -----------------------------
--------- 考试 Exam ---------           -          -              -       -
ceval                                   -          naive_average  gen     59.42
agieval                                 -          -              -       -
mmlu                                    -          -              -       -
GaokaoBench                             -          -              -       -
ARC-c                                   -          -              -       -
--------- 语言 Language ---------       -          -              -       -
WiC                                     -          -              -       -
summedits                               -          -              -       -
chid-dev                                -          -              -       -
afqmc-dev                               -          -              -       -
bustm-dev                               -          -              -       -
cluewsc-dev                             -          -              -       -
WSC                                     -          -              -       -
winogrande                              -          -              -       -
flores_100                              -          -              -       -
--------- 知识 Knowledge ---------      -          -              -       -
BoolQ                                   -          -              -       -
commonsense_qa                          -          -              -       -
nq                                      -          -              -       -
triviaqa                                -          -              -       -
--------- 推理 Reasoning ---------      -          -              -       -
cmnli                                   -          -              -       -
ocnli                                   -          -              -       -
ocnli_fc-dev                            -          -              -       -
AX_b                                    -          -              -       -
AX_g                                    -          -              -       -
CB                                      -          -              -       -
RTE                                     -          -              -       -
story_cloze                             -          -              -       -
COPA                                    -          -              -       -
ReCoRD                                  -          -              -       -
hellaswag                               -          -              -       -
piqa                                    -          -              -       -
siqa                                    -          -              -       -
strategyqa                              -          -              -       -
math                                    -          -              -       -
gsm8k                                   -          -              -       -
TheoremQA                               -          -              -       -
openai_humaneval                        -          -              -       -
mbpp                                    -          -              -       -
bbh                                     -          -              -       -
--------- 理解 Understanding ---------  -          -              -       -
C3                                      -          -              -       -
CMRC_dev                                -          -              -       -
DRCD_dev                                -          -              -       -
MultiRC                                 -          -              -       -
race-middle                             -          -              -       -
race-high                               -          -              -       -
openbookqa_fact                         -          -              -       -
csl_dev                                 -          -              -       -
lcsts                                   -          -              -       -
Xsum                                    -          -              -       -
eprstmt-dev                             -          -              -       -
lambada                                 -          -              -       -
tnews-dev                               -          -              -       -
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
```
```
raw format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
-------------------------------
Model: internlm2-chat-7b-turbomind
ceval-computer_network: {'accuracy': 57.89473684210527}
ceval-operating_system: {'accuracy': 57.89473684210527}
ceval-computer_architecture: {'accuracy': 57.14285714285714}
ceval-college_programming: {'accuracy': 62.16216216216216}
ceval-college_physics: {'accuracy': 47.368421052631575}
ceval-college_chemistry: {'accuracy': 33.33333333333333}
ceval-advanced_mathematics: {'accuracy': 26.31578947368421}
ceval-probability_and_statistics: {'accuracy': 44.44444444444444}
ceval-discrete_mathematics: {'accuracy': 31.25}
ceval-electrical_engineer: {'accuracy': 43.24324324324324}
ceval-metrology_engineer: {'accuracy': 58.333333333333336}
ceval-high_school_mathematics: {'accuracy': 27.77777777777778}
ceval-high_school_physics: {'accuracy': 42.10526315789473}
ceval-high_school_chemistry: {'accuracy': 42.10526315789473}
ceval-high_school_biology: {'accuracy': 36.84210526315789}
ceval-middle_school_mathematics: {'accuracy': 47.368421052631575}
ceval-middle_school_biology: {'accuracy': 80.95238095238095}
ceval-middle_school_physics: {'accuracy': 63.1578947368421}
ceval-middle_school_chemistry: {'accuracy': 95.0}
ceval-veterinary_medicine: {'accuracy': 47.82608695652174}
ceval-college_economics: {'accuracy': 50.90909090909091}
ceval-business_administration: {'accuracy': 54.54545454545454}
ceval-marxism: {'accuracy': 84.21052631578947}
ceval-mao_zedong_thought: {'accuracy': 70.83333333333334}
ceval-education_science: {'accuracy': 79.3103448275862}
ceval-teacher_qualification: {'accuracy': 77.27272727272727}
ceval-high_school_politics: {'accuracy': 89.47368421052632}
ceval-high_school_geography: {'accuracy': 68.42105263157895}
ceval-middle_school_politics: {'accuracy': 76.19047619047619}
ceval-middle_school_geography: {'accuracy': 66.66666666666666}
ceval-modern_chinese_history: {'accuracy': 73.91304347826086}
ceval-ideological_and_moral_cultivation: {'accuracy': 84.21052631578947}
ceval-logic: {'accuracy': 50.0}
ceval-law: {'accuracy': 41.66666666666667}
ceval-chinese_language_and_literature: {'accuracy': 69.56521739130434}
ceval-art_studies: {'accuracy': 72.72727272727273}
ceval-professional_tour_guide: {'accuracy': 79.3103448275862}
ceval-legal_professional: {'accuracy': 47.82608695652174}
ceval-high_school_chinese: {'accuracy': 52.63157894736842}
ceval-high_school_history: {'accuracy': 75.0}
ceval-middle_school_history: {'accuracy': 81.81818181818183}
ceval-civil_servant: {'accuracy': 65.95744680851064}
ceval-sports_science: {'accuracy': 68.42105263157895}
ceval-plant_protection: {'accuracy': 77.27272727272727}
ceval-basic_medicine: {'accuracy': 63.1578947368421}
ceval-clinical_medicine: {'accuracy': 50.0}
ceval-urban_and_rural_planner: {'accuracy': 63.04347826086957}
ceval-accountant: {'accuracy': 46.93877551020408}
ceval-fire_engineer: {'accuracy': 54.83870967741935}
ceval-environmental_impact_assessment_engineer: {'accuracy': 45.16129032258064}
ceval-tax_accountant: {'accuracy': 48.97959183673469}
ceval-physician: {'accuracy': 57.14285714285714}
ceval-stem: {'naive_average': 50.12591254625007}
ceval-social-science: {'naive_average': 71.78333569032297}
ceval-humanities: {'naive_average': 66.24262901172294}
ceval-other: {'naive_average': 58.2648931091204}
ceval-hard: {'naive_average': 36.8375365497076}
ceval: {'naive_average': 59.42181444533667}
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
```
