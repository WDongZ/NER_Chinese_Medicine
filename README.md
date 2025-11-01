# 中医药命名实体识别 (NER_Chinese_MED)

## 项目简介
本项目基于 **Qwen2.5-7B + QLoRA** 对中医药领域文本进行命名实体识别（NER）。  
识别实体类别包括：中药、方剂、临床表现、中医诊断、西医治疗等。  
可应用于信息抽取、问答系统、文本摘要和检索等任务。

## 数据集
- 训练集 / 验证集 / 测试集共约 6000 条标注数据  
- 数据格式为 BIO 标注，例如：

猪 B-中药
苓 I-中药
全 B-临床表现
身 I-临床表现

- 数据文件：
  - `train.jsonl` 训练集
  - `dev.jsonl` 验证集
  - `test.jsonl` 测试集

## 模型
- 基座模型：Qwen2.5-7B  
- 微调方法：QLoRA / LoRA  
- 微调输出模型路径：`./tcm_ner_qlora_model/`  
- 使用 PEFT 技术节省显存  

## 使用方法

数据预处理

python preprocess.py

模型训练

python train.py

	•	支持 LoRA / QLoRA 微调
	•	可使用 SwanLab 可视化训练进程

模型评测

python eval_tcm_ner_swanlab.py

	•	自动计算词级 F1 / Precision / Recall
	•	SwanLab 可视化评测进程

推理示例

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("./tcm_ner_qlora_model")
model = AutoModelForCausalLM.from_pretrained("./tcm_ner_qlora_model")
model.config.pad_token_id = tokenizer.eos_token_id

input_text = "请识别下面文本中的中药实体：茯苓猪苓"
encoding = tokenizer(input_text, return_tensors="pt", padding=True).to(model.device)
outputs = model.generate(
    input_ids=encoding["input_ids"],
    attention_mask=encoding["attention_mask"],
    max_new_tokens=256,
    pad_token_id=model.config.pad_token_id
)
pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(pred_text)

模型评测示例
	•	中药 F1: ~0.787
	•	临床表现 F1: ~0.604
	•	中医诊断 F1: ~0.258

F1 值为词级指标，Precision / Recall 同步计算，可通过 SwanLab 可视化查看各实体类别表现。
