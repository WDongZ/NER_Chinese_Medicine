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

python eval.py

	•	自动计算词级 F1 / Precision / Recall

推理示例
```python
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
```
## 模型评测结果

下表展示了基于 **Qwen2.5-7B + QLoRA** 的中医药命名实体识别模型在测试集上的表现，  
指标为 **词级（word-level）F1 / Precision / Recall**。

| 实体类别 | Precision | Recall | F1 Score |
|:----------|-----------:|--------:|----------:|
| 中药 | 0.7812 | 0.7936 | **0.7874** |
| 临床表现 | 0.4999 | 0.7634 | **0.6042** |
| 西医诊断 | 0.6825 | 0.6231 | **0.6515** |
| 中医证候 | 0.6103 | 0.6811 | **0.6438** |
| 西医治疗 | 0.4727 | 0.7027 | **0.5652** |
| 方剂 | 0.4520 | 0.4074 | **0.4285** |
| 中医治疗 | 0.6734 | 0.4342 | **0.5279** |
| 其他治疗 | 0.2727 | 0.3749 | **0.3157** |
| 中医治则 | 0.3448 | 0.3124 | **0.3278** |
| 中医诊断 | 0.5714 | 0.1666 | **0.2580** |

---

### 模型性能分析

- **最佳类别**：中药类实体（F1≈0.79），识别效果稳定，说明模型对药物命名有较强的理解能力。  
- **表现良好**：中医证候、西医诊断类（F1≈0.64–0.65），反映模型能较好理解医学症状与诊断。  
- **中等表现**：临床表现、西医治疗类（F1≈0.56–0.60），存在召回率偏高但精确率不足的情况。  
- **待提升类别**：中医治则、中医诊断、其他治疗（F1 < 0.35），推测因样本量不足或语义模糊导致。  

---
