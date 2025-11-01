import json
import re
from collections import defaultdict

import torch
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model
import swanlab

# ======================
# 1. 数据处理函数
# ======================
def format_prompt(example):
    """拼接instruction + input + output为训练文本"""
    instruction = example["instruction"]
    text_input = example["input"]
    output = json.dumps(example["output"], ensure_ascii=False)
    return f"{instruction}\n{text_input}\n答：{output}"


def tokenize_fn(example, tokenizer, max_length=1024):
    prompt = format_prompt(example)
    tokenized = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


# ======================
# 2. 实体提取与F1计算
# ======================
def extract_entities_from_text(output_text):
    """
    从模型生成的文本中解析出实体，假设输出格式是JSON
    """
    entities = defaultdict(list)
    try:
        json_part = re.search(r"\{.*\}", output_text, re.S)
        if json_part:
            parsed = json.loads(json_part.group())
            for k, v in parsed.items():
                if isinstance(v, list):
                    entities[k] = v
    except Exception:
        pass
    return entities


def compute_f1_per_type(y_true_all, y_pred_all):
    result = {}
    for etype in sorted(set(y_true_all.keys()) | set(y_pred_all.keys())):
        y_true = y_true_all.get(etype, [])
        y_pred = y_pred_all.get(etype, [])
        y_true_set, y_pred_set = set(y_true), set(y_pred)
        tp = len(y_true_set & y_pred_set)
        fp = len(y_pred_set - y_true_set)
        fn = len(y_true_set - y_pred_set)
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        result[etype] = {"precision": p, "recall": r, "f1": f1}
    return result


# ======================
# 3. 主训练函数
# ======================
def main():
    # ========= 数据加载 =========
    dataset = load_dataset(
        "json",
        data_files={
            "train": "train.jsonl",
            "validation": "dev.jsonl",
            "test": "test.jsonl"
        }
    )

    model_name = "Qwen/Qwen2.5-7B"

    # ========= 模型与Tokenizer =========
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # ========= Tokenize =========
    tokenized_datasets = dataset.map(
        lambda x: tokenize_fn(x, tokenizer),
        batched=False
    )

    # ========= SwanLab 初始化 =========
    run = swanlab.init(
        project="tcm-ner-qlora",
        experiment_name="Qwen2.5-7B-QLoRA",
        description="Fine-tune Qwen2.5-7B on TCM NER dataset with LoRA",
    )

    # ========= 训练参数 =========
    training_args = TrainingArguments(
        output_dir="./outputs_tcm_ner",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        fp16=True,
        do_eval=True,
        eval_steps=200,
        save_steps=200,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        logging_steps=10,
        save_total_limit=3,
        report_to=["swanlab"],  # ✅ 把日志推送到 SwanLab
        run_name="Qwen2.5-7B-QLoRA"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
    )

    trainer.train()
    model.save_pretrained("./tcm_ner_qlora_model")
    tokenizer.save_pretrained("./tcm_ner_qlora_model")

    print("✅ 模型训练完成，已保存至 ./tcm_ner_qlora_model")

    # ========= 生成验证集预测 =========
    print("\n>>> 开始生成验证集预测并计算词级F1...")

    entity_types = set()
    y_true_all = defaultdict(list)
    y_pred_all = defaultdict(list)

    for sample in dataset["test"]:
        input_text = f"{sample['instruction']}\n{sample['input']}\n答："
        true_entities = sample["output"]

        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
        outputs = model.generate(
            input_ids,
            max_new_tokens=256,
            temperature=0.0,
            do_sample=False
        )
        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred_entities = extract_entities_from_text(pred_text)

        # 记录真实与预测
        for etype, ents in true_entities.items():
            entity_types.add(etype)
            y_true_all[etype].extend(ents)
        for etype, ents in pred_entities.items():
            entity_types.add(etype)
            y_pred_all[etype].extend(ents)

    # ========= 计算词级F1 =========
    print("\n📊 每类实体F1:")
    f1_result = compute_f1_per_type(y_true_all, y_pred_all)
    for etype, scores in f1_result.items():
        print(f"{etype:10s} | P={scores['precision']:.4f} R={scores['recall']:.4f} F1={scores['f1']:.4f}")
        run.log({
            f"F1/{etype}": scores["f1"],
            f"Precision/{etype}": scores["precision"],
            f"Recall/{etype}": scores["recall"],
        })

    run.finish()


if __name__ == "__main__":
    main()
