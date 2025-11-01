import json
import re
from collections import defaultdict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import swanlab

# =========================
# 实体提取函数
# =========================
def extract_entities_from_text(output_text):
    """
    从模型生成的文本中解析出实体，假设输出是JSON格式
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

# =========================
# 计算词级F1
# =========================
def compute_f1(y_true_all, y_pred_all):
    f1_results = {}
    all_entity_types = set(y_true_all.keys()) | set(y_pred_all.keys())
    for etype in all_entity_types:
        y_true = set(y_true_all.get(etype, []))
        y_pred = set(y_pred_all.get(etype, []))
        tp = len(y_true & y_pred)
        fp = len(y_pred - y_true)
        fn = len(y_true - y_pred)
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        f1_results[etype] = {"precision": precision, "recall": recall, "f1": f1}
    return f1_results

# =========================
# 主函数
# =========================
def main():
    # SwanLab 初始化
    run = swanlab.init(
        project="tcm-ner-qlora",
        experiment_name="NER_Eval",
        description="Evaluate TCM NER QLoRA model with attention_mask and pad_token fix",
    )

    # =========================
    # 加载模型和 tokenizer
    # =========================
    model_dir = "./tcm_ner_qlora_model"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    model.eval()
    # 设置 pad_token_id 避免警告
    model.config.pad_token_id = tokenizer.eos_token_id

    # =========================
    # 加载测试集
    # =========================
    test_dataset = load_dataset("json", data_files={"test": "test.jsonl"})["test"]

    # =========================
    # 推理 & 统计 F1
    # =========================
    y_true_all = defaultdict(list)
    y_pred_all = defaultdict(list)

    for idx, sample in enumerate(test_dataset):
        input_text = f"{sample['instruction']}\n{sample['input']}\n答："
        true_entities = sample["output"]

        # tokenize，生成 attention_mask
        encoding = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
                max_new_tokens=256,
                temperature=0.0,
                do_sample=False,
                pad_token_id=model.config.pad_token_id
            )

        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred_entities = extract_entities_from_text(pred_text)

        # 收集实体
        for etype, ents in true_entities.items():
            y_true_all[etype].extend(ents or [])
        for etype, ents in pred_entities.items():
            y_pred_all[etype].extend(ents or [])

        # 可视化进程
        if (idx + 1) % 10 == 0:
            run.log({"progress": (idx + 1) / len(test_dataset)})

    # =========================
    # 计算 F1 并打印
    # =========================
    f1_results = compute_f1(y_true_all, y_pred_all)
    print("\n📊 测试集每类实体F1:")
    for etype, scores in f1_results.items():
        print(f"{etype:10s} | P={scores['precision']:.4f} R={scores['recall']:.4f} F1={scores['f1']:.4f}")
        run.log({
            f"F1/{etype}": scores["f1"],
            f"Precision/{etype}": scores["precision"],
            f"Recall/{etype}": scores["recall"]
        })

    run.finish()
    print("✅ 评估完成，SwanLab 可视化结束。")

if __name__ == "__main__":
    main()
