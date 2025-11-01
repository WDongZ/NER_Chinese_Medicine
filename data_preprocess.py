import json

def parse_conll(file_path):
    """
    解析CoNLL格式文件，返回句子列表
    每个句子是 [(char, tag), (char, tag), ...]
    """
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    sentence = []
    for line in lines:
        line = line.strip()
        if not line:
            if sentence:
                sentences.append(sentence)
                sentence = []
        else:
            try:
                char, tag = line.split()
                sentence.append((char, tag))
            except:
                continue
    if sentence:
        sentences.append(sentence)
    return sentences


def extract_entities(sentence):
    """
    从句子[(char, tag), ...]中提取实体
    返回：{"实体类别": [实体1, 实体2, ...]}
    """
    entities = {}
    entity = ""
    entity_type = None

    for char, tag in sentence:
        if tag.startswith("B-"):
            # 新实体开始
            if entity and entity_type:
                entities.setdefault(entity_type, []).append(entity)
            entity = char
            entity_type = tag[2:]
        elif tag.startswith("I-") and entity_type == tag[2:]:
            entity += char
        else:
            # 非实体或标签不连续
            if entity and entity_type:
                entities.setdefault(entity_type, []).append(entity)
            entity = ""
            entity_type = None

    # 收尾
    if entity and entity_type:
        entities.setdefault(entity_type, []).append(entity)

    return entities


def convert_to_sft(sentences, output_file):
    """
    转换为SFT JSONL格式
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for sent in sentences:
            text = "".join([c for c, _ in sent])
            entities = extract_entities(sent)
            if not entities:
                continue  # 跳过无实体样本

            data = {
                "instruction": "请识别以下中医药文本中的命名实体，并指出它们的类别。",
                "input": text,
                "output": entities
            }
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    input_file_dev = "./data/medical_dev.txt"
    input_file_test = "./data/medical_test.txt"
    input_file_train = "./data/medical_train.txt"
    output_file_dev = "dev.jsonl"      
    output_file_test = "test.jsonl"      
    output_file_train = "train.jsonl"      

    s_dev = parse_conll(input_file_dev)
    print(f"共读取 {len(s_dev)} 个句子。")
    s_test = parse_conll(input_file_test)
    print(f"共读取 {len(s_test)} 个句子。")
    s_train = parse_conll(input_file_train)
    print(f"共读取 {len(s_train)} 个句子。")
    convert_to_sft(s_dev, output_file_dev)
    convert_to_sft(s_test, output_file_test)
    convert_to_sft(s_train, output_file_train)
