import json
import os
import re
from difflib import get_close_matches
from collections import Counter
import torch # 确保torch已导入
from transformers import AutoTokenizer, AutoModelForCausalLM
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# --- 配置参数 ---
MODEL_NAME = "/model/zhn/chatglm3-6b"  # 修改为你的模型路径
TRAIN_FILE_PATH = "/home/jgy/thesis-experiment/data/glm4-r8-data/02_ACE05_random/04_no_trigger_table/train.json" # 修改为你的训练数据路径
OUTPUT_DIR = "/home/jgy/thesis-experiment/data/60data-ace" # 输出分层数据的目录
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_PREDICTIONS_PER_SAMPLE = 10 # 每个样本预测的次数
TEMPERATURE_FOR_SAMPLING = 0.7 # 用于模型输出多样性的温度，如果你的model.chat支持

# 预定义的事件类别
DEFINED_LABELS = ['declare bankruptcy', 'transfer ownership', 'transfer money', 'marry',
                  'transport', 'die', 'phone write', 'arrest jail', 'convict', 'sentence', 'sue',
                  'end organization', 'start organization', 'end position', 'start position', 'meet',
                  'elect', 'attack', 'injure', 'born', 'fine', 'release parole', 'charge indict', 'extradite',
                  'trial hearing', 'demonstrate', 'divorce', 'nominate', 'appeal', 'pardon', 'execute', 'acquit',
                  'merge organization']

# --- 模型加载 ---
print(f"Loading model from: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True, torch_dtype=torch.float16) # 使用float16加速
model = model.to(DEVICE)
model.eval()
print(f"Model loaded on {DEVICE}.")

# --- 辅助函数 ---
def build_label_variants(defined_labels):
    label_variants_map = {}
    for label in defined_labels:
        variants = [label.lower()]
        # 你可以根据观察到的模型输出，手动添加更多变体
        # 例如，如果模型经常输出 "charge indictment" 而你的标签是 "charge indict"
        if label == "charge indict":
            variants.append("charge indictment")
            variants.append("indictment") # 更短的变体
            variants.append("indict")
        if label == "phone write":
            variants.append("letter writing")
            variants.append("write")
        if label == "arrest jail":
            variants.append("jail")
        # 简单处理复数或动名词等（更复杂的需要词形还原）
        if label.lower().endswith('s') and len(label) > 1:
             variants.append(label.lower()[:-1])
        if label.lower().endswith('ing') and len(label) > 3:
            variants.append(label.lower()[:-3])


        for v_raw in variants:
            v = v_raw.strip()
            if v: #确保变体非空
                 label_variants_map[v] = label
    return label_variants_map

LABEL_VARIANTS_MAP = build_label_variants(DEFINED_LABELS)

def extract_label_from_response_improved(response_text, predefined_labels, label_variants_map):
    if not response_text or not isinstance(response_text, str) or response_text.strip() == "":
        return None

    response_lower = response_text.lower().strip()

    # 1. 直接完全匹配 (大小写不敏感)
    for label in predefined_labels:
        if label.lower() == response_lower:
            return label
    
    # 2. 检查是否是明确的拒绝或无关回答
    rejection_phrases = [
        "none of the provided categories", "no suitable category", "does not fit any",
        "not related to any", "cannot be categorized", "n/a", "not applicable",
        "i'm sorry, but", "unable to determine", "difficult to classify"
    ]
    for phrase in rejection_phrases:
        if phrase in response_lower:
            return "None" # 特殊标记，表示模型明确拒绝

    # 3. 匹配变体 (优先匹配更长的、更精确的短语)
    # 按长度倒序排序key，这样 "charge indict" 会比 "indict" 先匹配
    sorted_variant_keys = sorted(label_variants_map.keys(), key=len, reverse=True)
    for variant_key in sorted_variant_keys:
        # 使用正则表达式确保是单词边界的匹配
        # re.escape处理特殊字符，例如 'end position' 中的空格
        try:
            if re.search(r'\b' + re.escape(variant_key) + r'\b', response_lower, re.IGNORECASE):
                return label_variants_map[variant_key]
        except re.error: # 处理 re.escape 可能的问题
            if variant_key in response_lower: # 退化为简单包含
                 return label_variants_map[variant_key]


    # 4. (可选) 模糊匹配 - 谨慎使用，可能引入错误
    # close_matches = get_close_matches(response_lower, [l.lower() for l in predefined_labels], n=1, cutoff=0.8) # 提高cutoff
    # if close_matches:
    #     matched_lower = close_matches[0]
    #     for original_label in predefined_labels:
    #         if original_label.lower() == matched_lower:
    #             return original_label
                
    return None # 默认返回None，表示无法明确提取


# --- 主处理逻辑 ---
def process_data_and_layer():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Loading training data from: {TRAIN_FILE_PATH}")
    with open(TRAIN_FILE_PATH, "r", encoding="utf-8") as fh:
        all_data = json.load(fh)
    print(f"Loaded {len(all_data)} samples from training data.")

    # 构建Prompt模板
    categories_string_for_prompt = ", ".join(DEFINED_LABELS)
    prompt_template = """Given the text, output ONLY ONE category from the provided list that best describes the event.
Category list: {categories_string}
Text: {sentence}
Selected Category:"""

    results_for_layering = []
    
    for idx, original_sample_obj in enumerate(all_data):
        # 原始样本的json结构是 "messages": [{"role":"system", ...}, {"role":"user", ...}, {"role":"assistant", ...}]
        # 我们需要的是原始的输入和输出，以及用于重新构建的原始样本对象
        try:
            sentence = original_sample_obj["messages"][1]["content"]
            true_label = original_sample_obj["messages"][2]["content"].split('#')[0] #"attack#those_attacker;Iraq_place;" 获取#前的类别
        except (IndexError, KeyError) as e:
            print(f"Skipping sample {idx} due to malformed structure: {e}")
            continue

        if true_label not in DEFINED_LABELS:
            print(f"Warning: True label '{true_label}' for sample {idx} is not in DEFINED_LABELS. Skipping.")
            continue

        predictions_for_this_sample_raw = []
        predictions_for_this_sample_extracted = []
        
        current_input_prompt = prompt_template.format(categories_string=categories_string_for_prompt, sentence=sentence)

        for _ in range(NUM_PREDICTIONS_PER_SAMPLE):
            try:
                # 确保你的model.chat能够接受temperature参数以引入随机性
                # 如果不支持，并且输出是确定性的，多次运行意义不大
                # response, history = model.chat(tokenizer, current_input_prompt, history=[], temperature=TEMPERATURE_FOR_SAMPLING)
                # 尝试不带 temperature，依赖模型本身的随机性（如果有的话）
                response, history = model.chat(tokenizer, current_input_prompt, history=[])
                raw_response_str = str(response)
            except Exception as e:
                print(f"Error during model.chat for sample {idx}: {e}")
                raw_response_str = "" # 发生错误时给一个空响应

            predictions_for_this_sample_raw.append(raw_response_str)
            extracted_label = extract_label_from_response_improved(raw_response_str, DEFINED_LABELS, LABEL_VARIANTS_MAP)
            predictions_for_this_sample_extracted.append(extracted_label)
            
        valid_extracted_predictions = [p for p in predictions_for_this_sample_extracted if p is not None and p != "None"]
        
        mastery_score = 0.0
        always_rejected_or_invalid = True

        if valid_extracted_predictions:
            always_rejected_or_invalid = False
            correct_valid_predictions = sum(1 for p in valid_extracted_predictions if p == true_label)
            mastery_score = correct_valid_predictions / NUM_PREDICTIONS_PER_SAMPLE
        elif any(p == "None" for p in predictions_for_this_sample_extracted): # 如果全是None或无效，但至少有一次是"None" (明确拒绝)
            always_rejected_or_invalid = True # 保持为True，但mastery_score仍为0


        results_for_layering.append({
            "original_sample_json": original_sample_obj, # 保存原始json对象
            "text": sentence, # 为方便调试也单独存一份
            "true_label": true_label,
            "raw_responses": predictions_for_this_sample_raw, 
            "extracted_labels": predictions_for_this_sample_extracted,
            "mastery_score": mastery_score,
            "is_problematic_output": always_rejected_or_invalid # 更明确的标志
        })

        if (idx + 1) % 10 == 0: # 每10个样本打印一次进度
            print(f"Processed sample {idx + 1}/{len(all_data)}. Sample: '{sentence[:30]}...' -> True: {true_label}, Mastery: {mastery_score:.2f}, Problematic: {always_rejected_or_invalid}")
            print(f"  Extracted (first 3 of 10): {predictions_for_this_sample_extracted[:3]}")


    # --- 数据分层 ---
    print("\n--- Layering Data ---")
    layered_samples_by_level = {0: [], 1: [], 2: [], 3: []}
    for res_item in results_for_layering:
        score = res_item["mastery_score"]
        is_problematic = res_item["is_problematic_output"]
        
        # 我们保留原始的JSON message结构，方便后续直接用于微调
        sample_to_save = res_item["original_sample_json"]
        # 可以在这里添加一些元数据到原始样本中，如果需要的话
        # sample_to_save["metadata_mastery_score"] = score
        # sample_to_save["metadata_is_problematic"] = is_problematic
        # sample_to_save["metadata_extracted_labels_for_debug"] = res_item["extracted_labels"]


        level = -1
        if is_problematic: # 优先处理有问题的输出，归为Level 0
            level = 0
        elif score <= 0.25: # (0, 0.25] -> Level 0 (非常困难，几乎总是错)
            level = 0
        elif 0.25 < score <= 0.5: # (0.25, 0.5] -> Level 1 (较困难，一半时间左右会错)
            level = 1
        elif 0.5 < score <= 0.75: # (0.5, 0.75] -> Level 2 (中等，多数时候对，但不够稳定)
            level = 2
        elif score > 0.75: # (0.75, 1.0] -> Level 3 (较容易，掌握较好)
            level = 3
        
        if level != -1:
            layered_samples_by_level[level].append(sample_to_save)
            
    for level, samples_in_level in layered_samples_by_level.items():
        print(f"Level {level}: {len(samples_in_level)} samples")
        output_path = os.path.join(OUTPUT_DIR, f"level_{level}.json")
        with open(output_path, "w", encoding="utf-8") as f_out:
            json.dump(samples_in_level, f_out, indent=4, ensure_ascii=False)
        print(f"Saved Level {level} data to {output_path}")

        if samples_in_level:
            true_labels_dist = Counter(s["messages"][2]["content"] for s in samples_in_level)
            print(f"  Top 5 true labels in Level {level}: {true_labels_dist.most_common(5)}")


if __name__ == "__main__":
    process_data_and_layer()