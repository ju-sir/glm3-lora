import json
import random
from collections import Counter, defaultdict

# --- 配置 ---
LEVEL_FILES = {
    0: "/home/jgy/thesis-experiment/data/60data-ace-onlytype/level_0.json",
    1: "/home/jgy/thesis-experiment/data/60data-ace-onlytype/level_1.json",
    2: "/home/jgy/thesis-experiment/data/60data-ace-onlytype/level_2.json",
    3: "/home/jgy/thesis-experiment/data/60data-ace-onlytype/level_3.json",
}
OUTPUT_SELECTED_FILE = "/home/jgy/thesis-experiment/data/60data-ace-onlytype/selected_60_samples.json"
NUM_TOTAL_SAMPLES_TO_SELECT = 60
DEFINED_LABELS = ['declare bankruptcy', 'transfer ownership', 'transfer money', 'marry',
                  'transport', 'die', 'phone write', 'arrest jail', 'convict', 'sentence', 'sue',
                  'end organization', 'start organization', 'end position', 'start position', 'meet',
                  'elect', 'attack', 'injure', 'born', 'fine', 'release parole', 'charge indict', 'extradite',
                  'trial hearing', 'demonstrate', 'divorce', 'nominate', 'appeal', 'pardon', 'execute', 'acquit',
                  'merge organization']

# --- 加载数据 ---
all_level_data = {}
for level, path in LEVEL_FILES.items():
    try:
        with open(path, "r", encoding="utf-8") as f:
            all_level_data[level] = json.load(f)
        print(f"Loaded {len(all_level_data[level])} samples from Level {level}.")
    except FileNotFoundError:
        print(f"Warning: File for Level {level} not found at {path}. Skipping.")
        all_level_data[level] = []

# --- 辅助数据结构 ---
# 将每个level的数据按类别组织起来，方便按类别采样
# 并记录每个样本的原始索引或唯一标识符，以避免重复选择（如果样本在不同level有副本，虽然这里不应该）
# 这里假设每个样本在分层后只属于一个level，并且json对象本身就是唯一的
samples_by_level_and_category = defaultdict(lambda: defaultdict(list))
for level, samples in all_level_data.items():
    for sample_obj in samples:
        try:
            # true_label = sample_obj["messages"][2]["content"].split('#')[0]
            true_label = sample_obj["messages"][2]["content"]
            if true_label in DEFINED_LABELS:
                # 为避免直接修改原始对象，可以只存索引，或者如果对象不可哈希，则用其他唯一ID
                # 这里我们直接存对象，假设后续选择时会从列表中移除或标记已选
                samples_by_level_and_category[level][true_label].append(sample_obj)
        except (IndexError, KeyError, TypeError):
            print(f"Warning: Could not get true_label from sample: {sample_obj}")
            continue


# --- 选择策略 ---
selected_samples_list = []
selected_samples_identifiers = set() # 用于跟踪已选择的样本，避免重复（如果样本对象是可哈希的或有唯一ID）
                                  # 如果直接从列表中pop，则不需要这个

# 目标分配（可以根据你的分析调整，这里只是一个示例起点）
# 优先 Level 1 和 2，少量 Level 0 和 3
# 这个分配需要灵活调整，因为某些level或类别可能没有足够的样本
target_distribution_by_level = {
    1: 10,  # 目标从Level 1选25个
    2: 35,  # 目标从Level 2选20个
    0: 5,  # 目标从Level 0选10个 (谨慎选择)
    3: 10,   # 目标从Level 3选5个
}
# 确保总和是60，或者在后续逻辑中处理不足的情况

# 为了类别均衡，我们先尝试每个类别都选到一些样本
num_categories = len(DEFINED_LABELS)
# 理想情况下，每个类别至少有 NUM_TOTAL_SAMPLES_TO_SELECT / num_categories 个样本
# 但60 / 33 ≈ 1.8，所以目标是每个类别至少1个，有些可以有2个

# 跟踪每个类别已选择的样本数
category_counts = Counter()

# 打乱类别顺序，避免选择偏向列表前面的类别
shuffled_labels = random.sample(DEFINED_LABELS, len(DEFINED_LABELS))

# 优先顺序：Level 1 -> Level 2 -> Level 0 (非常谨慎) -> Level 3
priority_levels = [1, 2, 0, 3]

# 阶段1: 尝试为每个类别从高优先级Level中至少选一个样本
print("\n--- Stage 1: Ensuring basic category coverage ---")
for label in shuffled_labels:
    if len(selected_samples_list) >= NUM_TOTAL_SAMPLES_TO_SELECT:
        break
    if category_counts[label] > 0: # 如果这个类别已经选过了，暂时跳过，后续再补充
        continue

    selected_for_this_label = False
    for level in priority_levels: # 按优先级尝试Level
        if label in samples_by_level_and_category[level] and samples_by_level_and_category[level][label]:
            # 从该level该类别中随机选一个
            # 为了避免重复选择同一个样本对象，如果直接pop，需要注意原始列表会被修改
            # 更安全的方式是复制列表或使用索引
            available_samples = samples_by_level_and_category[level][label]
            if available_samples: # 确保列表非空
                # 为了演示，我们随机选一个，并假设我们不会重复选（实际中可能需要更复杂的去重）
                # 如果要严格去重，需要使用样本的唯一标识符
                chosen_sample = random.choice(available_samples)
                
                # 简单去重：假设样本内容可以作为粗略的唯一标识
                sample_text_for_id = chosen_sample["messages"][1]["content"]
                if sample_text_for_id not in selected_samples_identifiers:
                    selected_samples_list.append(chosen_sample)
                    selected_samples_identifiers.add(sample_text_for_id)
                    category_counts[label] += 1
                    # 从源列表中移除，避免后续重复选（如果你不想修改原始数据，就只用selected_samples_identifiers）
                    # samples_by_level_and_category[level][label].remove(chosen_sample)
                    print(f"  Selected 1 for '{label}' from Level {level}. Total selected: {len(selected_samples_list)}")
                    selected_for_this_label = True
                    break # 这个类别选到了，换下一个类别
        if selected_for_this_label:
            break
    if not selected_for_this_label:
        print(f"  Warning: Could not find any sample for category '{label}' in priority levels.")


# 阶段2: 根据目标Level分配和类别均衡，补足剩余名额
print("\n--- Stage 2: Filling remaining slots based on level targets and balance ---")
remaining_slots = NUM_TOTAL_SAMPLES_TO_SELECT - len(selected_samples_list)

# 仍然按优先级level来填充
for level in priority_levels:
    if remaining_slots <= 0:
        break
    
    # 当前level的目标选择数
    # 我们需要考虑已经通过阶段1选择的样本，调整当前level还需选择的数量
    # 简单起见，我们直接看这个level的目标还能选多少
    target_for_this_level = target_distribution_by_level.get(level, 0)
    
    # 获取当前level下所有类别的所有可用样本（未被选中的）
    # 为了简单，我们先不严格去重，依赖于selected_samples_identifiers，或者随机选择的低碰撞概率
    # 更严谨的做法是维护每个level每个类别下未被选择的样本列表
    
    # 收集当前level所有可供选择的样本，并打乱顺序
    potential_samples_this_level = []
    for label in shuffled_labels: # 再次打乱顺序遍历类别，试图均衡
        if label in samples_by_level_and_category[level]:
            potential_samples_this_level.extend(
                [s for s in samples_by_level_and_category[level][label] if s["messages"][1]["content"] not in selected_samples_identifiers]
            )
    random.shuffle(potential_samples_this_level)

    can_select_from_this_level = min(remaining_slots, target_for_this_level, len(potential_samples_this_level))
    
    for i in range(can_select_from_this_level):
        if not potential_samples_this_level: break # 以防万一

        chosen_sample = potential_samples_this_level.pop(0) # 取一个
        sample_text_for_id = chosen_sample["messages"][1]["content"]

        # 再次检查去重（如果potential_samples_this_level没有实时更新的话）
        if sample_text_for_id not in selected_samples_identifiers:
            selected_samples_list.append(chosen_sample)
            selected_samples_identifiers.add(sample_text_for_id)
            true_label = chosen_sample["messages"][2]["content"]
            # true_label = chosen_sample["messages"][2]["content"].split('#')[0]
            category_counts[true_label] += 1
            remaining_slots -= 1
            print(f"  Selected 1 (label: '{true_label}') from Level {level} to fill slots. Total: {len(selected_samples_list)}. Remaining slots: {remaining_slots}")
            if remaining_slots <= 0:
                break
        else: # 如果这个样本已经被选过了（理论上不应该发生，如果列表维护得好）
            i -= 1 # 重新尝试选一个
            continue


# 阶段3: 如果名额仍未满（比如某些level样本不足），从高优先级Level中按类别均衡查漏补缺
print("\n--- Stage 3: Final fill if slots remain, prioritizing balance ---")
remaining_slots = NUM_TOTAL_SAMPLES_TO_SELECT - len(selected_samples_list)
if remaining_slots > 0:
    print(f"  {remaining_slots} slots still remain. Attempting to fill...")
    # 优先补充那些当前选中数量最少的类别
    # sorted_categories_by_count = sorted(DEFINED_LABELS, key=lambda x: category_counts[x])
    
    # 简单地从高优先级level的剩余样本中随机抽取，直到满额，不特别强调类别均衡了，因为前面已经尝试过
    # 或者可以再次遍历类别，从样本最少的类别开始补充
    
    # 收集所有剩余的、未被选中的样本，按优先级level
    all_remaining_potentials = []
    for level in priority_levels:
        for label in DEFINED_LABELS:
            if label in samples_by_level_and_category[level]:
                all_remaining_potentials.extend(
                     [s for s in samples_by_level_and_category[level][label] if s["messages"][1]["content"] not in selected_samples_identifiers]
                )
    random.shuffle(all_remaining_potentials)
    
    can_select_finally = min(remaining_slots, len(all_remaining_potentials))
    for i in range(can_select_finally):
        chosen_sample = all_remaining_potentials.pop(0)
        sample_text_for_id = chosen_sample["messages"][1]["content"]
        # 再次检查去重
        if sample_text_for_id not in selected_samples_identifiers:
            selected_samples_list.append(chosen_sample)
            selected_samples_identifiers.add(sample_text_for_id)
            true_label = chosen_sample["messages"][2]["content"]
            # true_label = chosen_sample["messages"][2]["content"].split('#')[0]
            category_counts[true_label] += 1
            remaining_slots -= 1
            print(f"  Final fill: Selected 1 (label: '{true_label}'). Total: {len(selected_samples_list)}. Remaining slots: {remaining_slots}")
        else:
            i -=1
            continue


# --- 输出结果 ---
print(f"\n--- Final Selection Summary ---")
print(f"Total samples selected: {len(selected_samples_list)}")
print("Category distribution in selected samples:")
for label, count in category_counts.most_common():
    print(f"  {label}: {count}")

# 保存选中的样本
with open(OUTPUT_SELECTED_FILE, "w", encoding="utf-8") as f_out:
    json.dump(selected_samples_list, f_out, indent=4, ensure_ascii=False)
print(f"Saved {len(selected_samples_list)} selected samples to {OUTPUT_SELECTED_FILE}")

if len(selected_samples_list) < NUM_TOTAL_SAMPLES_TO_SELECT:
    print(f"Warning: Could only select {len(selected_samples_list)} samples, less than the target {NUM_TOTAL_SAMPLES_TO_SELECT}.")