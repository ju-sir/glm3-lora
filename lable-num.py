import json
import os

# 文件路径
file_path = '/home/juguoyang/finetune/A100/event-finetune/event-only2label-del/test.json'

# 读取文件内容
with open(file_path, "r", encoding="utf-8") as fh:
    data = json.load(fh)

# 标签列表
labels = ['治安隐患', '困难救助', '矛盾纠纷', '城市管理', '消防安全', '生态环境', '文教体育', '信访维稳', '卫生健康', '农林水利', '治危拆违', 
          '食药安全', '扫黄打非', '自然灾害', '优抚安置', '老龄殡葬', '安全生产', '工商监管', '拯救老屋', '交通安全', '质量监管', '金融监管', 
          '应急管理', '其他']

# 用于统计每个标签出现的次数
label_counts = {label: 0 for label in labels}

# 遍历数据，根据标签分类
for i in range(len(data)):
    # 获取当前消息的 content 内容
    content = data[i]["messages"][2]["content"]
    # content = data[i]["messages"][2]["content"].split(",")[1]
    
    # 遍历每个标签，检查其是否出现在 content 中
    for label in labels:
        if label in content:
            label_counts[label] += 1

# 输出每个标签的统计数量
num =0
for label, count in label_counts.items():
    num+=count
    print(f"{label}: {count}")
print(f"总共：{num}")
