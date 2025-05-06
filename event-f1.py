import json
import os
from pathlib import Path
from typing import Union
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

def _resolve_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()

ModelType = Union[PreTrainedModel, PeftModelForCausalLM]
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

def load_model_and_tokenizer(model_dir: Union[str, Path], trust_remote_code: bool = True) -> tuple[ModelType, TokenizerType]:
    model_dir = _resolve_path(model_dir)
    if (model_dir / 'adapter_config.json').exists():
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=trust_remote_code, device_map='auto'
        )
        tokenizer_dir = model.peft_config['default'].base_model_name_or_path
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=trust_remote_code, device_map='auto'
        )
        tokenizer_dir = model_dir

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=trust_remote_code)

    return model, tokenizer

def f1(true_labels, predicted_labels):
    metrics = {'f1': 0, 'p': 0, 'r': 0, 'acc': 0}
    for label in set(true_labels + predicted_labels):
        true_positive = sum(1 for t, p in zip(true_labels, predicted_labels) if t == p == label)
        predicted_positive = sum(1 for p in predicted_labels if p == label)
        actual_positive = sum(1 for t in true_labels if t == label)
        
        if predicted_positive > 0:
            precision = true_positive / predicted_positive
            metrics['p'] += precision
        if actual_positive > 0:
            recall = true_positive / actual_positive
            metrics['r'] += recall
        if true_positive > 0:
            metrics['f1'] += 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            metrics['acc'] += true_positive / len(true_labels)

    # Normalize metrics by the number of labels
    for key in metrics:
        metrics[key] /= len(set(true_labels + predicted_labels)) if set(true_labels + predicted_labels) else 1

    return metrics

def inference_f1_event(path_model, path_test, loop_num):
    labels = ['治安隐患', '困难救助', '矛盾纠纷', '城市管理', '消防安全', '生态环境', '文教体育', '信访维稳', '卫生健康', '农林水利', 
              '治危拆违', '食药安全', '扫黄打非', '自然灾害', '优抚安置', '老龄殡葬', '安全生产', '工商监管', '拯救老屋', '交通安全', 
              '质量监管', '金融监管', '应急管理', '其他']

    # Initialize metrics for each label
    label_metrics = {label: {"f1": 0, "p": 0, "r": 0, "acc": 0, "count": 0} for label in labels}
    overall_metrics = {"f1": 0, "p": 0, "r": 0, "acc": 0, "count": 0}

    model, tokenizer = load_model_and_tokenizer(path_model)
    
    with open(path_test, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    valid_sample_num = 0
    mistake_data = []

    prompt_text0 = "你是一个政策事件分类方面的专家，请根据下面的事件内容判断属于哪个标签。\n标签包括：['治安隐患','困难救助','矛盾纠纷','城市管理','消防安全','生态环境','文教体育','信访维稳','卫生健康','农林水利','治危拆违','食药安全','扫黄打非','自然灾害','优抚安置','老龄殡葬','安全生产','工商监管','拯救老屋','交通安全','质量监管','金融监管','应急管理','其他']\n例子1：输入事件内容：12月3日，道太乡雁溪村网格员入户走访探访关爱并对其信息核实。分类的结果标签是：'困难救助'\n例子2：输入事件内容：2023年4月23日榴溪村网格员下村宣传反诈骗！。分类的结果标签是：'治安隐患'\n例子3：输入事件内容：2023年1月18日，五莲村栋头村民潘伟阳因宅基地与邻居发生争执。分类的结果标签是：'矛盾纠纷'\n请判断下面的事件内容属于哪个标签：\n"
    # 加强版prompt1
    # prompt_text1 = "你是一个政府网格事件分类专家，请严格遵循以下规则分类：\n\n1. 优先选择最具体标签\n2. 涉及多标签时按危害程度排序：治安>安全>管理>其他\n3. 民生类问题优先困难救助\n\n标签定义：\n- 治安隐患：涉及人身安全、社会治安、诈骗、非法活动等。\n- 矛盾纠纷：邻里、劳资、商业、家庭等纠纷问题。\n- 困难救助：针对弱势群体或特殊情况的救助。\n- 安全生产：工作场所的安全隐患和事故管理。\n- 消防安全：火灾预防、消防设施检查、用火用电安全教育。\n- 自然灾害：自然现象引发的灾害，包括次生灾害。\n- 城市管理：城市环境秩序、公共设施管理。\n- 农林水利：农业、林业、水利资源的管理和保护。\n- 工商监管：商业活动的合法性和规范性检查。\n- 食药安全：食品和药品的安全管理。\n- 文教体育：教育、文化、体育活动相关的管理和宣传。\n- 信访维稳：社情民意的收集、信访处理、稳定管理。\n- 卫生健康：公共健康、医疗卫生服务。\n- 扫黄打非：文化市场的非法活动监管。\n- 应急管理：突发事件的应急响应和管理。\n- 其他：不属于上述任何标签的事件。\n\n核心标签示范：\n- 治安隐患：\n  - 输入：村民接到诈骗电话 → 标签：治安隐患\n  - 输入：涉黑线索 → 标签：治安隐患\n- 矛盾纠纷：\n  - 输入：村民因自留地纠纷发生争执 → 标签：矛盾纠纷\n  - 输入：劳资纠纷 → 标签：矛盾纠纷\n- 困难救助：\n  - 输入：慰问独居老人 → 标签：困难救助\n  - 输入：为低保户提供帮助 → 标签：困难救助\n\n关键修正示例：\n- 错误案例1：输入：家庭暴力 → 原标\"矛盾纠纷\" → 正确应为\"治安隐患\"\n- 错误案例2：输入：暴雨导致道路积水 → 原标\"城市管理\" → 正确应为\"自然灾害\"\n- 错误案例3：输入：非法捕鱼 → 原标\"工商监管\" → 正确应为\"农林水利\"\n\n新增判断规则：\n- 涉及人身安全的直接归治安隐患（如家暴、性侵）。\n- 自然灾害相关事件统一归自然灾害（含次生问题）。\n- 野生动物/农林问题统一归农林水利（不论交易环节）。\n\n精简后的标签优先级：\n['治安隐患', '矛盾纠纷', '困难救助', '安全生产', '消防安全', '自然灾害', '城市管理', '农林水利', '工商监管', '食药安全', '文教体育', '信访维稳', '卫生健康', '扫黄打非', '应急管理', '其他']\n\n扩展标签示范：\n- 安全生产：\n  - 输入：企业安全生产检查 → 标签：安全生产\n- 消防安全：\n  - 输入：发现商铺未配备灭火器 → 标签：消防安全\n- 自然灾害：\n  - 输入：天气预报提示强对流天气 → 标签：自然灾害\n- 城市管理：\n  - 输入：车辆违法停车 → 标签：城市管理\n- 农林水利：\n  - 输入：五水共治宣传 → 标签：农林水利\n- 工商监管：\n  - 输入：发现非法传销 → 标签：工商监管\n- 食药安全：\n  - 输入：卫生检查发现过期食品 → 标签：食药安全\n- 文教体育：\n  - 输入：开展反诈宣传 → 标签：文教体育\n- 信访维稳：\n  - 输入：反邪教宣传活动 → 标签：信访维稳\n- 卫生健康：\n  - 输入：发放防疫包 → 标签：卫生健康\n- 扫黄打非：\n  - 输入：巡查非法出版物 → 标签：扫黄打非\n- 应急管理：\n  - 输入：参加防汛应急演练 → 标签：应急管理\n\n根据这些规则，请为以下事件选择适当的标签："
    prompt_text2 = "你是一名政府事件分类专家，请根据事件内容严格选择最匹配的单一标签。\n【标签定义】\n治安隐患：涉黑涉恶,管制器具,流动人口未登记,极端行为倾向,电诈线索等\n矛盾纠纷：家庭,邻里,劳资等纠纷及暴力行为（不含明确违法行为）\n城市管理：市政设施损坏,占道经营,市容环境问题,流浪动物等\n困难救助：帮扶困难群体,信息核实,特殊人群关爱等\n安全生产：危险品违规操作,生产设施隐患,无证上岗等\n消防安全：消防设施问题,电动车违规充电,森林火情等\n治危拆违：危房隐患,违法建筑,违规装修等\n信访维稳：群体上访,特殊人员异常动态,反动活动等\n应急管理：突发事故处置,灾后救援等\n生态环境：污染排放,噪音扬尘,水体污染等\n农林水利：侵占河道,动植物疫情,农资违规等\n自然灾害：极端天气灾害,房屋倒塌,道路受损等\n扫黄打非：非法出版物,低俗演出,文物违法等\n卫生健康：非法行医,传染病预警,性别鉴定等\n老龄殡葬：损害老人权益,违规殡葬事件等\n食药安全：食品药品违规经营,卫生隐患等\n工商监管：假冒伪劣,违法广告,非法经营等\n质量监管：特种设备违规,计量问题等\n金融监管：非法集资,违规融资等\n文教体育：非法办学,设施损坏,宣传活动等\n优抚安置：退役军人事务,烈士纪念等\n交通安全：道路隐患,违法占道,交通事故等\n拯救老屋：老屋修缮,巡查,活化利用等\n其他：无法匹配上述标签时使用\n【关键判断规则】\n1. 优先级：治安>安全类>救助类>管理类>其他\n2. 同时涉及多标签时：\n   - 出现违法行为的优先治安隐患\n   - 涉消防设施优先消防安全，涉生产流程优先安全生产\n   - 帮扶行为存在时优先困难救助\n3. 自然灾害导致的问题归自然灾害，人为损坏归城市管理\n【分类示例】\n输入：村民接到诈骗电话 → 治安隐患\n输入：调解商铺租赁合同纠纷 → 矛盾纠纷\n输入：为残障人士配送生活物资 → 困难救助\n输入：清理违规设置的广告灯箱 → 城市管理\n输入：建筑工地塔吊未年检 → 安全生产\n输入：超市安全出口被货物堵塞 → 消防安全\n输入：老屋屋顶瓦片脱落巡查 → 拯救老屋\n输入：东湖村网格员送防疫包给村民 → 卫生健康\n根据这些规则，请为以下事件选择适当的标签："
    if loop_num<=3:
        prompt_text = prompt_text0
    else : prompt_text = prompt_text2
    for i in range(len(data) - 1, -1, -1):
        sentence = data[i]["messages"][1]["content"]
        print(f"event: {sentence}")
        if sentence == "":
            del data[i]
            continue
        
        # prompt_text = data[i]["messages"][0]["content"]
        input = f'{prompt_text}{sentence}'
        label = data[i]["messages"][2]["content"]
        # print(f"shuru：{input}")
        
        response, history = model.chat(tokenizer, input, history=[])
        response = str(response)

        # Compare predicted response with actual label
        # 将字符串按逗号分割
        split_labels = label.split(",")
        # 获取第二个元素（如果有的话），否则返回第一个元素
        true_labels = [split_labels[1]] if len(split_labels) > 1 else [split_labels[0]]
        print(f"true_labels:{true_labels}")
        predicted_labels = response.split(";")
        print(f"predicted_labels{predicted_labels}")
        # print("---------------------------------------------")

        sample_metrics = f1(true_labels, predicted_labels)
        # print(f"zhibiao:{sample_metrics}")
        
        # Update individual label metrics
        for label in true_labels:
            if label in label_metrics:
                for metric in ["f1", "p", "r", "acc"]:
                    label_metrics[label][metric] += sample_metrics[metric]
                label_metrics[label]["count"] += 1

        # Update overall metrics
        for metric in ["f1", "p", "r", "acc"]:
            overall_metrics[metric] += sample_metrics[metric]
        overall_metrics["count"] += 1

        valid_sample_num += 1

        # If true_labels and predicted_labels are not equal, add to mistake data
        # if true_labels != predicted_labels:
        #     mistake_data.append(data[i])

    # Calculate averages for each label
    print("Metrics for each label:")
    for label, metrics in label_metrics.items():
        if metrics["count"] > 0:
            for metric in ["f1", "p", "r", "acc"]:
                metrics[metric] /= metrics["count"]
            # print(f"{label}: F1 = {metrics['f1']:.4f}, Precision = {metrics['p']:.4f}, Recall = {metrics['r']:.4f}, Accuracy = {metrics['acc']:.4f}")
        print(f"{label}: F1 = {metrics['f1']:.4f}")

    # Calculate overall metrics
    if overall_metrics["count"] > 0:
        for metric in ["f1", "p", "r", "acc"]:
            overall_metrics[metric] /= overall_metrics["count"]
        print("\nOverall Metrics:")
        print(f"Overall F1 = {overall_metrics['f1']:.4f}, Precision = {overall_metrics['p']:.4f}, Recall = {overall_metrics['r']:.4f}, Accuracy = {overall_metrics['acc']:.4f}")

    # Save mistake data to a file
    # mistake_file = f"/home/juguoyang/finetune/A100/event-finetune/mistake_data/mistake{loop_num + 1}.json"
    # with open(mistake_file, "w", encoding="utf-8") as fh:
    #     json.dump(mistake_data, fh, ensure_ascii=False, indent=4)

    return label_metrics, overall_metrics

if __name__ == '__main__':
    model_list = [
                   "/home/juguoyang/finetune/A100/event-finetune/output-glm3/event-only2label-pepf-chushiprompt/checkpoint-400000",
                                   
                     ]

    test_list = [                   
                    "/home/juguoyang/finetune/A100/event-finetune/event-only2label-prompt/test.json",
                          
             ]

    for i in range(len(model_list)):
        path_model = model_list[i]
        path_test = test_list[i]
        
        print(f"Testing with model: {path_model}")
        print(f"Test file: {path_test}")

        label_metrics, overall_metrics = inference_f1_event(path_model, path_test, i)
        print(f"{path_test}中每一个标签的指标: {label_metrics}")
        print(f"{path_test}的总的指标: {overall_metrics}")
        # print(f"错误的数据：{mistake_data}")
        # Here you would process or save label_metrics and overall_metrics as needed

