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
    labels = ['declare bankruptcy', 'transfer ownership', 'transfer money', 'marry', 
             'transport', 'die', 'phone write', 'arrest jail', 'convict', 'sentence', 'sue', 
             'end organization', 'start organization', 'end position', 'start position', 'meet', 
             'elect', 'attack', 'injure', 'born', 'fine', 'release parole', 'charge indict', 'extradite', 
             'trial hearing', 'demonstrate', 'divorce', 'nominate', 'appeal', 'pardon', 'execute', 'acquit', 
             'merge organization']

    # onelabel = ['综合执法;城市管理','综合执法;消防安全','便民服务;文教体育','便民服务;困难救助','综合执法;其他','综合执法;生态环境','综合执法;矛盾纠纷','便民服务;卫生健康','综治工作;信访维稳','综合执法;治危拆违','综治工作;治安隐患','市场监管;食药安全','综合执法;扫黄打非','综合执法;安全生产','综合执法;自然灾害','市场监管;工商监管','便民服务;老龄殡葬','综治工作;拯救老屋','市场监管;质量监管','市场监管;金融监管','其他;其他','综合执法;应急管理','综合执法;农林水利','便民服务;优抚安置']
    
    # Initialize metrics for each label
    label_metrics = {label: {"f1": 0, "p": 0, "r": 0, "acc": 0, "count": 0} for label in labels}
    overall_metrics = {"f1": 0, "p": 0, "r": 0, "acc": 0, "count": 0}

    model, tokenizer = load_model_and_tokenizer(path_model)
    
    with open(path_test, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    valid_sample_num = 0
    mistake_data = []

  
       # if loop_num<=3:
    #     prompt_text = prompt_text0
    # else : prompt_text = prompt_text2
    for i in range(len(data) - 1, -1, -1):
        sentence = data[i]["messages"][1]["content"]
        print(f"event: {sentence}")
        if sentence == "":
            del data[i]
            continue
        
        prompt_text = data[i]["messages"][0]["content"]

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
        print(f"{label}: F1 = {metrics['f1']:.4f} , count_num = {metrics['count']}")

    # Calculate overall metrics
    if overall_metrics["count"] > 0:
        for metric in ["f1", "p", "r", "acc"]:
            overall_metrics[metric] /= overall_metrics["count"]
        print("\nOverall Metrics:")
        print(f"Overall F1 = {overall_metrics['f1']:.4f}, Precision = {overall_metrics['p']:.4f}, Recall = {overall_metrics['r']:.4f}, Accuracy = {overall_metrics['acc']:.4f}")

    # Save mistake data to a file
    # mistake_file = f"/home/jgy/A10备份/mistake{loop_num + 1}.json"
    # with open(mistake_file, "w", encoding="utf-8") as fh:
    #     json.dump(mistake_data, fh, ensure_ascii=False, indent=4)

    return label_metrics, overall_metrics

if __name__ == '__main__':
    model_list = [
                #    "/home/jgy/thesis-experiment/outglm3/ace-3layer-onlytype/checkpoint-50000",
                "/home/jgy/thesis-experiment/outglm3/ace-lora/checkpoint-100000",
                                   
                     ]

    test_list = [                   
                    # "/home/jgy/thesis-experiment/data/02_ACE05_random_onlytype/04_no_trigger_table//test.json",
                    "/home/jgy/thesis-experiment/data/02_ACE05_random_onlytype/04_no_trigger_table/test.json",
                          
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

