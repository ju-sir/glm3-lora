import json
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from pathlib import Path
from typing import Union
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)


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

    print(model)
    for i, para in enumerate(model.named_parameters()):
        print(f"{i}, {para[0]}\t {para[1].device} \t{para[1].dtype}")

    return model, tokenizer


def merg_list_in_dict(pre_res):
    event_type_dic = set(pre_res.keys())

    merged_list = []
    for sublist in pre_res.values():
        merged_list.extend(sublist)
    event_rolo_dic = set(merged_list)

    return event_type_dic, event_rolo_dic


def f1(real_event_type, real_event_rolo, pre_event_type, pre_event_rolo):
    # 1.先计算事件类型的f1指标
    same_event_type = pre_event_type & real_event_type
    p_type = len(same_event_type) / len(pre_event_type) if pre_event_type else 0.0
    r_type = len(same_event_type) / len(real_event_type)
    f1_type = 2 * p_type * r_type / (p_type + r_type) if (p_type + r_type) != 0.0 else 0.0

    # 2.再计算事件角色的f1指标
    if len(pre_event_rolo) == 0 and len(real_event_rolo) == 0:
        f1_rolo = 1.0
    elif len(pre_event_rolo) != 0 and len(real_event_rolo) == 0:
        f1_rolo = 0.0
    elif len(pre_event_rolo) == 0 and len(real_event_rolo) != 0:
        f1_rolo = 0.0
    else:
        same_event_rolo = pre_event_rolo & real_event_rolo
        p_rolo = len(same_event_rolo) / len(pre_event_rolo) if pre_event_rolo else 0.0
        r_rolo = len(same_event_rolo) / len(real_event_rolo)
        f1_rolo = 2 * p_rolo * r_rolo / (p_rolo + r_rolo) if (p_rolo + r_rolo) != 0.0 else 0.0

    output_dic = {
        "f1_type": f1_type,
        "f1_rolo": f1_rolo
    }
    return output_dic


def inference_f1_event_role(path_model, path_test,path_save):
    model, tokenizer = load_model_and_tokenizer(path_model)

    # with open(path_prompt, "r", encoding="utf8") as f:
    #     prompt_text = f.read().strip()

    data = ""
    with open(path_test, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    error_list = []    # 模型在部分样本上输出的结果是字典！很奇怪！
    save_data = []
    sum_dic = {"f1_type_sum": 0, "f1_rolo_sum": 0}
    sum_time, valid_sample_num = 0, 0
    for i in range(len(data)):
        print(f" >>>>>>>>>>>>>>>>>> [{i}/{len(data)}]")
        sentence = data[i]["messages"][1]["content"]
        prompt_text = data[i]["messages"][0]["content"]
        input = '{}{}'.format(prompt_text, sentence)
        print(f"[input]：{input}")
        label = data[i]["messages"][2]["content"]
        print(f"[real_label]：{label}")

        start_time = time.time()
        response, history = model.chat(tokenizer, input, history=[])
        sampe_time = time.time() - start_time
        sum_time = sum_time + sampe_time
        print(f"[response]：{response}")

        if isinstance(response, dict):
            error_list.append(i)
            continue
        else:
            valid_sample_num = valid_sample_num + 1
        # ==============================================================================================
        #                                         处理模型回答的结果
        # 格式：字典，键为事件类型，值为事件角色列表。
        # 例：real_res={'企业融资': ['6月9日_披露时间', '韦拓生物_被投资方', 'B_融资轮次', '道彤投资_投资方']}
        #     pre_res={'企业融资': ['6月9日_披露时间', '韦拓生物_被投资方', 'B_融资轮次', '道彤投资_投资方']}
        # ==============================================================================================
        real_res = {
            event.split("#")[0]: event.split("#")[1].split(";")
            for event in label.split("\n")
        }
        # pre_res = {
        #     rr.split("#")[0]: rr.split("#")[1].split(";")
        #     for rr in response.split("\n")
        # }
        pre_res = {}  
        for line in response.split("\n"):    
            key_value = line.split("#")  
            if len(key_value) == 2:  # 确保有两个部分  
                key = key_value[0]  
                value = key_value[1].split(";")  
                pre_res[key] = value
            
        print(f"[pre_res] {pre_res}")
        print(f"[real_res] {real_res}")
        # ==============================================================================================
        #                                     评估模型回答的结果性能
        # ==============================================================================================
        real_event_type, real_event_rolo = merg_list_in_dict(real_res)
        pre_event_type, pre_event_rolo = merg_list_in_dict(pre_res)
        output_dic = f1(real_event_type, real_event_rolo, pre_event_type, pre_event_rolo)

        sum_dic["f1_type_sum"] += output_dic["f1_type"]
        sum_dic["f1_rolo_sum"] += output_dic["f1_rolo"]
        # ==============================================================================================
        #                                         Log 打印
        # ==============================================================================================
        print(f"[sampe_time / avg_time (s)]：{sampe_time} / {sum_time / valid_sample_num}")    
        print(f">>>>>> [log] true_type/total_type={len(pre_event_type)}/{len(real_event_type)}, "
              f"f1_type_avg={sum_dic['f1_type_sum'] / valid_sample_num}")
        print(f">>>>>> [log] true_role/total_role={len(pre_event_rolo)}/{len(real_event_rolo)}, "
              f"f1_rolo_avg={sum_dic['f1_rolo_sum'] / valid_sample_num}")
        # ==============================================================================================
        #                                         结果保存到.json
        # ==============================================================================================
        save_data.append(
            {
                "glm4输入内容": input,
                "glm4回答内容": response,

                "real_res": real_res,
                "pre_res": pre_res,

                "f1_type | f1_rolo": [output_dic['f1_type'], output_dic['f1_rolo']],

                "f1_type_avg | f1_rolo_avg": [sum_dic['f1_type_sum'] / valid_sample_num, sum_dic['f1_rolo_sum'] / valid_sample_num],
                "time | avg_time": [sampe_time, sum_time / valid_sample_num]
            }
        )

    print(f"valid_sample_num = {valid_sample_num}")
    with open(path_save, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=4)
    print(error_list)

    return (sum_dic['f1_type_sum'] / valid_sample_num), (sum_dic['f1_rolo_sum'] / valid_sample_num)


if __name__ == '__main__':
    # python test-glm4-from-chatglm3.py
    # ***************************************************
    # dataset = "02_ACE05- 01_CASIE"
    model_list = ["/ssd0/jgy/thesis_experiment/lys_llm_exp/output-glm3/casie-dummyclass/checkpoint-100000",
                  "/ssd0/jgy/thesis_experiment/lys_llm_exp/output-glm4/casie-dummyclass/checkpoint-100000",
                  "/ssd0/jgy/thesis_experiment/lys_llm_exp/output-glm3/ace-dummyclass/checkpoint-100000",
                  "/ssd0/jgy/thesis_experiment/lys_llm_exp/output-glm4/ace-dummyclass/checkpoint-100000",
                  "/ssd0/jgy/thesis_experiment/lys_llm_exp/output-glm3/casie-hyy/checkpoint-100000",
                  "/ssd0/jgy/thesis_experiment/lys_llm_exp/output-glm4/casie-hyy/checkpoint-100000",
                  ]
    
    test_list = ["/ssd0/jgy/thesis_experiment/lys_llm_exp/glm4-r8-data/01_CASIE/04_no_trigger_table/test.json",
                 "/ssd0/jgy/thesis_experiment/lys_llm_exp/glm4-r8-data/01_CASIE/04_no_trigger_table/test.json",
                 "/ssd0/jgy/thesis_experiment/lys_llm_exp/glm4-r8-data/02_ACE05/04_no_trigger_table/test.json",
                 "/ssd0/jgy/thesis_experiment/lys_llm_exp/glm4-r8-data/02_ACE05/04_no_trigger_table/test.json",
                 "/ssd0/jgy/thesis_experiment/lys_llm_exp/glm4-r8-data/01_CASIE/04_no_trigger_table/test.json",
                 "/ssd0/jgy/thesis_experiment/lys_llm_exp/glm4-r8-data/01_CASIE/04_no_trigger_table/test.json",
                 ]
    for i in range(6):
        path_model = model_list[i]
        path_test = test_list[i]
        # path_test = "data/" + dataset + "/struct_data/glm4/04_no_trigger_table/test.json"
        print(path_model)
        print(path_test)

        # 测试3次取平均值
        sum_dic = {"f1_type_sum": [], "f1_rolo_sum": []}
        for num in range(3):
            path_save = path_model + "/test" + str(num) + ".json"
            # path_save = path_model + "/test" + str(num) + ".json"
            # f1_type, f1_rolo = inference_f1_event_role(path_model, path_prompt, path_test, path_save)
            f1_type, f1_rolo = inference_f1_event_role(path_model, path_test , path_save)
            sum_dic["f1_type_sum"].append(f1_type)
            sum_dic["f1_rolo_sum"].append(f1_rolo)

        print(path_model)
        print(path_test)
        print(sum_dic)
        print(f"f1_type_avg={sum(sum_dic['f1_type_sum']) / len(sum_dic['f1_type_sum'])}, f1_rolo_avg={sum(sum_dic['f1_rolo_sum']) / len(sum_dic['f1_rolo_sum'])}")
        sum_dic.clear