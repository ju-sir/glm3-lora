import os
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, GenerationConfig
from peft import PeftModel
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


os.environ["CUDA_VISIBLE_DEVICES"] = "3"


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


def lora_merge_main(path_model):
    path_save_model = path_model + "-merge"
    model, tokenizer = load_model_and_tokenizer(path_model)
    print('origin config =', model.config)
    print(model)

    model = model.merge_and_unload()
    print('merge config =', model.config)
    print(model)

    print(f"Saving the target model to {path_save_model}")
    if not os.path.exists(path_save_model):
        os.makedirs(path_save_model)
    model.save_pretrained(path_save_model)
    tokenizer.save_pretrained(path_save_model)


if __name__ == '__main__':
    path_model = "/home/juguoyang/finetune/output_dir/lora/debug/checkpoint-50000"
    lora_merge_main(path_model)