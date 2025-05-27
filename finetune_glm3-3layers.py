# -*- coding: utf-8 -*-
import os
import jieba
import torch.nn.functional as F # <--- 导入 F 用于 Dropout
import types   # <--- 导入 types 用于猴子补丁
import dataclasses as dc
import functools
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Optional, Union
import numpy as np
import ruamel.yaml as yaml
import torch
import typer
from datasets import Dataset, DatasetDict, NamedSplit, Split, load_dataset
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from peft import (
    PeftConfig,
    PeftModelForCausalLM,
    get_peft_config,
    get_peft_model
)
from rouge_chinese import Rouge
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EvalPrediction,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    Seq2SeqTrainingArguments, AutoConfig,PretrainedConfig,
)
from transformers import DataCollatorForSeq2Seq as _DataCollatorForSeq2Seq

from transformers import Seq2SeqTrainer as _Seq2SeqTrainer



ModelType = Union[PreTrainedModel, PeftModelForCausalLM]
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
app = typer.Typer(pretty_exceptions_show_locals=False)


class LocalConv1D(nn.Module):
    def __init__(self, config: PretrainedConfig, kernel_size: int = 3, device=None, dtype=None): # dtype 参数在这里实际未被用于卷积层权重精度
        super().__init__()
        self.config = config
        self.hidden_size = getattr(config, 'hidden_size', 4096)
        self.kernel_size = kernel_size
        self.groups = self.hidden_size
        add_bias = getattr(config, 'add_bias_linear', False)

        # --- 关键：创建卷积层时，直接指定其参数为 FP32 ---
        try:
            self.conv1d = nn.Conv1d(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                groups=self.groups,
                bias=add_bias,
                device=device,      # 目标设备
                dtype=torch.float32 # <--- 参数强制为 FP32
            )
            print(f"LocalConv1D ({kernel_size=}) created. Weight on {self.conv1d.weight.device}, Weight dtype: {self.conv1d.weight.dtype}")
        except Exception as e:
            print(f"!!! ERROR creating Conv1d layer with FP32 dtype: {e}")
            raise e

        # --- 初始化 (现在权重已经是 FP32) ---
        std = getattr(config, 'initializer_range', 0.02)
        try:
            print(f"  Initializing FP32 conv1d weight (std={std})...")
            nn.init.normal_(self.conv1d.weight, mean=0.0, std=std) # 权重已经是 FP32
            if not torch.isfinite(self.conv1d.weight).all():
                 raise RuntimeError("FP32 Weight became NaN/Inf immediately after init!")
            if self.conv1d.bias is not None:
                print(f"  Initializing FP32 conv1d bias to zeros...")
                nn.init.zeros_(self.conv1d.bias) # 偏置已经是 FP32
                if not torch.isfinite(self.conv1d.bias).all():
                     raise RuntimeError("FP32 Bias became NaN/Inf immediately after init!")
            print("  FP32 Initialization successful and finite.")
        except Exception as e:
            print(f"!!! ERROR during LocalConv1D FP32 initialization: {e}")
            raise e

        # --- 冻结参数 ---
        for param in self.conv1d.parameters():
            param.requires_grad = True
        print(f"  LocalConv1D (FP32 weights) parameters frozen.")

    def forward(self, hidden_states):
        input_dtype_original = hidden_states.dtype # 保存原始输入类型，例如 float16
        layer_num_info = f"Layer {getattr(self, 'layer_number', 'N/A')}" # 获取层号

        if not torch.isfinite(hidden_states).all():
            print(f"!!! WARNING: Input to LocalConv1D forward ({input_dtype_original}) is not finite! {layer_num_info}...")
            # return hidden_states # 或者抛出错误，或者返回一个0张量以避免进一步污染

        # 1. 输入转换为 FP32
        hidden_states_fp32 = hidden_states.to(torch.float32)
        # 检查转换后的 hidden_states_fp32
        if not torch.isfinite(hidden_states_fp32).all():
            print(f"!!! WARNING: hidden_states_fp32 (after to(torch.float32)) is not finite! {layer_num_info}...")
        
        x_permuted_fp32 = hidden_states_fp32.permute(1, 2, 0)

        # 检查权重是否有效
        if not torch.isfinite(self.conv1d.weight).all():
            print(f"!!! ERROR: Weight is NaN/Inf at start of forward! {layer_num_info}")
            assert False, f"Weight is NaN/Inf at start of forward! {layer_num_info}"
        
        # 3. 执行卷积 (输入 FP32, 权重 FP32)
        try:
            conv_output_fp32 = self.conv1d(x_permuted_fp32)
            if not torch.isfinite(conv_output_fp32).all():
                print(f"!!! ERROR: NaN/Inf after conv1d (FP32 calc) in {layer_num_info}. Input was finite: {torch.isfinite(x_permuted_fp32).all()}")
                # 可以在这里打印 x_permuted_fp32 和 self.conv1d.weight 的一些统计信息
                # print(f"    x_permuted_fp32 min: {torch.min(x_permuted_fp32)}, max: {torch.max(x_permuted_fp32)}")
                # print(f"    conv1d.weight min: {torch.min(self.conv1d.weight)}, max: {torch.max(self.conv1d.weight)}")
                assert False, f"NaN/Inf after conv1d (FP32 calc) in {layer_num_info}"
        except Exception as e:
             print(f"!!! ERROR during conv1d(FP32) forward calculation! {layer_num_info}: {e}")
             raise e
        
        output_fp32 = conv_output_fp32.permute(2, 0, 1)
        # 4. 将结果转换回原始输入类型
        output = output_fp32.to(input_dtype_original)

        if not torch.isfinite(output).all():
            print(f"!!! WARNING: Output of LocalConv1D (after to original_dtype {input_dtype_original}) is not finite! {layer_num_info}")
            # assert False, f"Output of LocalConv1D not finite after casting back in {layer_num_info}" # 暂时注释掉，看看是否能继续
        return output





class DataCollatorForSeq2Seq(_DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        output_ids = (
            [feature['output_ids'] for feature in features]
            if 'output_ids' in features[0].keys()
            else None
        )
        if output_ids is not None:
            max_output_length = max(len(out) for out in output_ids)
            if self.pad_to_multiple_of is not None:
                max_output_length = (
                        (
                                max_output_length + self.pad_to_multiple_of - 1) //
                        self.pad_to_multiple_of * self.pad_to_multiple_of
                )
            for feature in features:
                remainder = [self.tokenizer.pad_token_id] * (
                        max_output_length - len(feature['output_ids'])
                )
                if isinstance(feature['output_ids'], list):
                    feature['output_ids'] = feature['output_ids'] + remainder
                else:
                    feature['output_ids'] = np.concatenate(
                        [feature['output_ids'], remainder]
                    ).astype(np.int64)
        return super().__call__(features, return_tensors)


class Seq2SeqTrainer(_Seq2SeqTrainer):
    def prediction_step(
            self,
            model: nn.Module,
            inputs: dict[str, Any],
            prediction_loss_only: bool,
            ignore_keys=None,
            **gen_kwargs,
    ) -> tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.args.predict_with_generate:
            output_ids = inputs.pop('output_ids')
        input_ids = inputs['input_ids']
        loss, generated_tokens, labels = super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs
        )
        generated_tokens = generated_tokens[:, input_ids.size()[1]:]
        if self.args.predict_with_generate:
            labels = output_ids
        return loss, generated_tokens, labels
    # For P-Tuning a new save_model function is fine for the prefix_encoder model
    # but may cost problems for the whole model loading

    # def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
    #     if output_dir is None:
    #         output_dir = self.args.output_dir
    #     os.makedirs(output_dir, exist_ok=True)
    #     ptuning_params = {k: v for k, v in self.model.transformer.prefix_encoder.state_dict().items()}
    #
    #     torch.save(ptuning_params, os.path.join(output_dir, 'pytorch_model.bin'))
    #
    #     print(f"P-Tuning model weights saved in {output_dir}")
    #
    #     if self.tokenizer is not None:
    #         self.tokenizer.save_pretrained(output_dir)


def _resolve_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()


def _sanity_check(
        input_ids: Sequence[int],
        output_ids: Sequence[int],
        tokenizer: PreTrainedTokenizer,
):
    print('--> Sanity check')
    for in_id, out_id in zip(input_ids, output_ids):
        if in_id == 0:
            continue
        if in_id in tokenizer.tokenizer.index_special_tokens:
            in_text = tokenizer.tokenizer.index_special_tokens[in_id]
        else:
            in_text = tokenizer.decode([in_id])
        print(f'{repr(in_text):>20}: {in_id} -> {out_id}')


@functools.cache
def _get_yaml_parser() -> yaml.YAML:
    parser = yaml.YAML(typ='safe', pure=True)
    parser.indent(mapping=2, offset=2, sequence=4)
    parser.default_flow_style = False
    return parser


@dc.dataclass
class DataConfig(object):
    train_file: str
    val_file: Optional[str] = None
    test_file: Optional[str] = None

    num_proc: Optional[int] = None

    @property
    def data_format(self) -> str:
        return Path(self.train_file).suffix

    @property
    def data_files(self) -> dict[NamedSplit, str]:
        return {
            split: data_file
            for split, data_file in zip(
                [Split.TRAIN, Split.VALIDATION, Split.TEST],
                [self.train_file, self.val_file, self.test_file],
            )
            if data_file is not None
        }


@dc.dataclass
class FinetuningConfig(object):
    data_config: DataConfig

    max_input_length: int
    max_output_length: int

    training_args: Seq2SeqTrainingArguments = dc.field(
        default_factory=lambda: Seq2SeqTrainingArguments(output_dir='./output')
    )
    peft_config: Optional[PeftConfig] = None

    def __post_init__(self):
        if not self.training_args.do_eval or self.data_config.val_file is None:
            # skips the evaluation stage when `do_eval` or `eval_file` is not provided
            self.training_args.do_eval = False
            self.training_args.evaluation_strategy = 'no'
            self.data_config.val_file = None
        else:
            self.training_args.per_device_eval_batch_size = (
                    self.training_args.per_device_eval_batch_size
                    or self.training_args.per_device_train_batch_size
            )

    @classmethod
    def from_dict(cls, **kwargs) -> 'FinetuningConfig':
        training_args = kwargs.get('training_args', None)
        if training_args is not None and not isinstance(
                training_args, Seq2SeqTrainingArguments
        ):
            gen_config = training_args.get('generation_config')
            # TODO: a bit hacky
            if not isinstance(gen_config, GenerationConfig):
                training_args['generation_config'] = GenerationConfig(
                    **gen_config
                )
            kwargs['training_args'] = Seq2SeqTrainingArguments(**training_args)

        data_config = kwargs.get('data_config')
        if not isinstance(data_config, DataConfig):
            kwargs['data_config'] = DataConfig(**data_config)

        peft_config = kwargs.get('peft_config', None)
        if peft_config is not None and not isinstance(peft_config, PeftConfig):
            kwargs['peft_config'] = get_peft_config(peft_config)
        return cls(**kwargs)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> 'FinetuningConfig':
        path = _resolve_path(path)
        kwargs = _get_yaml_parser().load(path)
        return cls.from_dict(**kwargs)


def _load_datasets(
        data_dir: Path,
        data_format: str,
        data_files: dict[NamedSplit, str],
        num_proc: Optional[int],
) -> DatasetDict:
    if data_format in ('.csv', '.json', '.jsonl'):
        dataset_dct = load_dataset(
            data_format[1:],
            data_dir=data_dir,
            data_files=data_files,
            num_proc=num_proc,
        )
    else:
        err_msg = f"Cannot load dataset in the '{data_format}' format."
        raise NotImplementedError(err_msg)

    return dataset_dct


class DataManager(object):
    def __init__(self, data_dir: str, data_config: DataConfig):
        self._num_proc = data_config.num_proc

        self._dataset_dct = _load_datasets(
            _resolve_path(data_dir),
            data_config.data_format,
            data_config.data_files,
            self._num_proc,
        )

    def _get_dataset(self, split: NamedSplit) -> Optional[Dataset]:
        return self._dataset_dct.get(split, None)

    def get_dataset(
            self,
            split: NamedSplit,
            process_fn: Callable[[dict[str, Any]], dict[str, Any]],
            batched: bool = True,
            remove_orig_columns: bool = True,
    ) -> Optional[Dataset]:
        orig_dataset = self._get_dataset(split)
        if orig_dataset is None:
            return

        if remove_orig_columns:
            remove_columns = orig_dataset.column_names
        else:
            remove_columns = None
        return orig_dataset.map(
            process_fn,
            batched=batched,
            remove_columns=remove_columns,
            num_proc=self._num_proc,
        )


def print_model_size(model: PreTrainedModel):
    print("--> Model")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--> model has {total_params / 1e6}M params\n")


def process_batch(
        batch: Mapping[str, Sequence],
        tokenizer: PreTrainedTokenizer,
        max_input_length: int,
        max_output_length: int,
) -> dict[str, list]:
    batched_tools = batch.get('tools', None)
    batched_conv = batch['messages']
    batched_input_ids = []
    batched_labels = []

    if batched_tools is None:
        batched_tools = [None] * len(batched_conv)

    for tools, conv in zip(batched_tools, batched_conv):
        input_ids, loss_masks = [
            tokenizer.get_command('[gMASK]'),
            tokenizer.get_command('sop'),
        ], [False, False]

        if tools is not None:
            raise NotImplementedError()

        for message in conv:
            if message['role'] in ('system', 'user'):
                loss_mask_val = False
            else:
                loss_mask_val = True

            if message['role'] == 'tool':
                raise NotImplementedError()
            else:
                # Ensure message['content'] is a string
                
                new_input_ids = tokenizer.build_single_message(
                    message['role'], '', str(message['content'])
                )
                new_loss_masks = [loss_mask_val] * len(new_input_ids)

            input_ids += new_input_ids
            loss_masks += new_loss_masks

        input_ids.append(tokenizer.eos_token_id)
        loss_masks = [False, *loss_masks]
        labels = []
        for input_id, mask in zip(input_ids, loss_masks):
            if mask:
                labels.append(input_id)
            else:
                labels.append(-100)
        max_length = max_input_length + max_output_length + 1
        batched_input_ids.append(input_ids[:max_length])
        batched_labels.append(labels[:max_length])
    return {'input_ids': batched_input_ids, 'labels': batched_labels}


def process_batch_eval(
        batch: Mapping[str, Sequence],
        tokenizer: PreTrainedTokenizer,
        max_input_length: int,
        max_output_length: int,
) -> dict[str, list]:
    batched_tools = batch.get('tools', None)
    batched_conv = batch['messages']
    batched_input_ids = []
    # To avoid computing loss, we do not provide the `labels` field in the input dictionary.
    batched_output_ids = []

    if batched_tools is None:
        batched_tools = [None] * len(batched_conv)

    for tools, conv in zip(batched_tools, batched_conv):
        input_ids = [
            tokenizer.get_command('[gMASK]'),
            tokenizer.get_command('sop'),
        ]

        if tools is not None:
            raise NotImplementedError()

        for message in conv:
            if len(input_ids) >= max_input_length:
                break
            if message['role'] == 'tool':
                raise NotImplementedError()
            else:
                ###
                new_input_ids = tokenizer.build_single_message(
                    message['role'], '', str(message['content'])
                )
                if message['role'] == 'assistant':
                    output_prompt, output_ids = (
                        new_input_ids[:1],
                        new_input_ids[1:],
                    )
                    output_ids.append(tokenizer.eos_token_id)
                    batched_input_ids.append(
                        input_ids[:max_input_length] + output_prompt[:1]
                    )
                    batched_output_ids.append(output_ids[:max_output_length])
                input_ids += new_input_ids
    return {'input_ids': batched_input_ids, 'output_ids': batched_output_ids}


# Not sure if this is necessary, can set it to half.
# If train with cpu, cast all params to fp32 instead of trainable ones.
def _prepare_model_for_training(model: nn.Module, use_cpu: bool):
    for param in model.parameters():
        if param.requires_grad or use_cpu:
            param.data = param.data.to(torch.float32)


def load_tokenizer_and_model(
        model_dir: str,
        peft_config: Optional[PeftConfig] = None,
) -> tuple[PreTrainedTokenizer, nn.Module]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if peft_config is not None:
        if peft_config.peft_type.name == "PREFIX_TUNING":
            config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
            config.pre_seq_len = peft_config.num_virtual_tokens
            config.use_cache = False
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,
                config=config,
            )
        if peft_config.peft_type.name == "LORA":
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,
                empty_init=False,
                use_cache=False
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            empty_init=False,
            use_cache=False
        )
    print_model_size(model)
    return tokenizer, model


def compute_metrics(eval_preds: EvalPrediction, tokenizer: PreTrainedTokenizer):
    batched_pred_ids, batched_label_ids = eval_preds

    metrics_dct = {'rouge-1': [], 'rouge-2': [], 'rouge-l': [], 'bleu-4': []}
    for pred_ids, label_ids in zip(batched_pred_ids, batched_label_ids):
        pred_txt = tokenizer.decode(pred_ids).strip()
        label_txt = tokenizer.decode(label_ids).strip()
        pred_tokens = list(jieba.cut(pred_txt))
        label_tokens = list(jieba.cut(label_txt))
        rouge = Rouge()
        scores = rouge.get_scores(' '.join(pred_tokens), ' '.join(label_tokens))
        for k, v in scores[0].items():
            metrics_dct[k].append(round(v['f'] * 100, 4))
        metrics_dct['bleu-4'].append(
            sentence_bleu(
                [label_tokens],
                pred_tokens,
                smoothing_function=SmoothingFunction().method3,
            )
        )
    return {k: np.mean(v) for k, v in metrics_dct.items()}


@app.command()
def main(
        data_dir: Annotated[str, typer.Argument(help='')],
        model_dir: Annotated[
            str,
            typer.Argument(
                help='A string that specifies the model id of a pretrained model configuration hosted on huggingface.co, or a path to a directory containing a model configuration file.'
            ),
        ],
        config_file: Annotated[str, typer.Argument(help='')],
        auto_resume_from_checkpoint: str = typer.Argument(
            default='',
            help='If entered as yes, automatically use the latest save checkpoint. If it is a numerical example 12 15, use the corresponding save checkpoint. If the input is no, restart training'
        ),

):
    ft_config = FinetuningConfig.from_file(config_file)
    tokenizer, model = load_tokenizer_and_model(model_dir, peft_config=ft_config.peft_config)
    data_manager = DataManager(data_dir, ft_config.data_config)

    train_dataset = data_manager.get_dataset(
        Split.TRAIN,
        functools.partial(
            process_batch,
            tokenizer=tokenizer,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )
    print('train_dataset:', train_dataset)
    val_dataset = data_manager.get_dataset(
        Split.VALIDATION,
        functools.partial(
            process_batch_eval,
            tokenizer=tokenizer,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )
    if val_dataset is not None:
        print('val_dataset:', val_dataset)
    test_dataset = data_manager.get_dataset(
        Split.TEST,
        functools.partial(
            process_batch_eval,
            tokenizer=tokenizer,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )
    if test_dataset is not None:
        print('test_dataset:', test_dataset)

    # checks encoded dataset
    _sanity_check(
        train_dataset[0]["input_ids"], train_dataset[0]["labels"], tokenizer
    )

    # turn model to fp32
    # _prepare_model_for_training(model, ft_config.training_args.use_cpu)


    # 4. 模型准备（转精度和移动设备）
    print("Preparing model for training (_prepare_model_for_training)...")
    # --- 确保 _prepare_model_for_training 正确处理设备 ---
    # --- 或者在这里显式移动 ---
    target_device = ft_config.training_args.device
    print(f"Explicitly moving model to target device: {target_device} before _prepare...")
    model.to(target_device)
    _prepare_model_for_training(model, ft_config.training_args.use_cpu) # use_cpu 通常应为 False
    target_device = model.device # 获取模型最终设备
    print(f"Model prepared and placed on device: {target_device}")

    # --- 获取模型配置和数据类型 ---
    model_config = model.config
    try:
        current_model_dtype = next(model.parameters()).dtype
        print(f"Current model parameter dtype: {current_model_dtype}")
    except StopIteration:
        current_model_dtype = torch.float # Fallback
        print("!!! WARNING: Model has no parameters? Using torch.float.")
    # 对于这个版本的 LocalConv1D, dtype=None 传递进去，内部会用 torch.float32


    # ============================================================
    # === 动态获取 GLMBlock 类、添加层、猴子补丁 forward 方法 ===
    # ============================================================
    GLMBlock_cls = None
    layers_to_process_list = [] # 和之前一样

    model_for_accessing_layers = model
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        model_for_accessing_layers = model.base_model.model
    elif hasattr(model, "base_model"):
         model_for_accessing_layers = model.base_model

    if hasattr(model_for_accessing_layers, '__class__') and hasattr(model_for_accessing_layers.__class__, '__module__'):
        model_module_name = model_for_accessing_layers.__class__.__module__
        if "modeling_chatglm" in model_module_name:
             try:
                 GLMBlock_cls = getattr(__import__(model_module_name, fromlist=['GLMBlock']), 'GLMBlock')
                 print(f"Dynamically obtained GLMBlock class: {GLMBlock_cls.__name__} from module {model_module_name}")
             except Exception as e:
                  print(f"!!! WARNING: Failed to dynamically import GLMBlock from {model_module_name}: {e}")
    
    if GLMBlock_cls is None:
        raise RuntimeError("CRITICAL: GLMBlock class not found.")

    try:
        if hasattr(model_for_accessing_layers, 'transformer') and \
           hasattr(model_for_accessing_layers.transformer, 'encoder') and \
           hasattr(model_for_accessing_layers.transformer.encoder, 'layers'):
             layers_to_process_list = model_for_accessing_layers.transformer.encoder.layers
             print(f"Accessing layers via: model.transformer.encoder.layers. Found {len(layers_to_process_list)} layers.")
        # ... (其他获取 layers_to_process_list 的逻辑) ...
        else:
             raise RuntimeError("!!! ERROR: Could not find GLM layers list.")
    except Exception as e:
         raise RuntimeError(f"!!! ERROR accessing model layers: {e}.")

    # --- Patcher 函数 (与你提供的版本一致，只是现在只对第一层调用) ---
    # 它内部的 patched_forward 逻辑需要与 GLMBlock.forward 匹配
    # Update patch_glm_block to include scaling and stability checks
    def patch_glm_block(layer_index, layer_instance, global_model_config, device_to_use):
        if not hasattr(layer_instance, 'local_conv'):
            kernel_size = getattr(global_model_config, 'local_conv_kernel_size', 3)
            local_conv_instance = LocalConv1D(
                config=global_model_config,
                kernel_size=kernel_size,
                device=device_to_use,
                dtype=None  # Uses FP32 weights internally
            )
            setattr(layer_instance, 'local_conv', local_conv_instance)
            setattr(layer_instance.local_conv, 'layer_number', layer_index)

            dropout_rate = getattr(global_model_config, 'hidden_dropout', 0.0)
            local_conv_dropout_instance = nn.Dropout(dropout_rate).to(device_to_use)
            setattr(layer_instance, 'local_conv_dropout', local_conv_dropout_instance)

            # Add learnable scaling factor
            local_conv_scale = nn.Parameter(torch.ones(1, device=device_to_use, dtype=torch.float32))
            setattr(layer_instance, 'local_conv_scale', local_conv_scale)

            original_forward = layer_instance.forward

            def patched_forward(
                    self,
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    kv_cache=None,
                    use_cache=True,
            ):
                layernorm_output = self.input_layernorm(hidden_states)
                attention_output, kv_cache_updated = self.self_attention(
                    layernorm_output, attention_mask, rotary_pos_emb, kv_cache=kv_cache, use_cache=use_cache
                )

                # Apply LocalConv1D with scaling
                local_conv_res = self.local_conv(layernorm_output) * self.local_conv_scale
                local_conv_res_dropout = self.local_conv_dropout(local_conv_res)

                apply_res_post_ln = getattr(global_model_config, 'apply_residual_connection_post_layernorm', False)
                current_hidden_dropout = getattr(global_model_config, 'hidden_dropout', 0.0)

                if apply_res_post_ln:
                    residual1 = layernorm_output
                else:
                    residual1 = hidden_states

                attention_output_dropout = F.dropout(attention_output, p=current_hidden_dropout, training=self.training)
                if not (residual1.dtype == attention_output_dropout.dtype == local_conv_res_dropout.dtype):
                    common_dtype = attention_output_dropout.dtype
                    residual1 = residual1.to(common_dtype)
                    local_conv_res_dropout = local_conv_res_dropout.to(common_dtype)

                layernorm_input = residual1 + attention_output_dropout + local_conv_res_dropout

                # Stability check
                if not torch.isfinite(layernorm_input).all():
                    print(f"!!! ERROR: NaN/Inf in layernorm_input at layer {self.layer_number}")
                    raise RuntimeError(f"NaN/Inf detected in layer {self.layer_number}")

                layernorm_output_mlp = self.post_attention_layernorm(layernorm_input)
                mlp_res = self.mlp(layernorm_output_mlp)

                if apply_res_post_ln:
                    residual2 = layernorm_output_mlp
                else:
                    residual2 = layernorm_input

                mlp_res_dropout = F.dropout(mlp_res, p=current_hidden_dropout, training=self.training)
                final_output = residual2 + mlp_res_dropout

                return final_output, kv_cache_updated

            layer_instance.forward = types.MethodType(patched_forward, layer_instance)
            return True
        return False

    # In the main function, after loading the model
    layers_to_patch = [0, 7, 14]  # Target early and mid layers
    layers_added_count = 0

    print("-" * 30)
    print("Applying layer modification to selected GLMBlocks...")
    for layer_index, layer_instance in enumerate(layers_to_process_list):
        if layer_index in layers_to_patch and isinstance(layer_instance, GLMBlock_cls):
            if patch_glm_block(layer_index, layer_instance, model_config, target_device):
                layers_added_count += 1
                print(f"  Successfully patched GLMBlock (index {layer_index}).")
            else:
                print(f"  Skipping patch for GLMBlock (index {layer_index}), already patched or not applicable.")
        else:
            print(f"  Skipping layer {layer_index}: Not in target list or not a GLMBlock.")
    print(f"Finished patching process. Added LocalConv1D to {layers_added_count} layer(s).")
    print_model_size(model)  # Verify parameter count
    print("-" * 30)
    # ============================================================



    ft_config.training_args.generation_config.pad_token_id = (
        tokenizer.pad_token_id
    )
    ft_config.training_args.generation_config.eos_token_id = [
        tokenizer.eos_token_id,
        tokenizer.get_command('<|user|>'),
        tokenizer.get_command('<|observation|>'),
    ]

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    use_tokenizer = True
    if ft_config.peft_config is not None:
        use_tokenizer = False if ft_config.peft_config.peft_type == "LORA" else True

    trainer = Seq2SeqTrainer(
        model=model,
        args=ft_config.training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding='longest',
            return_tensors='pt',
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset.select(list(range(50))),
        tokenizer=tokenizer if use_tokenizer else None,
        compute_metrics=functools.partial(compute_metrics, tokenizer=tokenizer),
        callbacks=None,
    )

    if auto_resume_from_checkpoint.upper() == "" or auto_resume_from_checkpoint is None:
        trainer.train()
    else:
        def do_rf_checkpoint(sn):
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
            checkpoint_directory = os.path.join(output_dir, "checkpoint-" + sn)
            print("resume checkpoint from  checkpoint-" + sn)
            trainer.train(resume_from_checkpoint=checkpoint_directory)

        output_dir = ft_config.training_args.output_dir

        # resume from latest checkpoint
        if auto_resume_from_checkpoint.upper() == "YES":
            dirlist = os.listdir(output_dir)
            checkpoint_sn = 0
            # get latest checkpoint
            for checkpoint_str in dirlist:
                if checkpoint_str.find("eckpoint") > 0 and checkpoint_str.find("tmp") == -1:
                    checkpoint = int(checkpoint_str.replace("checkpoint-", ""))
                    if checkpoint > checkpoint_sn:
                        checkpoint_sn = checkpoint
            if checkpoint_sn > 0:
                do_rf_checkpoint(str(checkpoint_sn))
            else:
                trainer.train()
        else:
            # resume from specific checkpoint
            if auto_resume_from_checkpoint.isdigit() and int(auto_resume_from_checkpoint) > 0:
                do_rf_checkpoint(auto_resume_from_checkpoint)
            else:
                print(auto_resume_from_checkpoint,
                      "The specified checkpoint sn(" + auto_resume_from_checkpoint + ") has not been saved. Please search for the correct chkeckpoint in the model output directory")

    # test stage
    if test_dataset is not None:
        trainer.predict(test_dataset)


if __name__ == '__main__':
    app()