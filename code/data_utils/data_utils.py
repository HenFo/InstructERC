import pandas as pd
import logging
import os

import pandas as pd
import torch
from torch.utils.data import Dataset
import json
from dataclasses import dataclass, asdict
from transformers import StoppingCriteria
from sklearn.utils import class_weight
import numpy as np
from pprint import pprint
from transformers import LlamaTokenizer
from typing import List



logger = logging.getLogger(__name__)

def read_data(file_name, percent, random_seed):
    f = open(file_name, 'r', encoding='utf-8').readlines()
    data = [json.loads(d) for d in f]
    inputs = []
    targets = []
    for index, d in enumerate(data):
        if pd.isnull(d['target']) or pd.isna(d['target']):
            continue
        inputs.append(d['input'])
        targets.append(d['target'])
    dict_ = {'input': inputs, 'output': targets}
    df_data = pd.DataFrame(dict_)
    df_data.dropna(axis=0, how='any')

    # randomly extract *percent of the data
    num_samples = int(len(df_data)*percent)
    print(f'the number of num_samples is {len(df_data)}')
    df_data = df_data.sample(n=num_samples, random_state=random_seed)
    print(f'the number of num_samples is {len(df_data)}')

    return df_data


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_ids:list):
        self.keywords = keywords_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids[0][-1] in self.keywords:
            return True
        return False
    
def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}


class Seq2SeqDataset(Dataset):
    def __init__(self, args:"ModelArgs", data, mode):
        self.mode = mode
        if not args.emotion_prediction:
            inputs = list(data["input"])
            outputs = list(data['output'])
            self.examples = [[i, o] for i, o in zip(inputs, outputs)]
        elif mode == 'dev':
            inputs = list(data["input"])
            outputs = list(data['output'])
            self.examples = [[i, o] for i, o in zip(inputs, outputs)]
        else:
            inputs = list(data["input"])
            inputs = [i.split('***') for i in inputs]
            outputs = list(data['output'])
            self.examples = [[i[0], i[1], o] for i, o in zip(inputs, outputs)]

        outputs = [e[-1] for e in self.examples]
        labels = np.array(list(set(outputs)))
        balancing = "balanced" if args.class_balancing and not mode == "dev" else None
        class_weights = class_weight.compute_class_weight(balancing, classes=labels, y=outputs)
        class_weights = class_weights * self.inv_sigmoid(class_weights, alpha = float(args.class_balancing_alpha))
        self.class_weights = {l:w for l, w in zip(labels, class_weights)}
        pprint(self.class_weights)

    def inv_sigmoid(self, x, alpha=1.0):
        return 2 - (1 / (0.5 + 0.5*np.exp(-alpha*x)))


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        if self.mode == 'dev':
            return example
        return example, self.class_weights[example[-1]]

class Seq2SeqCollator(object):
    def __init__(self, args, tokenizer, mode="train"):
        self.tokenizer = tokenizer
        self.args = args    
        self.mode = mode

    def __call__(self, batch):
        if self.mode == "dev":
            batch = [d[0] for d in batch]
            inputs = self.tokenizer(batch, max_length=self.args.max_length, truncation=True, padding=True, return_tensors='pt')
        else:
            inputs = preprocess_data_batch(batch, self.tokenizer, self.args)

        return inputs


def preprocess_data_batch(data, tokenizer:LlamaTokenizer, args: "ModelArgs"):
    inputs = [d[0][0] for d in data]
    batch_sample_weights = [d[1] for d in data]
    targets = [d[0][-1] for d in data]
    
    inputs = tokenizer(
        inputs,
        max_length=args.max_length - 1,
        truncation=True
    )

    targets = tokenizer(
        targets,
        add_special_tokens=False,
    )
    input_ids = inputs['input_ids']
    target_ids = targets['input_ids']

    pred_mask = [0] * len(input_ids)
    if args.emotion_prediction:
        inputs_pred = [d[0][1] for d in data]
        pred_inputs = tokenizer(
            inputs_pred,
            max_length=args.max_length - 1,
            truncation=True
        )
        pred_ids = pred_inputs["input_ids"]
        pred_mask += [1] * len(pred_ids)

        input_ids += pred_ids
        target_ids *= 2

    max_batch_length = max(list(map(lambda x: min(args.max_length, len(x[0]+x[1])+1), zip(input_ids, target_ids))))

    batch_input_ids = []
    batch_attention_masks = []
    batch_labels = []
    batch_type_token_ids = []
    for input_ids_i, target_ids_i in zip(input_ids, target_ids):
        concat:List[int] = (input_ids_i + target_ids_i)[:args.max_length-1] + [tokenizer.eos_token_id]
        padding = [0] * (max_batch_length - len(concat))
        type_ids = padding + ([0] * len(input_ids_i) + [1] * (len(target_ids_i)))[:args.max_length-1] + [1]
        concat = padding + concat

        input_id = torch.Tensor(concat).long()
        attention_mask = torch.ones_like(input_id).long()
        attention_mask[input_id == 0] = 0
        type_token_ids = torch.Tensor(type_ids).long()
        labels = input_id.clone().contiguous().detach()
        labels[type_token_ids == 0] = -100

        batch_input_ids.append(input_id)
        batch_attention_masks.append(attention_mask)
        batch_type_token_ids.append(type_token_ids)
        batch_labels.append(labels)
                     

    batch_sample_weight = torch.mean(torch.Tensor(batch_sample_weights))
    return {
        "input_ids": torch.stack(batch_input_ids),
        "attention_mask": torch.stack(batch_attention_masks),
        "type_token_ids": torch.stack(batch_type_token_ids),
        "labels": torch.stack(batch_labels),
        "predict_mask": torch.Tensor(pred_mask).long(),
        "sample_weight": batch_sample_weight
    }
    


@dataclass
class ModelArgs:
    model_type: str = "decoder"
    model_name_or_path: str = "YOUR_MODEL_PATH"
    checkpoint_dir: str = None
    output_dir: str = "YOUR_OUTPUT_DIR_PATH"
    data_dir: str = "DATASET_PATH"
    deepspeed_config = "./deepspeed_config.json"
    do_train: bool = True
    do_eval: bool = False
    emotion_prediction: bool = False
    num_train_epochs = 10
    warmup_ratio: float = 0.1
    warmup_steps: int = None
    save_steps: int = 500
    weight_decay: float = 0.0
    max_seq_length: int = 96
    max_length: int = 32
    num_beams: int = 1
    do_sample: bool = False
    top_k: int = None
    top_p: float = None
    learning_rate: float = 3e-5
    preprocess_inputs: bool = True
    clip_norm: float = 1.0
    open_ended: bool = False
    batch_size: int = 32
    eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    lora: bool = True
    lora_dim: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_module_name: str = 'q_proj,k_proj,v_proj,query_key_value'
    seed: int = 42
    offload_optimizer: bool = False
    deepspeed_config: str = None
    zero_shot: bool = False
    mode: str = "sft"
    gradient_checkpointing: bool = False
    class_balancing: bool = False
    class_balancing_alpha: float = 0.2

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "model_args.json"), "w") as f:
            f.write(json.dumps(asdict(self), indent=5))

    def update(self, new_values):
        if isinstance(new_values, dict):
            for key, value in new_values.items():
                setattr(self, key, value)
        else:
            raise (TypeError(f"{new_values} is not a Python dict."))