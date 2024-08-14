import argparse
import json
import os
import random

import numpy as np
import torch
from data_utils.data_utils import *
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, SequentialSampler
from tqdm.auto import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer

# ADAPTER_PATH = "/home/fock/code/InstructERC/experiments/LLaMA2-base/lora_16/iemocap/True_two"
# MODEL_PATH = "/home/fock/code/InstructERC/LLM_bases/LLaMA2-base"
# DATA_DIR = "/home/fock/code/InstructERC/processed_data/iemocap/predict/window"


def report_score(dataset, golds, preds, mode="test"):
    if dataset == "iemocap":
        target_names = ["neutral", "angry", "frustrated", "happy", "excited", "sad"]
        digits = 6
    elif dataset == "meld":
        target_names = [
            "neutral",
            "surprise",
            "anger",
            "joy",
            "sadness",
            "fear",
            "disgust",
        ]
        digits = 7
    elif dataset == "EmoryNLP":
        target_names = [
            "Joyful",
            "Mad",
            "Peaceful",
            "Neutral",
            "Sad",
            "Powerful",
            "Scared",
        ]
        digits = 7

    res = {}
    res["Acc_SA"] = accuracy_score(golds, preds)
    res["F1_SA"] = f1_score(
        golds, preds, average="weighted", labels=target_names, zero_division=0.0
    )
    res["mode"] = mode
    for k, v in res.items():
        if isinstance(v, float):
            res[k] = round(v * 100, 3)

    res_matrix = classification_report(
        golds, preds, labels=target_names, digits=digits, zero_division=0.0
    )

    return res, res_matrix


argParser = argparse.ArgumentParser()

argParser.add_argument(
    "--dataset", type=str, default="iemocap", help="Dataset name or path"
)
argParser.add_argument("--model_name_or_path", type=str)
argParser.add_argument("--output_dir", type=str)
argParser.add_argument("--data_dir", type=str)
argParser.add_argument("--eval_batch_size", type=int, default=8)
argParser.add_argument("--max_length", type=int, default=1024)
argParser.add_argument("--seed", type=int, default=42)
argParser.add_argument("--data_percent", type=float, default=1.0)

model_args = argParser.parse_args()

args = ModelArgs()
args.update(model_args.__dict__)


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


device = torch.device("cuda")
tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
model = LlamaForCausalLM.from_pretrained(
    args.model_name_or_path, device_map="cpu"
).half()
model = PeftModel.from_pretrained(
    model, args.output_dir, is_trainable=False, device_map="cpu"
).to(device)
model = model.merge_and_unload(progressbar=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token

tokenizer.padding_side = "left"


test_file = os.path.join(args.data_dir, "test.json")

df_test = read_data(test_file, percent=1.0, random_seed=args.seed)
test_dataset = Seq2SeqDataset(args, df_test, mode="dev")
test_collator = Seq2SeqCollator(args, tokenizer, mode="dev")
targets = list(df_test["output"])


model.eval()
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=1,
    sampler=test_sampler,
    collate_fn=test_collator,
    num_workers=8,
)
all_outputs = []
test_inputs_iter = []
for test_step, test_batch in enumerate(tqdm(test_dataloader)):
    test_batch = test_batch.to(device)
    test_inputs_iter.extend(test_batch["input_ids"])

    with torch.no_grad():
        outputs = model.generate(
            **test_batch,
            max_length=args.max_length,
        )
    outputs[outputs[:, :] < 0] = tokenizer.pad_token_id
    all_outputs.extend(outputs)

ins = tokenizer.batch_decode(
    test_inputs_iter, skip_special_tokens=True, clean_up_tokenization_spaces=True
)
# all_outputs
outs = tokenizer.batch_decode(
    all_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
)
preds_for_eval = []
all_answers = [output.replace(prompt, "").strip() for prompt, output in zip(ins, outs)]

for index, (input_prompt, answer) in enumerate(zip(ins, all_answers)):
    this_eval_instance = {
        "index": index,
        "input": input_prompt,
        "output": answer,
        "target": targets[index],
    }
    preds_for_eval.append(this_eval_instance)

score, res_matrix = report_score(dataset=args.dataset, golds=targets, preds=all_answers)
print(f"##### Evaluation F1: {score} #####")

# statisics of model's output
preds_for_eval_path = os.path.join(args.output_dir, "preds_for_eval.text")
with open(preds_for_eval_path, "w", encoding="utf-8") as f:
    f.write(json.dumps(score))
    f.write(f"\n{res_matrix}\n\n")
    f.write(json.dumps(preds_for_eval, indent=5, ensure_ascii=False))
