
from peft import PeftModel
from tqdm.auto import tqdm
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from torch.utils.data import DataLoader, SequentialSampler
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os
from data_utils.data_utils import *
import random



ADAPTER_PATH = "/home/fock/code/InstructERC/experiments/LLaMA2/lora/meld/True_two_unbalanced"
MODEL_PATH = "/home/fock/code/InstructERC/LLM_bases/LLaMA2"
DATA_DIR = "/home/fock/code/InstructERC/processed_data/meld/predict/window"


model_args = {
    "eval_batch_size": 8,
    "max_length": 1024,
    "num_beams": 1,
    "seed":42,
    "mode": "dev",
    "data_percent": 1
}
args = ModelArgs()
args.update(model_args)


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


device = torch.device("cuda")
tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)
model = LlamaForCausalLM.from_pretrained(MODEL_PATH, device_map="cpu").half()
model = PeftModel.from_pretrained(model, ADAPTER_PATH, is_trainable=False,  device_map="cpu").to(device)
# model.merge_and_unload(progressbar=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token

tokenizer.padding_side = "left"


dev_file = os.path.join(DATA_DIR, "valid.json")

df_dev = read_data(dev_file, percent=.25, random_seed=args.seed)
dev_dataset = Seq2SeqDataset(args, df_dev, mode='dev')
dev_collator = Seq2SeqCollator(args, tokenizer, mode="dev")



model.eval()
test_sampler = SequentialSampler(dev_dataset)
eval_dataloader = DataLoader(dev_dataset, batch_size=1, sampler=test_sampler, collate_fn=dev_collator, num_workers=8)
all_outputs = []
eval_inputs_iter = []
for eval_step, eval_batch in enumerate(tqdm(eval_dataloader)):
    eval_batch = eval_batch.to(device)
    eval_inputs_iter.extend(eval_batch["input_ids"])

    with torch.no_grad():
        outputs = model.generate(
            **eval_batch,
            num_beams=args.num_beams,
            max_length=args.max_length,
            num_return_sequences=1
        )
    all_outputs.extend(outputs)








