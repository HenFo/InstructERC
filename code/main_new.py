from transformers import (
    LlamaForCausalLM,
    LlamaConfig,
    LlamaTokenizer,
    get_linear_schedule_with_warmup
)
import torch
import deepspeed
import json
import math
import os
import random
from torch.utils.data import DataLoader, SequentialSampler
import numpy as np
from data_utils.data_utils import *
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, PeftModel
import argparse
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
from tqdm.auto import tqdm

# packages for ERC
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score


# evaluation function for ERC
# def get_labels_attr(dataset):
#     label_list_set = {
#         'iemocap':['happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated'],
#         'meld':['neutral', 'surprise', 'fear', 'sad', 'joyful', 'disgust', 'angry'],
#         'EmoryNLP': ['Joyful','Mad','Peaceful', 'Neutral','Sad','Powerful','Scared'],
#     }
#     label_str_set = {
#         'iemocap':"'happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated'",
#         'meld':"'neutral', 'surprise', 'fear', 'sad', 'joyful', 'disgust', 'angry'",
#         'EmoryNLP': "'Joyful','Mad','Peaceful', 'Neutral','Sad','Powerful','Scared'",
#     }

#     emotional_label_dict = {text_label:num_label for num_label, text_label in enumerate(label_list_set[dataset])}
#     emotional_label_str = label_str_set[dataset]
#     return emotional_label_dict, emotional_label_str


def report_score(dataset, golds, preds, mode='test'):
    if dataset == 'iemocap':
        target_names = ["neutral", "angry", "frustrated", "happy", "excited", "sad"]
        digits = 6
    elif dataset == 'meld':
        target_names = ["neutral", "surprise", "anger", "joy", "sadness", "fear", "disgust"]
        digits = 7
    elif dataset == 'EmoryNLP':
        target_names = ['Joyful','Mad','Peaceful', 'Neutral','Sad','Powerful','Scared']
        digits = 7


    res = {}
    res['Acc_SA'] = accuracy_score(golds, preds)
    res['F1_SA'] = f1_score(golds, preds, average='weighted', labels=target_names, zero_division=0.0)
    res['mode'] = mode
    for k, v in res.items():
        if isinstance(v, float):
            res[k] = round(v * 100, 3)

    res_matrix = metrics.classification_report(golds, preds, labels=target_names, digits=digits, zero_division=0.0)

    return res, res_matrix     

def save_model(model, tokenizer, config, args, deepspeed_config): 
    print("#### Saving Model ####")
    tokenizer.save_pretrained(args.output_dir)
    config.save_pretrained(args.output_dir)
    args.save(args.output_dir)
    model.module.save_pretrained(args.output_dir)
    # model.save_checkpoint(args.output_dir)
    with open(os.path.join(args.output_dir, "deepspeed_config.json"), 'w', encoding='utf-8') as f:
        f.write(json.dumps(deepspeed_config, indent=5))


device = torch.device("cuda")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

world_size = int(os.getenv("WORLD_SIZE", '1'))

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    choices=['iemocap','meld','EmoryNLP'],
    help="Datasets that need to be evaluated"
)

parser.add_argument(
    "--data_dir",
    type=str,
    required=True,
    help="The input data dir. Should contain the source and target files for the task.",
)
parser.add_argument(
    "--model_name_or_path",
    type=str,
    required=True,
    help="Path to pretrained model or model identifier from huggingface.co/models",
)

parser.add_argument(
    "--checkpoint_dir",
    type=str,
    default=None,
    help="Path to the fine-tuned model checkpoint.",
)

parser.add_argument(
    "--gradient_checkpointing",
    action='store_true'
)

parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Path to save trained model.",
)

parser.add_argument(
    "--mode",
    type=str,
    default="sft"
)

parser.add_argument(
    "--deepspeed_config",
    type=str,
    default="./data_utils/deepspeed_config.json",
    help="Path to save trained model.",
)

parser.add_argument(
    "--num_train_epochs",
    default=10,
    type=int,
    help="Number of training epochs.",
)
parser.add_argument(
    "--gradient_accumulation_steps",
    default=1, type=int,
    help="gradient accumulation steps",
)

parser.add_argument(
    "--warmup_ratio",
    default=0.1,
    type=float,
    help="The ratio of warmup.",
)
parser.add_argument(
    '--local_rank', 
    default=-1
)
parser.add_argument(
    "--warmup_steps",
    default=None,
    type=int
)

parser.add_argument(
    "--learning_rate",
    default=3e-5,
    type=float
)
parser.add_argument(
    "--max_seq_length",
    default=256, type=int,
    help="Max output seq length",
)
parser.add_argument(
    "--max_length",
    default=2048, type=int,
    help="Max output seq length",
)
parser.add_argument(
    '--weight_decay',
    default=0.0, type=float,
    help='weight decay when updating parameters.'
)

parser.add_argument(
    '--save_steps',
    default=1000, type=int,
)

parser.add_argument(
    "--lora",
    default=False
    # type=bool,
)
parser.add_argument(
    "--lora_dim", type=int, default=16,
)
parser.add_argument(
    "--lora_alpha", type=int, default=16,
)
parser.add_argument(
    "--lora_dropout", type=float, default=0.05,
)
parser.add_argument(
    "--lora_module_name", type=str, default='q_proj,k_proj,v_proj,query_key_value',
)
parser.add_argument(
    "--batch_size",
    default=32,
    type=int
)
parser.add_argument(
    "--eval_batch_size",
    default=4,
    type=int
)
parser.add_argument(
    "--top_k",
    default=None,
    type=int
)
parser.add_argument(
    "--num_beams",
    default=1,
    type=int
)
parser.add_argument(
    "--seed",
    default=42,
    type=int
)

parser.add_argument(
    "--top_p",
    type=float,
    default=None
)

parser.add_argument(
    "--clip_norm",
    type=float,
    default=1.0
)

parser.add_argument(
    "--temp",
    type=float,
    default=None,
    help='Temperature for model generation.'
)
parser.add_argument(
    "--do_train",
    default=False,
    # action='store_true',
)
parser.add_argument(
    "--do_eval",
    default=False,
    # action='store_true'
)
parser.add_argument(
    "--few_shot",
    type=bool,
    default=False
)

parser.add_argument(
    '--statistic_mode',
    default='True'
)
parser.add_argument(
    "--offload_optimizer",
    action='store_true'
)

parser.add_argument(
    "--emotion_prediction",
    default=False
)

parser.add_argument(
    "--data_percent",
    default=1.0,
    type=float
)

parser.add_argument(
    "--beta",
    default=1.0,
    help='hyperparameter that determining the emotion prediction weights participating in the final loss'
)

parser.add_argument(
    "--theta",
    default=1.0,
    help='hyperparameter that determining the KL divergency weights participating in the final loss'
)

parser.add_argument(
    "--class_balancing",
    default=False
)

parser.add_argument(
    "--class_balancing_alpha",
    default=0.0,
    help="reduce distance between class weights"
)

parser.add_argument(
    "--fraction_neutral",
    default=1.0,
    help="take a fraction of neutral targets for training"
)

args = parser.parse_args()
do_sample = args.top_k is not None or args.top_p is not None or args.num_beams > 1 or args.temp is not None
'''
args.top_k is not None (top_k is a parameter that limits the number of possible tokens to consider at each step)
args.top_p is not None (top_p is a parameter that limits the cumulative probability of the next token)
args.num_beams > 1(num_beams is a parameter that controls the number of beams to use for beam search)
args.temp is not None (temp is a parameter that controls the temperature of softmax distribution used for sampling)

# if any of those conditions are true, then do_sample will be set to True.
# Otherwise, it will be set to False.
# This variable is likely being used later in the code to determine whether or not to use sampling when generating text.
'''

if args.do_train == 'True':
    args.do_train = True
else:
    args.do_train = False

if args.do_eval == 'True': 
    args.do_eval = True
else:
    args.do_eval = False

if args.emotion_prediction == 'True':
    args.emotion_prediction = True
else:
    args.emotion_prediction = False

if args.lora == 'True':
    args.lora = True
else:
    args.lora = False

# eval_result_path = args.eval_result_path if args.eval_result_path is not None else args.output_dir
# os.makedirs(eval_result_path, exist_ok=True)
# pdb.set_trace()
model_args = {
    "dataset": args.dataset,
    "model_name_or_path": args.model_name_or_path,
    "checkpoint_dir": args.checkpoint_dir,
    "data_dir": args.data_dir,
    "max_seq_length": args.max_seq_length,
    "batch_size": args.batch_size,
    "eval_batch_size": args.eval_batch_size,
    "gradient_accumulation_steps": args.gradient_accumulation_steps,
    "learning_rate": args.learning_rate,
    "num_train_epochs": args.num_train_epochs,
    "save_steps": args.save_steps,
    "output_dir": args.output_dir,
    "max_length": args.max_length,
    "warmup_ratio": args.warmup_ratio,
    "warmup_steps": args.warmup_steps,
    "weight_decay": args.weight_decay,
    'data_dir': args.data_dir,
    "lora": args.lora,
    "lora_dim": args.lora_dim,
    "lora_dropout": args.lora_dropout,
    "lora_alpha": args.lora_alpha,
    "lora_module_name": args.lora_module_name,
    "num_beams": args.num_beams,
    "top_k": args.top_k,
    "top_p": args.top_p,
    "do_sample": do_sample,
    "seed": args.seed,
    "do_train": args.do_train,
    "do_eval": args.do_eval,
    "offload_optimizer": args.offload_optimizer,
    "deepspeed_config": args.deepspeed_config,
    "statistic_mode": args.statistic_mode,
    "mode": args.mode,
    "emotion_prediction": args.emotion_prediction,
    "beta": args.beta,
    "theta": args.theta,
    "gradient_checkpointing": args.gradient_checkpointing,
    "data_percent": args.data_percent,
    "class_balancing": args.class_balancing,
    "class_balancing_alpha": args.class_balancing_alpha,
    "fraction_neutral": args.fraction_neutral
}
args = ModelArgs()
# pdb.set_trace()

args.update(model_args)
# pdb.set_trace()

print(args)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir, exist_ok=True)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=args.lora_dim,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    target_modules=args.lora_module_name.split(","),
    bias='none',
)
with open(args.deepspeed_config, 'r', encoding='utf-8') as f:
    deepspeed_config = json.load(f)
deepspeed_config["train_batch_size"] = args.batch_size
deepspeed_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
if deepspeed_config["zero_optimization"]["stage"] == 3:
    deepspeed_config["zero_optimization"]['mics_shard_size'] = world_size



def getOptimizerGroup(model):
    """
    Groups the parameters of the model for optimization.

    Puts parameters into two groups:
    1. Parameters with weight decay applied.
    2. Parameters without weight decay applied (biases and layernorm weights).

    This allows applying weight decay selectively to avoid decaying biases 
    and layernorm weights.

    Parameters:
        model: The model whose parameters will be grouped.

    Returns:
        A list of parameter groups for optimization.
    """
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n
                            for nd in no_decay) and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n
                        for nd in no_decay) and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    return optimizer_grouped_parameters


# def _get_pred_input_dict(batch):
#     # print(batch["input_ids"].shape,batch["labels"].shape,batch["attention_mask"].shape)
#     batch['input_ids'] = batch['input_ids'].unsqueeze(1)
#     batch['labels'] = batch['labels'].unsqueeze(1)
#     batch['attention_mask'] = batch['attention_mask'].unsqueeze(1)
#     batch['type_token_ids'] = batch['type_token_ids'].unsqueeze(1)
#     input_ids, labels, attention_mask, type_token_ids = batch["input_ids"][0], \
#         batch["labels"][0], batch["attention_mask"][0], batch["type_token_ids"][0]
    
#     pred_input_ids, pred_labels, pred_attention_mask, pred_type_token_ids = batch["input_ids"][1], \
#         batch["labels"][1], batch["attention_mask"][1], batch["type_token_ids"][1]
     
    
#     return {
#         "input_ids": input_ids.to(device),
#         "labels": labels.to(device),
#         "attention_mask": attention_mask.to(device) 
#     },{
#         "input_ids": pred_input_ids.to(device),
#         "labels": pred_labels.to(device),
#         "attention_mask": pred_attention_mask.to(device) 
#     }

def _get_input_dict(batch):
    input_ids, labels, attention_mask, type_token_ids = batch["input_ids"], \
        batch["labels"], batch["attention_mask"], batch["type_token_ids"]
    
    return {
        "input_ids": input_ids.to(device),
        "labels": labels.to(device),
        "attention_mask": attention_mask.to(device)
    }
def _get_input_dict_emotion(batch):
    input_ids, labels, attention_mask, predict_mask = batch["input_ids"], \
        batch["labels"], batch["attention_mask"], batch["predict_mask"]
    
    normal =  {
        "input_ids": input_ids[predict_mask == 0].to(device),
        "labels": labels[predict_mask == 0].to(device),
        "attention_mask": attention_mask[predict_mask == 0].to(device)
    }

    emotion_pred = {
        "input_ids": input_ids[predict_mask == 1].to(device),
        "labels": labels[predict_mask == 1].to(device),
        "attention_mask": attention_mask[predict_mask == 1].to(device)
    }
    

    return normal, emotion_pred

## prepare LLaMA2 model
config = LlamaConfig.from_pretrained(args.model_name_or_path)
tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
print("***** Load Base Model *****")
model = LlamaForCausalLM.from_pretrained(args.model_name_or_path, device_map="cpu").half()

deepspeed_config["bfloat16"]["enabled"] = True
deepspeed_config["fp16"]["enabled"] = False


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token

if tokenizer.eos_token is None:
    tokenizer.eos_token = tokenizer.pad_token

if args.model_type == "decoder":
    tokenizer.padding_side = "left"

# if lora is used and a checkpoint is provided, load the lora model from the checkpoint
# otherwise, initialize a new lora model if it should be trained
# if the model should not be trained, the model is loaded later for evaluation
if args.lora:
    if args.checkpoint_dir is not None:
        print("***** Load Peft Checkpoint *****")
        model = PeftModel.from_pretrained(model, args.checkpoint_dir, is_trainable=args.do_train,  device_map="cpu")
    elif args.do_train:
        print("***** Create new Peft Model *****")
        model = get_peft_model(model, lora_config)

if args.gradient_checkpointing:
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

# if args.checkpoint_dir is not None and:
#     # model = load_state_dict_from_zero_checkpoint(model, args.checkpoint_dir)
#     model = 

num_parameters = get_parameter_number(model)
with open(os.path.join(args.output_dir, "model_params.json"), 'w', encoding='utf-8') as f:
    f.write(json.dumps(num_parameters, indent=5))

## prepare data
train_file = os.path.join(args.data_dir, "train.json")
dev_file = os.path.join(args.data_dir, "valid.json")
test_file = os.path.join(args.data_dir, "test.json")

train_dataset, dev_dataset, test_dataset = None, None, None
train_collator, dev_collator, test_collator = None, None, None
if args.do_train:
    df_train = read_data(train_file, percent=args.data_percent, random_seed=args.seed)
    train_dataset = Seq2SeqDataset(args, df_train, mode='train')
    train_collator = Seq2SeqCollator(args, tokenizer, mode="train")
if args.do_eval:
    df_dev = read_data(dev_file, percent=1.0, random_seed=args.seed)
    dev_dataset = Seq2SeqDataset(args, df_dev, mode='dev')
    dev_collator = Seq2SeqCollator(args, tokenizer, mode="dev")
    df_test = read_data(test_file, percent=1.0, random_seed=args.seed)
    test_dataset = Seq2SeqDataset(args, df_test, mode='dev')
    test_collator = Seq2SeqCollator(args, tokenizer, mode="dev")

stop_word_list = ['sep_token_id', 'eos_token_id', 'pad_token_id']
stop_ids = []
for stop_word in stop_word_list:
    id_ = getattr(tokenizer, stop_word)
    if id_ is not None:
        stop_ids.append(id_)
stop_criteria = KeywordsStoppingCriteria(stop_ids)
## prepare deepspeed model training
os.environ["PATH"] += ":/home/fock/code/.conda/bin"
if args.do_train:
    t_total = math.ceil(len(train_dataset) / args.batch_size) * args.num_train_epochs
    warmup_steps = math.ceil(t_total * args.warmup_ratio) if args.warmup_steps is None else args.warmup_steps
    args.warmup_steps = warmup_steps

    # if args.offload_optimizer or args.zero_shot:
    if args.offload_optimizer:
        deepspeed_config["zero_optimization"]["offload_optimizer"]["device"] = "cpu"

    optimizer_grouped_parameters = getOptimizerGroup(model=model)

    # optimizer_class = DeepSpeedCPUAdam if deepspeed_config["zero_optimization"]\
        # ["offload_optimizer"]["device"] == "cpu" else FusedAdam
    optimizer_class = AdamW
    optimizer = optimizer_class(optimizer_grouped_parameters, lr=args.learning_rate, betas=[0.9, 0.95])
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_training_steps=t_total, num_warmup_steps=warmup_steps)

    model_engine, optimizer, train_dataloader, lr_scheduler = deepspeed.initialize(
        model=model,
        training_data=train_dataset,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=deepspeed_config,
        collate_fn=train_collator
    )
    model = model_engine    
    should_save = True
elif args.do_eval:
    print("***** Load Peft Model for evaluation *****")
    model = PeftModel.from_pretrained(model, args.output_dir, device_map="cpu")
    model.merge_and_unload(progressbar=True)
    model_engine = deepspeed.init_inference(
        model,
        tensor_parallel={"tp_size": world_size},
        replace_with_kernel_inject=True,
        dtype=torch.half,
    )
    model = model_engine

if __name__ == "__main__":
    eval_score_list = []
    # emotional_label_dict, emotional_label_str = get_labels_attr(dataset=args.dataset)
    if type(args.theta) == str:
        args.theta = torch.tensor(eval(args.theta))
        args.theta.to(device)
    if type(args.beta) == str:
        args.beta = torch.tensor(eval(args.beta))
        args.beta.to(device)
    # pdb.set_trace()
    if args.do_train:
        global_steps = 0
        best_f1_score = 0
        for epoch in range(args.num_train_epochs):
            model.train()
            batch_iterator = tqdm(
                train_dataloader,
                desc=f"Running epoch{epoch} / {args.num_train_epochs}",
                disable=False,
                mininterval=0,
            )
            for step, batch in enumerate(batch_iterator):
                if args.emotion_prediction:
                    batch_n, batch_e = _get_input_dict_emotion(batch)
                    sample_weights = batch["sample_weight"]
                    outputs_n = model(**batch_n)
                    outputs_e = model(**batch_e)
                    loss = outputs_n.loss * sample_weights + args.beta * outputs_e.loss * sample_weights
                else:
                    batch = _get_input_dict(batch)
                    outputs = model(**batch)
                    loss = outputs.loss

                model.backward(loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                model.step()
                    
                current_loss = loss.item()
                batch_iterator.set_description(
                    f"Epochs {epoch}/{args.num_train_epochs}. Current Loss: {current_loss:9.7f}"
                )

            # model starts evaluation
            model.eval()
            targets = list(df_dev["output"])
            eval_sampler = SequentialSampler(dev_dataset)
            eval_dataloader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, sampler=eval_sampler, collate_fn=dev_collator, num_workers=8)
            all_outputs = []

            preds_for_eval_path = os.path.join(args.output_dir, f"preds_for_eval_{epoch}.text")
            print("\n*****    Evaluating  *****\n")
            eval_inputs_iter = []
            for eval_step, eval_batch in enumerate(tqdm(eval_dataloader)):
                eval_batch = eval_batch.to(device)
                eval_inputs_iter.extend(eval_batch["input_ids"])
                with torch.no_grad():
                    outputs = model.generate(
                        **eval_batch,
                        max_length=args.max_length,
                    )
                outputs[outputs[:, :] < 0] = tokenizer.pad_token_id
                all_outputs.extend(outputs)
            ins = tokenizer.batch_decode(eval_inputs_iter, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            # all_outputs
            outs = tokenizer.batch_decode(all_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
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
            if args.statistic_mode == "True":

                score, res_matrix = report_score(dataset=args.dataset, golds=targets, preds=all_answers)
                print(f"##### Evaluation after Epoch {epoch} with F1: {score} #####")

                # statisics of model's output
                with open(preds_for_eval_path, 'w', encoding='utf-8') as f:
                    f.write(json.dumps(score))
                    f.write(f'\n{res_matrix}\n\n')
                    f.write(json.dumps(preds_for_eval, indent=5, ensure_ascii=False))

                if best_f1_score < score['F1_SA']:
                    best_f1_score = score['F1_SA']
                    save_model(model, tokenizer, config, args, deepspeed_config)
            
            if args.statistic_mode != 'True':
                with open(preds_for_eval_path, 'w', encoding='utf-8') as f:
                    f.write(json.dumps(preds_for_eval, indent=5, ensure_ascii=False))
                save_model(model, tokenizer, config, args, deepspeed_config)


    if not args.do_train and args.do_eval:
        # model starts to evaluation
        model.eval()
        targets = list(df_test["output"])
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, batch_size=1, sampler=test_sampler, collate_fn=test_collator, num_workers=8)
        all_outputs = []

        preds_for_eval_path = os.path.join(args.output_dir, f"preds_for_eval.text")
        print("\n*****    Evaluating  *****\n")
        eval_inputs_iter = []
        for eval_step, eval_batch in enumerate(tqdm(test_dataloader)):
            eval_batch = eval_batch.to(device)
            eval_inputs_iter.extend(eval_batch["input_ids"])
            with torch.no_grad():
                outputs = model.generate(
                    **eval_batch,
                    max_length=args.max_length,
                )
            outputs[outputs[:, :] < 0] = tokenizer.pad_token_id
            all_outputs.extend(outputs)
        ins = tokenizer.batch_decode(eval_inputs_iter, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        # all_outputs
        outs = tokenizer.batch_decode(all_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
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
        with open(preds_for_eval_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(score))
            f.write(f'\n{res_matrix}\n\n')
            f.write(json.dumps(preds_for_eval, indent=5, ensure_ascii=False))
            