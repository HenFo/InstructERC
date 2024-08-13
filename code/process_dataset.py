import pandas as pd
from typing import List
import os
import json
import argparse


label_mapping = {
    "iemocap": {
        "neu": "neutral",
        "ang": "angry",
        "fru": "frustrated",
        "hap": "happy",
        "exc": "excited",
        "sad": "sad"
    },
    "meld": {
        "neutral": "neutral",
        "surprise": "surprise",
        "anger": "anger",
        "joy": "joy",
        "sadness": "sadness",
        "fear": "fear",
        "disgust": "disgust"
    },
}


def transform_speaker_to_id(df: pd.DataFrame) -> pd.DataFrame:
    name_to_id = {name: i for i, name in enumerate(df["Speaker"].unique())}
    df["Speaker"] = df["Speaker"].apply(lambda name: f"Speaker_{name_to_id[name]}")
    return df


def prepare_dataset(path: str) -> pd.DataFrame:
    ds = pd.read_csv(path, index_col=0).reset_index(drop=True)
    ds["Utterance"] = ds["Utterance"].str.replace("’", "")
    ds["Utterance"] = ds["Utterance"].str.replace("‘", "'")
    ds = transform_speaker_to_id(ds)
    ds = ds.groupby("Dialogue_ID").apply(transform_speaker_to_id).reset_index(drop=True)
    ds["prompt"] = ds["Speaker"] + ': "' + ds["Utterance"] + '"'
    return ds


def create_window_view(df: pd.DataFrame, window: int = 5) -> List[pd.DataFrame]:
    groups = df.groupby("Dialogue_ID")
    dialogue_windows = []
    for _, group in groups:
        for i, (_, target_row) in enumerate(
            group.sort_values("Utterance_ID").iterrows()
        ):
            start = max(0, i - window)
            end = i + 1
            history = group.iloc[start:end]
            dialogue_windows.append(history)
    return dialogue_windows


def generate_inputs(dialogs: List[pd.DataFrame], emotional_labels: str) -> List[dict]:
    inputs = []
    for dialog in dialogs:
        dialog_chain = " \t ".join(dialog["prompt"])
        target = dialog.iloc[-1]
        instruction = f"Please select the emotional label of <{target['prompt']}> from <{emotional_labels}>:"
        prompt = f"Now you are expert of sentiment and emotional analysis. The following conversation noted between '### ###' involves several speaker. ### {dialog_chain} ### {instruction}"
        inputs.append({"input": prompt, "target": target["Emotion"]})
    return inputs


def generate_emotion_prediction(
    dialogs: List[pd.DataFrame], emotional_labels: str
) -> List[dict]:
    inputs = []
    for dialog in dialogs:
        if len(dialog) > 1:
            dialog_chain = " \t ".join(dialog["prompt"].iloc[:-1])
            target = dialog.iloc[-1]
            instruction = f"Based on the above historical utterances, next utterance is spoken by <{target['Speaker']}>, please predict the emotion states of <{target['Speaker']}> from <{emotional_labels}>:"
            prompt = f"Now you are expert of sentiment and emotional analysis. The following conversation noted between '### ###' involves several speaker. ### {dialog_chain} ### {instruction}"
            inputs.append({"input": prompt, "target": target["Emotion"]})
        else:
            inputs += generate_inputs([dialog], emotional_labels)
    return inputs


def generate_speaker_ident_task(
    dialogs: List[pd.DataFrame], speaker_labels: str
) -> List[dict]:
    inputs = []
    for dialog in dialogs:
        dialog_chain = " \t ".join(dialog["prompt"].iloc[:-1])
        target = dialog.iloc[-1]
        instruction = f"Please select the Speaker label of the next utterance <Speaker: {target['Utterance']}> from <{speaker_labels}>:"
        prompt = f"Now you are expert of sentiment and emotional analysis. The following conversation noted between '### ###' involves several speaker. ### {dialog_chain} ### {instruction}"
        inputs.append({"input": prompt, "target": target["Speaker"]})
    return inputs


def merge_inputs(inputs: List[List[dict]], joiner: str = "***") -> List[dict]:
    merged_prompts = []
    for prompts in zip(*inputs):
        target = prompts[0]["target"]
        input_prompts = list(map(lambda x: x["input"], prompts))
        merged_prompts.append({"input": joiner.join(input_prompts), "target": target})
    return merged_prompts


def save_dataset(ds: List[dict], dataset: str, stage: str, task: str):
    path = os.path.join("processed_data", dataset, task)
    name = stage + ".json"
    if not os.path.exists(path):
        os.makedirs(path)

    ds = list(map(lambda x: json.dumps(x) + "\n", ds))
    with open(os.path.join(path, name), "w") as f:
        f.writelines(ds)

    return path


def process_dataset(
    dataset,
    window=20,
    speaker_task="True",
    predictions="True",
    autoregressive_emotion="True",
):
    def map_emotions(df:pd.DataFrame):
        df["Emotion"] = df["Emotion"].map(label_mapping[dataset])
        return df

    if dataset == "meld":
        ds_path = "original_data/meld/RAW/"
        ds_train = prepare_dataset(os.path.join(ds_path, "train_sent_emo.csv"))
        ds_train = map_emotions(ds_train)
        ds_dev = prepare_dataset(os.path.join(ds_path, "dev_sent_emo.csv"))
        ds_dev = map_emotions(ds_dev)
        ds_test = prepare_dataset(os.path.join(ds_path, "test_sent_emo.csv"))
        ds_test = map_emotions(ds_test)

    if dataset == "iemocap":
        ds_path = "original_data/iemocap/"
        ds = prepare_dataset(os.path.join(ds_path, "iemocap.csv"))
        ds = ds[ds["Emotion"].isin(label_mapping[dataset])]
        ds = map_emotions(ds)
        split = pd.read_csv(os.path.join(ds_path, "iemocap_split.csv"))
        ds_train = ds[
            ds["Dialogue_ID"].isin(split[split["Split"] == "train"]["Dialogue_ID"])
        ]
        ds_dev = ds[
            ds["Dialogue_ID"].isin(split[split["Split"] == "dev"]["Dialogue_ID"])
        ]
        ds_test = ds[
            ds["Dialogue_ID"].isin(split[split["Split"] == "test"]["Dialogue_ID"])
        ]

    for stage, ds in (("train", ds_train), ("valid", ds_dev), ("test", ds_test)):
        windowed_dialogs = create_window_view(ds, window=window)
        assert not any(map(lambda x: len(x) == 0, windowed_dialogs))

        if speaker_task == "True":
            speaker_labels = ", ".join(ds["Speaker"].unique())
            speaker_task_data = generate_speaker_ident_task(
                windowed_dialogs, speaker_labels
            )
            path = save_dataset(speaker_task_data, dataset, stage, "speaker")
        else:
            emotional_labels = ", ".join(label_mapping[dataset].values())
            windowed_inputs = generate_inputs(windowed_dialogs, emotional_labels)
            task = "window"
            if predictions == "True":
                if stage == "train":
                    emotion_prediction = generate_emotion_prediction(
                        windowed_dialogs, emotional_labels
                    )
                    windowed_inputs = merge_inputs(
                        [windowed_inputs, emotion_prediction]
                    )
                task = os.path.join("predict", task)
            path = save_dataset(windowed_inputs, dataset, stage, task)

    return path


parser = argparse.ArgumentParser(description="Data processing script")
parser.add_argument("--dataset", type=str, default="meld", help="Dataset name or path")
parser.add_argument(
    "--historical_window", type=int, default=20, help="Historical window size"
)
parser.add_argument(
    "--speaker_task",
    type=str,
    default="add speaker_task to main task",
    help="Speaker task type",
)
parser.add_argument(
    "--emotion_prediction",
    type=str,
    default="add emotion_prediction to main task",
    help="Emotion prediction task type",
)
parser.add_argument(
    "--autoregressive_emotion",
    type=str,
    default="False",
    help="True, to include the emotional history in the input",
)
args = parser.parse_args()

# args = argparse.Namespace(dataset='iemocap', historical_window=10, speaker_task='False', emotion_prediction='True', autoregressive_emotion='False')
# os.chdir("..")

# Process data
processed_data_path = process_dataset(
    dataset=args.dataset,
    window=args.historical_window,
    speaker_task=args.speaker_task,
    predictions=args.emotion_prediction,
    autoregressive_emotion=args.autoregressive_emotion,
)

print(processed_data_path)
