import argparse
import os
import glob
import json
import yaml
import torch
import pandas as pd

from utils.loading_data import load_excel
from utils.preprocess import normalize, interpolate_df
from src.models.LSTMModel import LSTMClassifier
from src.models.ConvSeqModel import ConvSequenceClassifier


def load_model(model_type: str, config: dict, checkpoint: str):
    """Instantiate and load model weights."""
    input_size = len(config["data"]["columns"]) + len(config["data"].get("derivative_columns", []))
    num_classes = 2  # binary classification by default

    if model_type.lower() == "lstm":
        model = LSTMClassifier(
            input_size=input_size,
            hidden_size=config["model"]["hidden_size"],
            num_layers=config["model"]["num_layers"],
            num_classes=num_classes,
        )
    elif model_type.lower() == "conv":
        model = ConvSequenceClassifier(
            input_size=input_size,
            num_classes=num_classes,
            conv_channels=config["model"].get("conv_channels", 32),
            conv_dilations=config["model"].get("conv_dilations", [1, 2, 4]),
            rnn_hidden=config["model"].get("hidden_size", 64),
            rnn_layers=config["model"].get("num_layers", 1),
            sequence_module=config["model"].get("sequence_module", "lstm"),
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    return model


def main(args):
    # load config
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # load normalization statistics
    with open(config["data"]["stats_path"]) as f:
        stats = yaml.safe_load(f) if config["data"]["stats_path"].endswith(".yaml") else json.load(f)
    global_min = pd.Series(stats["min"])
    global_max = pd.Series(stats["max"])
    min_seq_len = stats["min_seq_len"]

    checkpoint = args.checkpoint or config["train"]["model_save_path"]
    model = load_model(args.model_type, config, checkpoint)

    test_folder = args.test_folder
    test_files = glob.glob(os.path.join(test_folder, "*/*.xls"))

    for fpath in test_files:
        try:
            df = load_excel(fpath, config["data"]["columns"], time_format="%I:%M:%S%p")
            for col in config["data"].get("derivative_columns", []):
                if col in df.columns:
                    df[f"{col}_rate"] = df[col].diff().fillna(0)

            df_norm = normalize(df, global_min, global_max)
            df_final = interpolate_df(df_norm, interval=1, max_length=min_seq_len)
            sequence = torch.tensor(df_final.values.astype("float32")).unsqueeze(0)

            with torch.no_grad():
                output = model(sequence)
                probs = torch.softmax(output, dim=1).squeeze()
                pred = torch.argmax(probs).item()
                conf = probs[pred].item()

            print(f"File: {os.path.basename(fpath)} | Prediction: {pred} | Confidence: {conf:.2%}")
        except Exception as e:
            print(f"Failed to process {os.path.basename(fpath)}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequence model inference")
    parser.add_argument("--model_type", choices=["lstm", "conv"], default="lstm", help="Model architecture used during training")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model weights")
    parser.add_argument("--test_folder", type=str, default="data/test", help="Folder with test samples")
    main(parser.parse_args())
