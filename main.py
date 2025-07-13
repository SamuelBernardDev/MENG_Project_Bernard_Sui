import argparse
import yaml

from utils.data_loader import ExcelDatasetTimeSeries
from utils.help_training import cross_validate, train_full


def main():
    parser = argparse.ArgumentParser(description="Full training pipeline")
    parser.add_argument("--model_type", choices=["lstm", "conv"], default="lstm")
    parser.add_argument("--skip_cv", action="store_true", help="Skip cross-validation step")
    args = parser.parse_args()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    full_ds = ExcelDatasetTimeSeries(
        root_folder=config["data"]["root_folder"],
        columns=config["data"]["columns"],
        derivative_columns=config["data"].get("derivative_columns", []),
        min_required_length=config["data"].get("min_required_length", 50),
        stats_path=None,
        is_train=False,
    )
    indices = list(range(len(full_ds)))
    labels = [full_ds.labels[i] for i in indices]

    if not args.skip_cv:
        cross_validate(args.model_type, config, indices, labels)

    train_full(args.model_type, config)


if __name__ == "__main__":
    main()
