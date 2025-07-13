import yaml

from utils.data_loader import ExcelDatasetTimeSeries
from utils.help_training import cross_validate, train_full


def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    model_type = config.get("model_type", "lstm")
    skip_cv = config.get("skip_cv", False)

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

    best_threshold = 0.5
    if not skip_cv:
        best_threshold = cross_validate(model_type, config, indices, labels)

    train_full(model_type, config, threshold=best_threshold)


if __name__ == "__main__":
    main()
