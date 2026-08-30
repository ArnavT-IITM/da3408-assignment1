"""Train and compare MLP classifiers on MNIST with MLflow.

Running this file without arguments performs the six-run sweep required for
DA3408 Assignment 1, Question 2. Command-line arguments can narrow or extend
the sweep without editing the source code.
"""

import argparse
import os
import random
import uuid
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
DEFAULT_EXPERIMENT_NAME = "mnist-mlp-q2"
DEFAULT_LEARNING_RATES = [0.001, 0.003, 0.01]
DEFAULT_BATCH_SIZES = [32, 128]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run an MLflow-tracked MNIST MLP hyperparameter sweep."
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI),
        help=(
            "MLflow tracking server URI. Defaults to MLFLOW_TRACKING_URI, "
            f"or {DEFAULT_TRACKING_URI} if the variable is unset."
        ),
    )
    parser.add_argument(
        "--experiment-name",
        default=os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME),
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        default=DEFAULT_LEARNING_RATES,
        help="One or more learning rates (default: 0.001 0.003 0.01).",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=DEFAULT_BATCH_SIZES,
        help="One or more batch sizes (default: 32 128).",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Directory in which torchvision stores MNIST.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.hidden_dim < 1:
        raise ValueError("--hidden-dim must be at least 1")
    if any(lr <= 0 for lr in args.learning_rates):
        raise ValueError("All learning rates must be positive")
    if any(batch_size < 1 for batch_size in args.batch_sizes):
        raise ValueError("All batch sizes must be positive")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_mnist(data_dir, seed):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    full_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    return random_split(
        full_dataset,
        [50_000, 10_000],
        generator=torch.Generator().manual_seed(seed),
    )


class MLP(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10),
        )

    def forward(self, inputs):
        return self.network(inputs)


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total


def train_and_log(
    train_dataset,
    val_dataset,
    learning_rate,
    batch_size,
    hidden_dim,
    epochs,
    seed,
    sweep_id,
):
    set_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    model = MLP(hidden_dim=hidden_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    run_name = f"mlp-lr-{learning_rate:g}-bs-{batch_size}"

    with mlflow.start_run(run_name=run_name) as run:
        # Parameters are fixed inputs and are logged once per run.
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("hidden_dim", hidden_dim)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("dataset", "MNIST")
        mlflow.log_param("seed", seed)

        mlflow.set_tag("sweep_id", sweep_id)
        mlflow.set_tag("device", str(DEVICE))

        best_val_accuracy = 0.0
        best_epoch = 0
        final_train_loss = 0.0
        final_val_accuracy = 0.0

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            total_examples = 0

            for images, labels in train_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()
                logits = model(images)
                loss = loss_function(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * labels.size(0)
                total_examples += labels.size(0)

            final_train_loss = total_loss / total_examples
            final_val_accuracy = evaluate(model, val_loader)

            if final_val_accuracy > best_val_accuracy:
                best_val_accuracy = final_val_accuracy
                best_epoch = epoch + 1

            # step=epoch creates the time-series curves used to assess overfitting.
            mlflow.log_metric("train_loss", final_train_loss, step=epoch)
            mlflow.log_metric("val_accuracy", final_val_accuracy, step=epoch)

            print(
                f"{run_name} | epoch={epoch + 1}/{epochs} "
                f"| train_loss={final_train_loss:.4f} "
                f"| val_accuracy={final_val_accuracy:.4f}"
            )

        # Summary metrics are convenient columns in MLflow's comparison table.
        mlflow.log_metric("best_val_accuracy", best_val_accuracy)
        mlflow.log_metric("best_epoch", best_epoch)
        mlflow.log_metric("final_val_accuracy", final_val_accuracy)
        mlflow.log_metric("final_train_loss", final_train_loss)

        print(
            f"Completed {run_name} | run_id={run.info.run_id} "
            f"| best_val_accuracy={best_val_accuracy:.4f}"
        )
        return run.info.run_id, best_val_accuracy


def print_sweep_summary(experiment_name, sweep_id):
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string=f"tags.sweep_id = '{sweep_id}'",
        order_by=["metrics.best_val_accuracy DESC"],
    )

    columns = [
        "run_id",
        "tags.mlflow.runName",
        "params.learning_rate",
        "params.batch_size",
        "metrics.best_val_accuracy",
        "metrics.best_epoch",
        "metrics.final_val_accuracy",
        "metrics.final_train_loss",
    ]

    print("\nRuns from this invocation:")
    print(runs[columns].to_string(index=False))

    analysis_df = runs.copy()
    analysis_df["learning_rate"] = analysis_df[
        "params.learning_rate"
    ].astype(float)
    analysis_df["batch_size"] = analysis_df["params.batch_size"].astype(int)

    print("\nMean best validation accuracy by learning rate:")
    print(
        analysis_df.groupby("learning_rate")["metrics.best_val_accuracy"].mean()
    )

    print("\nMean best validation accuracy by batch size:")
    print(analysis_df.groupby("batch_size")["metrics.best_val_accuracy"].mean())

    best_run = runs.iloc[0]
    print(
        "\nBest run: "
        f"{best_run['run_id']} "
        f"({best_run['tags.mlflow.runName']}, "
        f"best_val_accuracy={best_run['metrics.best_val_accuracy']:.4f})"
    )


def main():
    args = parse_args()
    validate_args(args)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Experiment: {args.experiment_name}")
    print(f"Device: {DEVICE}")

    train_dataset, val_dataset = load_mnist(args.data_dir, args.seed)
    sweep_id = uuid.uuid4().hex[:12]

    combinations = [
        (learning_rate, batch_size)
        for learning_rate in args.learning_rates
        for batch_size in args.batch_sizes
    ]
    print(f"Sweep ID: {sweep_id}")
    print(f"Number of runs: {len(combinations)}")

    for learning_rate, batch_size in combinations:
        train_and_log(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            learning_rate=learning_rate,
            batch_size=batch_size,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            seed=args.seed,
            sweep_id=sweep_id,
        )

    print_sweep_summary(args.experiment_name, sweep_id)


if __name__ == "__main__":
    main()
