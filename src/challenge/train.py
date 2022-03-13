"""
Training script using fastText
"""
import random
import tempfile
from logging import getLogger
from pathlib import Path

import fasttext  # type: ignore
import pandas as pd  # type: ignore
from challenge.io import dump_input
from rich import print as rprint
from rich.panel import Panel
from rich.table import Column, Table

log = getLogger("challenge")


def train(xy: pd.DataFrame):
    with tempfile.NamedTemporaryFile() as input_file:
        dump_input(xy, Path(input_file.name))
        log.info(f"Training on {len(xy)} examples")
        model = fasttext.train_supervised(
            input=input_file.name, lr=0.1, epoch=20, wordNgrams=2
        )
    log.info("Finished training")
    return model


def test(model, xy: pd.DataFrame):
    with tempfile.NamedTemporaryFile() as input_file:
        dump_input(xy, Path(input_file.name))
        total, precision, recall = model.test(input_file.name)
    rprint(f"Accuracy on {total} tests : {100*precision:.2f}%")
    return total, precision, recall


def predict(model, xy: pd.DataFrame):
    for i, row in xy.iterrows():
        x = row.x
        (predicted_label,), proba = model.predict(x)
        y_pred = int(predicted_label[-1])
        xy.loc[i, "y_pred"] = y_pred
        xy.loc[i, "confidence"] = proba[0]


def mistakes(model, xy: pd.DataFrame, k: int = 1):
    indices = list(xy.index)
    random.shuffle(indices)
    false_positives = []
    false_negatives = []

    predict(model, xy)
    rprint(f"Number of false positives : {len(xy[(xy.y == 0) & (xy.y_pred == 1)])}")
    rprint(f"Number of false negatives : {len(xy[(xy.y == 1) & (xy.y_pred == 0)])}")
    for i, row in xy.loc[indices].iterrows():
        if row.y == 1 and row.y_pred == 0 and len(false_negatives) < k:
            false_negatives.append((row.x, row.confidence))
        elif row.y == 0 and row.y_pred == 1 and len(false_positives) < k:
            false_positives.append((row.x, row.confidence))

        if len(false_positives) == len(false_negatives) == k:
            break

    table = Table(
        Column(header="Review", justify="center"),
        Column(
            header="Confidence", vertical="middle", justify="center", style="bold blue"
        ),
        Column(header="Kind", vertical="middle", justify="center", style="red"),
        title="Mistakes",
    )

    for kind, reviews in [
        ("False positive", false_positives),
        ("False negative", false_negatives),
    ]:
        for review, confidence in reviews:
            table.add_row(Panel(review), f"{100*confidence[0]:.0f}%", kind)
        if kind == "False positive":
            table.add_row()
    rprint(table)
