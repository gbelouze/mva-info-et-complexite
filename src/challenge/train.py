"""
Training script using fastText
"""
import tempfile
from logging import getLogger
from pathlib import Path

import fasttext  # type: ignore
import pandas as pd  # type: ignore
from challenge.io import dump_input
from rich import print as rprint

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
