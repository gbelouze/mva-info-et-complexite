"""
Training script using fastText
"""
import tempfile
from pathlib import Path
from challenge.io import dump_input
import pandas as pd  # type: ignore
import fasttext  # type: ignore
from rich import print as rprint
from logging import getLogger

log = getLogger("challenge")


def train(xy: pd.DataFrame):
    with tempfile.NamedTemporaryFile() as input_file:
        dump_input(xy, Path(input_file.name))
        log.info(f"Training on {len(xy)} examples")
        model = fasttext.train_supervised(
            input=input_file.name, lr=0.1, epoch=20, wordNgrams=2)
    log.info("Finished training")
    return model


def test(model, xy: pd.DataFrame):
    with tempfile.NamedTemporaryFile() as input_file:
        dump_input(xy, Path(input_file.name))
        total, precision, recall = model.test(input_file.name)
    rprint(f"Accuracy on {total} tests : {100*precision:.2f}%")
    return total, precision, recall
