import logging
from pathlib import Path

import challenge.denoise as denoise_
import challenge.download as dwn
import challenge.io as io
import challenge.train as tr
import click
import pandas as pd
from rich import print as rprint
from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])

log = logging.getLogger("challenge")


@click.group()
@click.option("-v", "--verbose", is_flag=True)
@click.option("--quiet/--no-quiet", default=False)
def main(verbose, quiet):
    if verbose:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.INFO)
    if quiet:
        log.setLevel(logging.ERROR)


@main.command()
@click.argument("x", type=click.Path(exists=True, path_type=Path))
@click.argument("y", type=click.Path(exists=True, path_type=Path))
@click.argument("out", type=click.Path(path_type=Path))
@click.option("--overwrite/--no-overwrite", default=False)
def xy(x, y, out, overwrite):
    """Merge x and y files.
    X is a collection of reviews. Y is their associated sentiment.
    """
    xy = io.xy(io.loadx(x), io.loady(y))
    io.dumpxy(out, xy, overwrite=overwrite)


@main.command()
@click.argument("xys", type=click.Path(exists=True, path_type=Path), nargs=-1)
@click.argument("out", type=click.Path(path_type=Path), nargs=1)
@click.option("--overwrite/--no-overwrite", default=False)
def submit(xys, out, overwrite):
    """Create a single submission from potentially many xy files."""
    if xys:
        io.submit(out, [io.loadxy(xy) for xy in xys], overwrite=overwrite)


@main.command()
@click.argument("train", type=click.Path(exists=True, path_type=Path))
@click.argument("test", type=click.Path(exists=True, path_type=Path))
@click.option("-m", "--show-mistakes", default=0, help="Show examples of mistakes.")
def eval(train, test, show_mistakes):
    """Train fasttext on a submission file TRAIN and evaluate it on TEST"""
    xy_train = io.loadxy(train)
    xy_test = io.loadxy(test)
    model = tr.train(xy_train)
    tr.test(model, xy_test)
    if show_mistakes:
        tr.mistakes(model, xy_test, k=show_mistakes)


@main.command()
@click.option("--name", type=click.Choice(list(dwn.all_downloaders.keys()), case_sensitive=False))
@click.option("--all", is_flag=True)
def download(name, all):
    """Download dataset. See command [info] for more information about the datasets."""
    if all:
        for downloader in dwn.all_downloaders.values():
            downloader.make_xy()
    else:
        downloader = dwn.all_downloaders[name]
        downloader.make_xy()


@main.command()
@click.option("--name", type=click.Choice(list(dwn.all_downloaders.keys()), case_sensitive=False))
@click.option("--all", is_flag=True)
def info(name, all):
    """Print info about dataset"""
    if all:
        for name, downloader in dwn.all_downloaders.items():
            rprint(f"[bold]{name}[/]: {downloader.describe()}")
    else:
        downloader = dwn.all_downloaders[name]
        rprint(f"[bold]{name}[/]: {downloader.describe()}")


@main.command()
@click.argument("xy", type=click.Path(exists=True, path_type=Path))
@click.argument("out", type=click.Path(path_type=Path))
@click.option("--overwrite/--no-overwrite", default=False)
@click.option(
    "--vocabs",
    type=click.Path(path_type=Path),
    default=[],
    multiple=True,
    help=(
        "Datasets to use for vocabulary denoising. May take a while, but can be interrupted at any point "
        "to save a partially denoised dataset."
    ),
)
def denoise(xy, out, overwrite, vocabs):
    """Clean file [XY] and save it to [OUT]"""
    xy = io.loadxy(xy)
    xy = denoise_.remove_non_ascii(xy)
    xy = denoise_.remove_bad_tokens(xy, io.data_dir / "bad_tokens.json")
    xy = denoise_.remove_non_english(xy)
    if vocabs:
        vocab_xy = pd.concat([io.loadxy(vocab) for vocab in vocabs], ignore_index=True)
        vocab = denoise_.vocab(vocab_xy)
        xy = denoise_.correct_with_vocab(xy, vocab)
    io.dumpxy(out, xy, overwrite=overwrite)


@main.command()
@click.option("--name", type=click.Choice(list(dwn.all_downloaders.keys()), case_sensitive=False))
@click.option("--all", is_flag=True)
def clean(name, all):
    """Remove downloaded files of a dataset. See command [info] for more information about the datasets."""
    if all:
        for downloader in dwn.all_downloaders.values():
            downloader.clean()
    else:
        downloader = dwn.all_downloaders[name]
        downloader.clean()
