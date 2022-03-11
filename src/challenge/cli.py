import logging
from pathlib import Path

import challenge.download as dwn
import challenge.io as io
import challenge.train as tr
import click
from rich import print as rprint
from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

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
    """Create a single submission from potentially many xy files.
    """
    if xys:
        io.submit(out, [io.loadxy(xy) for xy in xys], overwrite=overwrite)


@main.command()
@click.argument("train", type=click.Path(exists=True, path_type=Path))
@click.argument("test", type=click.Path(exists=True, path_type=Path))
def eval(train, test):
    """Train fasttext on a submission file TRAIN and evaluate it on TEST"""
    xy_train = io.loadxy(train)
    xy_test = io.loadxy(test)
    model = tr.train(xy_train)
    tr.test(model, xy_test)


@main.command()
@click.option(
    "--name", type=click.Choice(list(dwn.all_downloaders.keys()), case_sensitive=False)
)
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
@click.option(
    "--name", type=click.Choice(list(dwn.all_downloaders.keys()), case_sensitive=False)
)
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
@click.option(
    "--name", type=click.Choice(list(dwn.all_downloaders.keys()), case_sensitive=False)
)
@click.option("--all", is_flag=True)
def clean(name, all):
    """Clean dataset. See command [info] for more information about the datasets."""
    if all:
        for downloader in dwn.all_downloaders.values():
            downloader.clean()
    else:
        downloader = dwn.all_downloaders[name]
        downloader.clean()
