import click
import logging
from pathlib import Path
from rich.logging import RichHandler
import challenge.io as io
import challenge.train as Ch

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

log = logging.getLogger("challenge")


@click.group()
@click.option("-v", "--verbose", count=True)
@click.option("--quiet/--no-quiet", default=False)
def main(verbose, quiet):
    if verbose == 0:
        log.setLevel(logging.WARNING)
    elif verbose == 1:
        log.setLevel(logging.INFO)
    elif verbose >= 2:
        log.setLevel(logging.DEBUG)
    if quiet:
        log.setLevel(logging.ERROR)


@main.command()
@click.argument('x', type=click.Path(exists=True, path_type=Path))
@click.argument('y', type=click.Path(exists=True, path_type=Path))
@click.argument('out', type=click.Path(path_type=Path))
@click.option("--overwrite/--no-overwrite", default=False)
def xy(x, y, out, overwrite):
    """Create a submission file.
    X is a collection of reviews. Y is their associated sentiment.
    """
    xy = io.xy(io.loadx(x), io.loady(y))
    io.dumpxy(out, xy, overwrite=overwrite)


@main.command()
@click.argument('train', type=click.Path(exists=True, path_type=Path))
@click.argument('test', type=click.Path(exists=True, path_type=Path))
def eval(train, test):
    """Train fasttext on a submission file TRAIN and evaluate it on TEST"""
    xy_train = io.loadxy(train)
    xy_test = io.loadxy(test)
    model = Ch.train(xy_train)
    Ch.test(model, xy_test)
