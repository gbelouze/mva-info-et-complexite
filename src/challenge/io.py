import csv
import logging
from pathlib import Path

import pandas as pd  # type: ignore

log = logging.getLogger("challenge")

base_dir = Path(__file__).resolve().parents[2]
data_dir = base_dir / "data"
download_dir = data_dir / "downloads"
submission_dir = data_dir / "submissions"
default_x = data_dir / "raw" / "x_baseline.csv"
default_y = data_dir / "raw" / "y_baseline.csv"
default_test = data_dir / "test" / "sample.csv"


def repr(path: Path):
    """Path name relative to working directory"""
    cwd = Path.cwd()
    if path.is_relative_to(cwd):
        return str(path.relative_to(cwd))
    return path.resolve()


def loadx(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path, sep=",", index_col=0)
    if "review" not in x.columns:
        log.error(
            f'Table does not contain "reviews" [magenta]{repr(path)}[/]',
            extra={"markup": True},
        )
        raise ValueError
    log.info(f"X file loaded from [magenta]{repr(path)}[/]", extra={"markup": True})
    return x


def loady(path: Path) -> pd.DataFrame:
    y = pd.read_csv(path, sep=",", index_col=0)
    if "category" not in y.columns:
        log.error(
            f'Table does not contain "category" [magenta]{repr(path)}[/]',
            extra={"markup": True},
        )
        raise ValueError
    log.info(f"Y file loaded from [magenta]{repr(path)}[/]", extra={"markup": True})
    return y


def xy(x: pd.DataFrame, y: pd.DataFrame):
    xy = pd.DataFrame()
    xy["x"] = x.review
    xy["y"] = y.category
    return xy


def loadxy(path: Path) -> pd.DataFrame:
    xy = pd.read_csv(path, sep=",", index_col=0)
    if "category" not in xy.columns or "review" not in xy.columns:
        log.error(
            f'Table does not contain "review" and "category" [magenta]{repr(path)}[/]',
            extra={"markup": True},
        )
        raise ValueError
    log.info(f"XY file loaded from [magenta]{repr(path)}[/]", extra={"markup": True})
    return xy.rename(columns={"review": "x", "category": "y"})


def ovewrite_guard(func):
    def wrapped(path: Path, *args, **kwargs):
        overwrite = False
        if "overwrite" in kwargs:
            overwrite = kwargs["overwrite"]
            del kwargs["overwrite"]
        if path.exists() and not overwrite:
            log.error(
                f"File already exists [magenta]{repr(path)}[/]", extra={"markup": True}
            )
            raise FileExistsError
        return func(path, *args, **kwargs)

    return wrapped


@ovewrite_guard
def dumpx(path: Path, x: pd.DataFrame):
    x.to_csv(path, sep=",", index=True, columns=["review"])
    log.info(f"Wrote X file to [magenta]{repr(path)}[/]", extra={"markup": True})


@ovewrite_guard
def dumpy(path: Path, y: pd.DataFrame):
    y.to_csv(path, sep=",", index=True, columns=["category"])
    log.info(f"Wrote Y file to [magenta]{repr(path)}[/]", extra={"markup": True})


@ovewrite_guard
def dumpxy(path: Path, xy: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame()
    df["review"] = xy.x
    df["category"] = xy.y
    df.to_csv(path, sep=",", index=True)
    log.info(
        f"Wrote submission file to [magenta]{repr(path)}[/]", extra={"markup": True}
    )


def fasttext_input(xy: pd.DataFrame):
    return "__label__" + xy.y.astype(str) + " " + xy.x


def dump_input(xy: pd.DataFrame, path: Path):
    """Write [df] to [path] in the input format expected by fasttext"""
    fasttext_input(xy).to_csv(
        path,
        index=False,
        header=False,
        quoting=csv.QUOTE_NONE,
        quotechar="",
        escapechar="\\",
    )
