import csv
import logging
import random
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd  # type: ignore
from rich.progress import Progress

log = logging.getLogger("challenge")

base_dir = Path(__file__).resolve().parents[2]
data_dir = base_dir / "data"
download_dir = data_dir / "downloads"
xy_dir = data_dir / "xy"
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
            log.error(f"File already exists [magenta]{repr(path)}[/]", extra={"markup": True})
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
def dumpxy(path: Path, xy: pd.DataFrame, file_kind="xy"):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame()
    df["review"] = xy.x
    df["category"] = xy.y
    df.to_csv(path, sep=",", index=True)
    log.info(f"Wrote {file_kind} file to [magenta]{repr(path)}[/]", extra={"markup": True})


def submit(path: Path, xys: List[pd.DataFrame], *args, merge=False, **kwargs):
    xy = pd.concat(xys, ignore_index=True)
    if len(xy) > 20_000:
        if merge:
            with Progress() as progress:
                merge_task = progress.add_task("Merging tables...", total=20_000)
                merge_count = 0

                positives = list(xy.index[xy.y == 1])
                negatives = list(xy.index[xy.y == 0])
                random.shuffle(positives)
                random.shuffle(negatives)
                new_xy = pd.DataFrame(columns=["x", "y"])
                for offset, y_value, indices in [(0, 1, positives), (10_000, 0, negatives)]:
                    for i in range(10_000):
                        merge_count += len(range(i, len(indices), 10_000))
                        s = " ".join(xy.x.loc[index] for index in indices[i : len(indices) : 10_000])
                        if s:
                            new_xy.loc[offset + i] = {"x": s, "y": y_value}
                        progress.update(merge_task, advance=1)
                xy = new_xy
                log.debug(f"Merged [cyan bold]{merge_count:_}[/] entries", extra={"markup": True})
        else:
            xy = xy.iloc[np.random.choice(np.arange(len(xy)), size=20_000, replace=False)]
            log.debug("Trimed data to length [cyan bold]20_000[/]", extra={"markup": True})

    dumpxy(path, xy, *args, file_kind="submission", **kwargs)


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
