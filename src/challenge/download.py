import abc
import tarfile
import tempfile
from logging import getLogger
from pathlib import Path

import challenge.io as io
import pandas as pd  # type: ignore
import requests  # type: ignore

log = getLogger("challenge")


def download(url: str, path: Path):
    log.debug(
        f"Downloading [blue underline]{url}[/] to [magenta]{io.repr(path)}[/]",
        extra={"markup": True},
    )
    with requests.get(url, stream=True) as r:
        with open(path, "wb+") as out:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    out.write(chunk)
    log.info(
        f"Downloaded [blue underline]{url}[/] to [magenta]{io.repr(path)}[/]",
        extra={"markup": True},
    )


class DatasetDownloader(abc.ABC):
    @abc.abstractmethod
    def make_xy(self, out: Path):
        """Produce dataset in xy format (see challenge.io)"""
        pass

    @abc.abstractmethod
    def clean(self):
        pass

    @abc.abstractmethod
    def describe(self) -> str:
        pass

    @abc.abstractmethod
    def __len__(self):
        pass


class PangLee2005(DatasetDownloader):

    url = "http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz"

    def __init__(self):
        self.name = "pang_lee_2005"
        self.tarfile = io.download_dir / f"{self.name}.tar.gz"
        self.csvfile = io.xy_dir / f"{self.name}.csv"

    def download(self):
        if not self.tarfile.is_file():
            download(self.url, self.tarfile)
        else:
            log.debug(
                f"Using cached download of [blue underline]{self.url}[/] " "at [magenta]{io.repr(self.tarfile)}[/]",
                extra={"markup": True},
            )

    def make_xy(self, out: Path = io.xy_dir):
        assert not out.is_file()
        self.download()
        with tempfile.TemporaryDirectory() as _tmp:
            tmp = Path(_tmp)
            with tarfile.open(self.tarfile) as tarball:
                tarball.extractall(path=tmp)
            log.debug(
                f"Decompressed [magenta]{io.repr(self.tarfile)}[/]",
                extra={"markup": True},
            )

            pos_path = tmp / "rt-polaritydata" / "rt-polarity.pos"
            neg_path = tmp / "rt-polaritydata" / "rt-polarity.neg"

            with open(pos_path, "r", encoding="latin-1") as pos_file:
                pos = pd.DataFrame({"x": [line.strip("\n") for line in pos_file], "y": 1})
            with open(neg_path, "r", encoding="latin-1") as neg_file:
                neg = pd.DataFrame({"x": [line.strip("\n") for line in neg_file], "y": 0})

            xy = pd.concat([pos, neg], ignore_index=True)

        out.mkdir(parents=True, exist_ok=True)
        self.csvfile = out / f"{self.name}.csv"
        io.dumpxy(self.csvfile, xy)

    def clean(self):
        for file in [self.tarfile, self.csvfile]:
            if file.is_file():
                file.unlink()
                log.info(
                    f"[red]removed[/] [magenta]{io.repr(file)}[/]",
                    extra={"markup": True},
                )

    def describe(self):
        return (
            f"[cyan bold]{len(self):_}[/] Rotten Tomatoes movie reviews annotated with polarity "
            "(see http://www.cs.cornell.edu/people/pabo/movie-review-data/ )"
        )

    def __len__(self):
        return 5331 * 2


class Maas2011(DatasetDownloader):

    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

    def __init__(self):
        self.name = f"maas_2011_{self.mode()}"
        self.tarfile = io.download_dir / "maas_2011.tar.gz"
        self.csvfile = io.xy_dir / f"{self.name}.csv"

    @property
    @abc.abstractmethod
    def mode(self):
        ...

    def download(self):
        if not self.tarfile.is_file():
            download(self.url, self.tarfile)
        else:
            log.debug(
                f"Using cached download of [blue underline]{self.url}[/] " "at [magenta]{io.repr(self.tarfile)}[/]",
                extra={"markup": True},
            )

    def make_xy(self, out: Path = io.xy_dir):
        assert not out.is_file(), f"A directory was expected but {out} is a file"
        self.download()
        with tempfile.TemporaryDirectory() as _tmp:
            tmp = Path(_tmp)
            with tarfile.open(self.tarfile) as tarball:
                tarball.extractall(path=tmp)
            log.debug(
                f"Decompressed [magenta]{io.repr(self.tarfile)}[/]",
                extra={"markup": True},
            )

            pos_path = tmp / "aclImdb" / self.mode() / "pos"
            neg_path = tmp / "aclImdb" / self.mode() / "neg"

            x = []
            y = []

            for file in pos_path.iterdir():
                with open(file, "r") as pos:
                    x.append(pos.read())
                y.append(1)

            for file in neg_path.iterdir():
                with open(file, "r") as neg:
                    x.append(neg.read())
                y.append(0)

            xy = pd.DataFrame({"x": x, "y": y})

        out.mkdir(parents=True, exist_ok=True)
        self.csvfile = out / f"{self.name}.csv"
        io.dumpxy(self.csvfile, xy)

    def clean(self):
        for file in [self.tarfile, self.csvfile]:
            if file.is_file():
                file.unlink()
                log.info(
                    f"[red]removed[/] [magenta]{io.repr(file)}[/]",
                    extra={"markup": True},
                )

    def describe(self):
        return (
            f"[cyan bold]{len(self):_}[/] Imdb movie reviews annotated with polarity"
            "(see [blue underline]https://ai.stanford.edu/~amaas/data/sentiment/[/]) "
            "\\[{self.mode()} subset]"
        )

    def __len__(self):
        return 25_000


class Maas2011_train(Maas2011):
    def mode(self):
        return "train"


class Maas2011_test(Maas2011):
    def mode(self):
        return "test"


all_downloaders = {
    "PangLee2005": PangLee2005(),
    "Maas2011_train": Maas2011_train(),
    "Maas2011_test": Maas2011_test(),
}
