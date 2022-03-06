import abc
from pathlib import Path
import tempfile
import tarfile
import requests  # type: ignore
from logging import getLogger
import pandas as pd  # type: ignore
import challenge.io as io


log = getLogger("challenge")


def download(url: str, path: Path):
    with requests.get(url, stream=True) as r:
        with open(path, "wb+") as out:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    out.write(chunk)
    log.info(f"Downloaded [cyan]{url}[/] to [magenta]{io.repr(path)}[/]", extra={"markup": True})


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
        self.csvfile = io.submission_dir / f"{self.name}.csv"

    def download(self):
        if not self.tarfile.is_file():
            download(self.url, self.tarfile)
        else:
            log.debug(
                f"Using cached download of [cyan]{self.url}[/] at [magenta]{io.repr(self.tarfile)}[/]", extra={"markup": True})

    def make_xy(self, out: Path = io.submission_dir):
        assert not out.is_file()
        self.download()
        with tempfile.TemporaryDirectory() as _tmp:
            tmp = Path(_tmp)
            with tarfile.open(self.tarfile) as tarball:
                tarball.extractall(path=tmp)
            log.debug(f"Decompressed [magenta]{io.repr(self.tarfile)}[/]", extra={"markup": True})

            pos_path = tmp / "rt-polaritydata" / "rt-polarity.pos"
            neg_path = tmp / "rt-polaritydata" / "rt-polarity.neg"

            with open(pos_path, "r", encoding="latin-1") as pos_file:
                pos = pd.DataFrame({"x": [line.strip('\n') for line in pos_file], "y": 1})
            with open(neg_path, "r", encoding="latin-1") as neg_file:
                neg = pd.DataFrame({"x": [line.strip('\n') for line in neg_file], "y": 0})

            xy = pd.concat([pos, neg], ignore_index=True)

        out.mkdir(parents=True, exist_ok=True)
        self.csvfile = out / f"{self.name}.csv"
        io.dumpxy(self.csvfile, xy)

    def clean(self):
        self.tarfile.unlink(missing_ok=True)
        self.csvfile.unlink(missing_ok=True)

    def describe(self):
        return f"{len(self)} Rotten Tomatoes movie reviews annotated with polarity (see http://www.cs.cornell.edu/people/pabo/movie-review-data/ )"

    def __len__(self):
        return 5331 * 2


all_downloaders = {"PangLee2005": PangLee2005()}
