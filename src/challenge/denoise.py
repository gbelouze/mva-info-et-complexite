import json
import re
from logging import getLogger
from pathlib import Path

import langid
import pandas as pd
from Levenshtein import distance
from rich import print as rprint
from tqdm import tqdm

log = getLogger("challenge")


class Vocab:
    def __init__(self, radius=5):
        self.children_keys: set[str] = set()
        self.vocab: dict[str, int] = {}
        self.children: dict[str, dict[str, int]] = {}
        self.radius = radius

    def __contains__(self, word):
        return word in self.children_keys or word in self.vocab

    def __getitem__(self, word):
        if word in self:
            count = self.vocab.get(word, 0)
            for name, child in self.children.items():
                if distance(word, name) <= self.radius:
                    count += child[word]
            return count
        raise KeyError

    def add(self, word, count=1):
        if word in self.children_keys:
            for name, child in self.children.items():
                if distance(name, word) <= self.radius:
                    # relies on python >= 3.7 dictionnaries respecting insertion order
                    child.add(word, count)
                    break
        self.vocab[word] = self.vocab.get(word, 0) + count
        self.balance()

    def balance(self, force=False):
        if self.radius > 1 and (force or len(self.vocab) > 50_000):
            for word, count in self.vocab.items():
                for key, child in self.children.items():
                    if distance(key, word) <= self.radius:
                        child.add(word, count)
                        break
                else:
                    new_vocab = Vocab(radius=self.radius - 1)
                    new_vocab.add(word, count)
                    self.children[word] = new_vocab
            self.children_keys.update(self.vocab.keys())
            self.vocab = {}

    def find_closest(self, word, threshold=2):
        """Find the entry [key, count] if it exists where
        [key] is at a maximal [threshold] distance to [word]
        [count] is maximal for such key

        Returns:
            key: str or None if such key does not exist
            count: int or undefined if such key does not exist
            distance: int or undefined if such key does not exist
        """
        best_word, best_distance, best_count = None, threshold, 0
        for key, count in self.vocab.items():
            d = distance(key, word)
            if d < best_distance or d == best_distance and count > best_count:
                best_word = key
                best_distance = d
                best_count = count
        for name, child in self.children.items():
            if distance(name, word) - self.radius <= best_distance:
                key, count, d = child.find_closest(word, threshold=best_distance)
                if key is not None and (
                    d < best_distance or d == best_distance and count > best_count
                ):
                    best_word = key
                    best_distance = d
                    best_count = count
        return best_word, best_count, best_distance


def vocab(xy: pd.DataFrame) -> Vocab:
    vocab = Vocab()
    for index, sentence in tqdm(xy.x.iteritems()):
        for word in re.split(r"[a-zA-Z]+", sentence):
            word = word.lower()
            vocab.add(word)
    vocab.balance(force=True)
    log.info("Loaded vocab")
    return vocab


def replace_token(token: str, vocab: Vocab):
    token = token.lower()
    if token in vocab:
        return token
    best_token, _, _ = vocab.find_closest(token)
    if best_token is None:
        return token
    return best_token


def tokenizer_pattern():
    repeated_punctuation = r"[.?!]+"
    punctuations = rf"(?:[,:;]|{repeated_punctuation})"
    blanks = r"(?:[ \t]|^|$)"
    left_parenthesis = r"[[{(-]"
    right_parenthesis = r"[\]})-]"
    mid_word = r"[/\-]"
    return "|".join(
        [
            f"(?:{pattern})"
            for pattern in (
                f"{punctuations}{blanks}",
                f"{right_parenthesis}{blanks}",
                f"{right_parenthesis}{punctuations}",
                f"{blanks}{left_parenthesis}",
                f"""{blanks}"|"{blanks}|'""",
                mid_word,
                blanks,
            )
        ]
    )


def full_sub(pattern, string, repl):
    """[pattern] must not contain any captured group"""
    tokens = re.split(f"({pattern})", string)
    rprint(tokens)
    for i, token in enumerate(tokens):
        tokens[i] = repl(token)
    return "".join(tokens)


def correct_sentence_with_vocab(s: str, vocab: Vocab):

    contains_alpha = re.compile(r"[a-zA-Z]")

    def repl(token: str):
        if len(token) >= 2 and contains_alpha.match(token):
            return replace_token(token, vocab)
        return token

    return full_sub(tokenizer_pattern(), s, repl)


def correct_with_vocab(xy: pd.DataFrame, vocab: Vocab) -> pd.DataFrame:
    xs = []
    try:
        for index, s in tqdm(xy.x.iteritems()):
            xs.append(correct_sentence_with_vocab(s, vocab))
    except KeyboardInterrupt:
        pass
    return pd.DataFrame(
        {"x": xs, "y": xy.y.iloc[xy.index[range(len(xs))]]},
        index=xy.index[range(len(xs))],
    )


def remove_non_ascii(baseline: pd.DataFrame) -> pd.DataFrame:
    non_ascii = r"[^\x00-\x7F]"
    is_ascii = baseline.apply(lambda row: not bool(re.match(non_ascii, row.x)), axis=1)
    baseline_ascii = baseline[is_ascii]

    # special non ascii case that we want to keep
    special_case = pd.DataFrame(
        {
            "x": [
                "I think this is one of the best films of all times and everybody must realize this movie. "
                "I'm a Turkish boy and a big cinema fan. And in this days our cinema industry is highing up. "
                "And UZAK is the best Turkish film of last ten years. And maybe one of the best films of all times. "
                "Director Nuri Bilge Ceylan is quite amazing, telling story, characters, atmosphere is wonderful. "
                "He is a minimalist direcor and tells about routine event family, dreams, expectations, life. "
                "Tells about you, tells about me, tells about us. I promise you will find a piece of your body "
                "in this move. Cinema life welcomes a new director. He is waiting to realize. "
                "I promise you you will love this movie please watch it"
            ],
            "y": [1],
        }
    )
    baseline_ascii = pd.concat([baseline_ascii, special_case], ignore_index=True)
    log.info(f"Removed {len(baseline) - len(baseline_ascii)} non-ascii entries")

    return baseline_ascii


def remove_bad_tokens(xy: pd.DataFrame, bad_tokens_file: Path) -> pd.DataFrame:
    xs = []
    with open(bad_tokens_file, "r") as f:
        bad_tokens = json.load(f)
    counts = {name: 0 for name in bad_tokens}
    for index, s in xy.x.iteritems():
        for from_token, to_token in bad_tokens.items():
            counts[from_token] += s.count(from_token)
            s = s.replace(from_token, to_token)
        xs.append(s)
    for from_token, count in counts.items():
        to_token = bad_tokens[from_token]
        if count > 0:
            if to_token:
                log.info(
                    f'Replaced {count} times [yellow]"{from_token}"[/] with [yellow]"{to_token}"[/]',
                    extra={"markup": True},
                )
            else:
                log.info(
                    f'Removed {count} occurrences of [yellow]"{from_token}"',
                    extra={"markup": True},
                )
    return pd.DataFrame({"x": xs, "y": xy.y}, index=xy.index)


def remove_non_english(xy: pd.DataFrame) -> pd.DataFrame:
    keep_indices = []
    for i, sentence in xy.x.iteritems():
        language, _score = langid.classify(sentence)
        if language == "en":
            keep_indices.append(i)
    count = len(xy) - len(keep_indices)
    if count:
        log.info(f"Removed {count} non-english entries")
    return xy.iloc[keep_indices]
