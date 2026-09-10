"""What the engine needs to know about a writing system.

Every threshold in the lexical layer is a count of characters: how wide a
character n-gram is, how short a word may be before it is noise, how much of a
word must be shared before containment counts as a compound. Each of those is a
property of the script, not the language: five letters are one short English
or German word, two Han characters are one Chinese word. So the numbers live
here, one row per script, and the code that uses them asks for the row rather
than testing for a language.
The other thing a script decides is whether text arrives pre-segmented. Latin,
Cyrillic, Greek, Arabic, Hangul and the Indic scripts put spaces between
words, so a tokeniser sees words. Han and kana do not, and a whitespace
tokeniser sees one token per phrase; `pieces` splits such a run at its
function words, which is what a reader does, and the bigram pass in `rank`
covers the overlap where two nouns still abut.
"""
from __future__ import annotations

import re
from typing import NamedTuple


class Script(NamedTuple):
    name: str
    # Words separated by whitespace in running text.
    spaced: bool
    # Character n-gram width, and the shortest token that gets n-grams at
    # all: below it a token is an abbreviation or a number, where a shared
    # substring is coincidence ("q3"/"q4"). Anchored n-grams mark a token's
    # ends so prefixes and suffixes count; a two-character word has no
    # interior to anchor.
    gram_n: int
    gram_min: int
    anchored: bool
    # The shortest token that is evidence rather than noise.
    min_word: int
    # A query word this long found inside a field word is a compound match,
    # and a field word this long found inside a query word likewise.
    compound_min: int
    compound_field_min: int


# n=5, after McNamee & Mayfield (2004), who found n=4 or 5 recovers most of
# dictionary decompounding across eight European languages. Three letters is
# where "art" stops being inside "smart" by accident.
ALPHABETIC = Script("alphabetic", spaced=True, gram_n=5, gram_min=5, anchored=True,
                    min_word=3, compound_min=5, compound_field_min=4)
# Most Chinese words are two characters and a one-character word is common,
# so the gram is the bigram, a single character is its own gram, and
# containment at two characters (量子 in 量子计算机) is already evidence.
HAN = Script("han", spaced=False, gram_n=2, gram_min=1, anchored=False,
             min_word=2, compound_min=2, compound_field_min=2)

# Han and kana. Hangul is excluded on purpose: Korean is written with spaces.
UNSPACED = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿]")
_RUNS = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿]+"
                   r"|[^぀-ヿ㐀-䶿一-鿿豈-﫿]+")
_KANA = re.compile(r"[぀-ヿ]")


def unspaced(text: str) -> bool:
    return bool(UNSPACED.search(text))


def script_of(tok: str) -> Script:
    return HAN if UNSPACED.search(tok) else ALPHABETIC


# ---------------------------------------------------------------- segmentation
# Function words and question chatter of the unspaced scripts. A run is
# *split* at these, not merely filtered, which is what turns
# "量子计算机的工作原理是什么" into 量子计算机 and 工作原理 with no dictionary.
# Chosen so that none is a common substring of a content word: 和 (共和国),
# 在 (在线), 有 (有限), 中 (中国), 最 (最新) and 了 (了解) are deliberately
# absent, and the compounds they do appear in are listed whole. Simplified
# and traditional spellings both.
_HAN_STOPS = (
    "是不是 有没有 有沒有 可不可以 能不能 什么样 什麼樣 怎么样 怎麼樣 怎么办 "
    "怎麼辦 为什么 為什麼 告诉我 告訴我 是什么 是什麼 有什么 有什麼 有哪些 "
    "什么 什麼 怎么 怎麼 怎样 怎樣 如何 为何 為何 哪些 哪个 哪個 哪里 哪裡 "
    "哪儿 哪兒 哪座 哪种 哪種 多少 几个 幾個 是否 能否 请问 請問 帮我 幫我 "
    "我想 我要 我们 我們 你们 你們 他们 他們 一下 介绍 介紹 解释 解釋 说明 "
    "說明 了解 关于 關於 以及 或者 还是 還是 并且 並且 但是 因为 因為 所以 "
    "如果 可以 应该 應該 需要 这个 這個 那个 那個 这些 這些 那些 很多 一些 "
    "一种 一種 意思 这本 這本 那本 什么时候 什麼時候 哪一年 哪年 何时 何時 "
    "有多长 有多長 有多少 有多高 有多大 有多远 有多遠 有多久 有多重 多长 多長 "
    "多高 多大 多远 多遠 多久 多重 是谁 是誰 谁写了 誰寫了 谁写 誰寫 写了 寫了 "
    "怎么用 怎麼用 "
    "的 是 吗 嗎 呢 吧 啊 么 麼 请 請 我 你 您 他 她 它 这 這 那 谁 誰"
).split()
# Japanese particles. Only kana; a kanji run under a Japanese query is left to
# the Han list, whose members are rare in Japanese and harmless there.
_KANA_STOPS = (
    "について とは として による ため です ます した する この その あの "
    "の は が を に で と も へ か"
).split()
# A chunk of exactly one of these is grammar left over after the split (在 in
# "在 postgres 中", 了 after a verb, 年 after a number), never a subject. Inside
# a longer chunk they stay, because there they are part of a word.
_WEAK_SINGLE = frozenset(
    "在 中 有 了 和 与 與 及 就 都 也 很 用 做 等 各 每 该 該 从 從 到 对 對 比 "
    "把 被 给 給 让 讓 会 會 能 要 想 为 為 以 不 没 沒 上 下 里 裡 内 內 外 之 "
    "于 於 而 且 又 再 还 還 只 才 太 更 已 将 將 由 向 跟 同 个 個 些 位 名 种 "
    "種 块 塊 座 本 条 條 张 張 项 項 次 年 月 日 号 號 总 總 共 约 約 大约 "
    "得 地 着 著 过 過 吧 哦 呀 嘛".split()
)


def _stop_re(words: list[str]) -> re.Pattern:
    return re.compile("|".join(re.escape(w) for w in
                               sorted(words, key=len, reverse=True)))


_HAN_STOP_RE = _stop_re(_HAN_STOPS)
_KANA_STOP_RE = _stop_re(_KANA_STOPS)


def pieces(tok: str) -> list[str]:
    """A token holding unspaced script, as the words a reader would see in it.

    Latin runs inside it ("python教程") come out whole; each unspaced run is
    split at its function words and the remaining chunks are the content
    words. No segmenter: the chunks are longer than a word where two nouns
    abut, and the bigram pass in `rank` covers the partial overlap.
    """
    out: list[str] = []
    after_digit = False
    for run in _RUNS.findall(tok):
        if not UNSPACED.search(run):
            run = run.strip("._'-+#")
            if run:
                out.append(run)
            after_digit = run[-1:].isdigit()
            continue
        # "2023年": the classifier after a number is grammar, not a word.
        if after_digit and run[0] in _WEAK_SINGLE:
            run = run[1:]
        after_digit = False
        stop_re = _KANA_STOP_RE if _KANA.search(run) else _HAN_STOP_RE
        out.extend(c for c in stop_re.split(run)
                   if c and not (len(c) == 1 and c in _WEAK_SINGLE))
    return out


# ------------------------------------------------------------------ sentences
# Sentence-final punctuation across scripts. The Latin set needs a following
# space to tell a full stop from an abbreviation or a decimal point; the
# full-width and other-script marks are unambiguous on their own.
SENT_END_SPACED = r"[.!?]"
SENT_END_BARE = r"[。！？؟।]"
