"""Urdu text to render into synthetic line images.

A bundled vocabulary is included so the generator works with no network at
all. When a text dataset can be streamed from the Hub it is used as well,
which gives real sentence structure instead of random word salad, but it
is never required.
"""

import random


# Common Urdu vocabulary, grouped loosely so generated lines stay plausible
# rather than being a random pile of unrelated nouns.
NOUNS = [
    "کتاب", "پانی", "دوست",
    "شہر", "زندگی", "وقت",
    "دنیا", "انسان", "علم",
    "گھر", "راستہ", "آسمان",
    "زمین", "دل", "سورج",
    "چاند", "ستارہ", "درخت",
    "پھول", "بارش", "ہوا",
    "رات", "دن", "صبح", "شام",
    "سال", "مہینہ", "ہفتہ",
    "لوگ", "بچہ", "ماں", "باپ",
    "بھائی", "بہن", "استاد",
    "سکول", "کالج", "سڑک",
    "گاڑی", "کھانا", "روٹی",
    "چائے", "دودھ", "پھل",
    "قلم", "کاغذ", "میز",
    "کرسی", "دروازہ",
    "کھڑکی", "دیوار", "کمرہ",
    "حکومت", "ملک", "قوم",
    "تاریخ", "ثقافت", "زبان",
    "شاعری", "کہانی", "خبر",
    "اخبار", "کمپیوٹر",
    "موبائل", "پیسہ", "دکان",
    "بازار", "قیمت", "محنت",
    "آرام", "خواب", "خوشی",
    "امید", "ہمت", "طاقت",
    "صحت", "دوا", "ڈاکٹر",
    "ہسپتال", "سچ", "انصاف",
    "قانون", "عدالت", "امن",
    "دوستی", "رشتہ", "شادی",
    "تقریب", "عید", "مسجد",
    "دعا", "ایمان", "صبر",
]

ADJECTIVES = [
    "اچھا", "بڑا", "چھوٹا",
    "نیا", "پرانا", "خوبصورت",
    "مشکل", "آسان", "تیز",
    "آہستہ", "روشن", "تاریک",
    "ساف", "گہرا", "بلند",
]

VERBS = [
    "ہے", "تھا", "ہوتا", "کرتا",
    "جاتا", "آتا", "دیکھا",
    "کہا", "لکھا", "پڑھا",
    "چاہتا", "سمجھتا",
]

CONNECTORS = [
    "کا", "کی", "کے", "میں", "سے",
    "پر", "اور", "یا", "لیکن",
    "کہ", "جو", "ہر", "اس", "ایک",
]

DIGITS = ["۰", "۱", "۲", "۳", "۴", "۵", "۶", "۷", "۸", "۹"]


def random_number(rng):
    length = rng.randint(1, 4)
    text = ""
    position = 0
    while position < length:
        text = text + rng.choice(DIGITS)
        position = position + 1
    return text


def make_line(rng, min_words=3, max_words=9):
    """Build one plausible looking Urdu line."""
    target = rng.randint(min_words, max_words)

    words = []
    count = 0
    while count < target:
        roll = rng.random()

        if roll < 0.40:
            words.append(rng.choice(NOUNS))
        elif roll < 0.60:
            words.append(rng.choice(CONNECTORS))
        elif roll < 0.75:
            words.append(rng.choice(ADJECTIVES))
        elif roll < 0.92:
            words.append(rng.choice(VERBS))
        else:
            words.append(random_number(rng))

        count = count + 1

    return " ".join(words)


def streamed_sentences(repo_id, text_column, wanted, seed=42):
    """Try to pull real Urdu sentences from the Hub. Returns [] on failure.

    Streaming keeps this to a few hundred kilobytes, and the generator works
    without it, so any error here is logged and ignored.
    """
    collected = []

    try:
        from datasets import load_dataset

        streamed = load_dataset(repo_id, streaming=True)
        split_names = list(streamed.keys())
        stream = streamed[split_names[0]]

        for row in stream:
            if text_column not in row:
                break
            value = row[text_column]
            if not isinstance(value, str):
                continue

            cleaned = " ".join(value.split())
            if len(cleaned) < 8:
                continue

            # keep lines short enough to look like a scanned text line
            words = cleaned.split()
            if len(words) > 12:
                cleaned = " ".join(words[:12])

            collected.append(cleaned)
            if len(collected) >= wanted:
                break
    except Exception as error:
        message = str(error)
        if len(message) > 160:
            message = message[:160] + "..."
        print("  could not stream text corpus (" + repo_id + "): " + message)
        return []

    return collected


def build_corpus(count, seed=42, repo_id=None, text_column="text"):
    """Return `count` Urdu lines, mixing streamed and generated text."""
    rng = random.Random(seed)

    lines = []

    if repo_id is not None and repo_id != "":
        wanted = int(count * 0.5)
        streamed = streamed_sentences(repo_id, text_column, wanted, seed)
        if len(streamed) > 0:
            print("  streamed " + str(len(streamed)) + " real urdu sentences")
            for sentence in streamed:
                lines.append(sentence)

    while len(lines) < count:
        lines.append(make_line(rng))

    rng.shuffle(lines)
    return lines[:count]
