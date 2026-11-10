"""Urdu text normalisation.

Urdu typing is inconsistent. The same word can be written with the Arabic
yeh or the Urdu yeh, with or without diacritics, with different space
characters. Without normalisation the CER numbers punish the model for
things it did not get wrong.
"""

import re
import unicodedata


# characters that mean the same thing but have different code points
REPLACEMENTS = {
    "ي": "ی",   # arabic yeh  -> farsi yeh
    "ى": "ی",   # alef maksura -> farsi yeh
    "ك": "ک",   # arabic kaf  -> keheh
    "ہ": "ہ",   # heh goal stays
    "ة": "ہ",   # teh marbuta -> heh goal
    "‌": " ",        # zero width non joiner
    "‍": "",         # zero width joiner
    " ": " ",        # non breaking space
}

# short vowels and other marks that almost never appear in printed text
DIACRITICS = re.compile("[ً-ْٰٓ-ٕ]")


def normalise(text):
    if text is None:
        return ""

    text = unicodedata.normalize("NFC", text)

    output = ""
    for char in text:
        if char in REPLACEMENTS:
            output = output + REPLACEMENTS[char]
        else:
            output = output + char

    output = DIACRITICS.sub("", output)
    output = re.sub(r"\s+", " ", output)
    return output.strip()


def is_mostly_urdu(text, threshold=0.5):
    """Cheap sanity check used by the data validation stage."""
    if len(text) == 0:
        return False

    urdu_chars = 0
    counted = 0
    for char in text:
        if char.isspace():
            continue
        counted = counted + 1
        code = ord(char)
        if 0x0600 <= code <= 0x06FF or 0xFB50 <= code <= 0xFDFF:
            urdu_chars = urdu_chars + 1

    if counted == 0:
        return False
    return (float(urdu_chars) / float(counted)) >= threshold
