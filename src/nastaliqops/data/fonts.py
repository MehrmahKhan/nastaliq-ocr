"""Find a Nastaliq font to render synthetic lines with.

Nastaliq matters here. Urdu written in a Naskh face looks like Arabic and
is a visibly easier recognition problem, so training on Naskh and claiming
to have built a Nastaliq OCR system would be dishonest. The resolver looks
for a real Nastaliq face first and only falls back to Naskh with a loud
warning.

Order of preference:
  1. any ttf or otf the user dropped into data/fonts
  2. a Nastaliq face already installed on the machine
  3. Noto Nastaliq Urdu, downloaded once into data/fonts
"""

import os
import sys

from pathlib import Path

from nastaliqops.config import Paths


FONT_DIR = Paths.root / "data" / "fonts"

# filenames of Nastaliq faces that ship with Windows or are commonly
# installed in Pakistan
NASTALIQ_HINTS = [
    "nastaliq",
    "nastaleeq",
    "aldhabi",
    "urdutypesetting",
    "urdtype",
    "jameel",
]

# naskh and other arabic script faces, usable but not ideal
NASKH_HINTS = [
    "notonaskh",
    "naskh",
    "arabtype",
    "trado",
    "amiri",
]

SYSTEM_FONT_DIRS = [
    "C:/Windows/Fonts",
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    str(Path.home() / ".fonts"),
    "/Library/Fonts",
    str(Path.home() / "Library" / "Fonts"),
]

DOWNLOAD_URLS = [
    "https://github.com/notofonts/nastaliq/raw/main/fonts/NotoNastaliqUrdu/hinted/ttf/NotoNastaliqUrdu-Regular.ttf",
    "https://github.com/google/fonts/raw/main/ofl/notonastaliqurdu/NotoNastaliqUrdu%5Bwght%5D.ttf",
    "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoNastaliqUrdu/NotoNastaliqUrdu-Regular.ttf",
]


# If a font lacks these, it cannot render Urdu and every generated image
# would be a row of empty boxes. That failure is silent at render time and
# only shows up much later as an impossible error rate, so it is checked
# up front and the font is rejected.
REQUIRED_CODEPOINTS = [
    0x0627,  # alef
    0x0628,  # beh
    0x0645,  # meem
    0x06CC,  # farsi yeh, the urdu yeh
    0x06A9,  # keheh, the urdu kaf
    0x06C1,  # heh goal
]


def covers_urdu(font_path):
    """True if the font has glyphs for the Urdu letters we actually use."""
    try:
        from fontTools.ttLib import TTFont
    except ImportError:
        # cannot check, assume the caller knows what it is doing
        return True

    try:
        font = TTFont(str(font_path), fontNumber=0, lazy=True)
        covered = set()
        for table in font["cmap"].tables:
            for codepoint in table.cmap:
                covered.add(codepoint)
        font.close()
    except Exception:
        return False

    for codepoint in REQUIRED_CODEPOINTS:
        if codepoint not in covered:
            return False
    return True


def filter_usable(font_paths):
    usable = []
    for font_path in font_paths:
        if covers_urdu(font_path):
            usable.append(font_path)
        else:
            print("  skipping " + font_path.name + " (no urdu glyphs)")
    return usable


def _matches_any(name, hints):
    lowered = name.lower()
    for hint in hints:
        if hint in lowered:
            return True
    return False


def _fonts_in(directory):
    found = []
    path = Path(directory)
    if not path.exists():
        return found

    try:
        entries = os.listdir(path)
    except Exception:
        return found

    for entry in entries:
        lowered = entry.lower()
        if lowered.endswith(".ttf") or lowered.endswith(".otf"):
            found.append(path / entry)
    return found


def _user_supplied():
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    return _fonts_in(FONT_DIR)


def _system_nastaliq():
    nastaliq = []
    naskh = []

    for directory in SYSTEM_FONT_DIRS:
        for font_path in _fonts_in(directory):
            if _matches_any(font_path.name, NASTALIQ_HINTS):
                nastaliq.append(font_path)
            elif _matches_any(font_path.name, NASKH_HINTS):
                naskh.append(font_path)

    return nastaliq, naskh


def _download():
    try:
        import requests
    except ImportError:
        return None

    FONT_DIR.mkdir(parents=True, exist_ok=True)
    target = FONT_DIR / "NotoNastaliqUrdu-Regular.ttf"

    for url in DOWNLOAD_URLS:
        try:
            print("  trying " + url)
            response = requests.get(url, timeout=60)
            if response.status_code != 200:
                continue
            if len(response.content) < 20000:
                continue

            handle = open(target, "wb")
            handle.write(response.content)
            handle.close()
            print("  saved " + str(target))
            return target
        except Exception as error:
            print("  failed: " + str(error))

    return None


def resolve_fonts():
    """Return a list of usable font paths, best first.

    Raises if nothing at all can be found, with instructions, because a
    silent fallback to a Latin font would produce boxes instead of Urdu
    and the failure would only show up much later as a terrible CER.
    """
    supplied = filter_usable(_user_supplied())
    if len(supplied) > 0:
        print("using font(s) from data/fonts:")
        for font_path in supplied:
            print("  " + font_path.name)
        return supplied

    nastaliq, naskh = _system_nastaliq()
    nastaliq = filter_usable(nastaliq)
    naskh = filter_usable(naskh)

    if len(nastaliq) > 0:
        print("found installed nastaliq font(s):")
        for font_path in nastaliq:
            print("  " + str(font_path))
        return nastaliq

    print("no nastaliq font installed, downloading Noto Nastaliq Urdu")
    downloaded = _download()
    if downloaded is not None:
        if covers_urdu(downloaded):
            return [downloaded]
        print("  the downloaded file has no urdu glyphs, ignoring it")

    if len(naskh) > 0:
        print("")
        print("WARNING: falling back to a naskh font, not nastaliq.")
        print("Urdu in naskh is an easier problem and does not match the")
        print("project's stated goal. Install a nastaliq font when you can.")
        print("")
        return naskh

    print("")
    print("no usable arabic script font was found.")
    print("fix it in one of these ways:")
    print("  - download Noto Nastaliq Urdu from fonts.google.com and put the")
    print("    .ttf into data/fonts/")
    print("  - on windows, install the Urdu language pack, which adds the")
    print("    Aldhabi and Urdu Typesetting nastaliq faces")
    sys.exit(1)
