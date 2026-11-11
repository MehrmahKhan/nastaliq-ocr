import os
from PIL import Image, ImageDraw, ImageFont, features

URDU_LINES = [
    "\u067e\u0627\u06a9\u0633\u062a\u0627\u0646 \u0645\u06cc\u06ba \u0627\u0631\u062f\u0648 \u0632\u0628\u0627\u0646 \u0628\u0648\u0644\u06cc \u062c\u0627\u062a\u06cc \u06c1\u06d2",
    "\u06cc\u06c1 \u0627\u06cc\u06a9 \u062e\u0648\u0628\u0635\u0648\u0631\u062a \u0627\u0648\u0631 \u067e\u0631\u0627\u0646\u06cc \u0632\u0628\u0627\u0646 \u06c1\u06d2",
    "\u0646\u0633\u062a\u0639\u0644\u06cc\u0642 \u0631\u0633\u0645 \u0627\u0644\u062e\u0637 \u067e\u0691\u06be\u0646\u0627 \u0645\u0634\u06a9\u0644 \u06c1\u06d2",
]

FONT_CANDIDATES = [
    "C:\\Windows\\Fonts\\UrduType.ttf",
    "C:\\Windows\\Fonts\\NotoNastaliqUrdu-Regular.ttf",
    "C:\\Windows\\Fonts\\arial.ttf",
]

font_path = None
index = 0
while index < len(FONT_CANDIDATES):
    candidate = FONT_CANDIDATES[index]
    if os.path.exists(candidate):
        font_path = candidate
        break
    index = index + 1

if font_path is None:
    print("no usable font found")
    raise SystemExit(1)

print("using font:", font_path)

has_raqm = features.check("raqm")
print("raqm shaping available:", has_raqm)

lines_to_draw = []
if has_raqm:
    lines_to_draw = list(URDU_LINES)
else:
    import arabic_reshaper
    from bidi.algorithm import get_display
    position = 0
    while position < len(URDU_LINES):
        reshaped = arabic_reshaper.reshape(URDU_LINES[position])
        lines_to_draw.append(get_display(reshaped))
        position = position + 1

font = ImageFont.truetype(font_path, 44)
image = Image.new("RGB", (1000, 320), "white")
draw = ImageDraw.Draw(image)

y = 40
row = 0
while row < len(lines_to_draw):
    if has_raqm:
        draw.text((950, y), lines_to_draw[row], font=font, fill="black",
                  anchor="ra", direction="rtl", language="ur")
    else:
        draw.text((950, y), lines_to_draw[row], font=font, fill="black", anchor="ra")
    y = y + 90
    row = row + 1

image.save("sample.png")
print("wrote sample.png")
