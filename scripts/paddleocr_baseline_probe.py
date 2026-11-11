import sys
from paddleocr import PaddleOCR

if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    image_path = "sample.png"

ocr = PaddleOCR(
    lang="ur",
    enable_mkldnn=False,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

print("running on:", image_path)
results = ocr.predict(image_path)

for result in results:
    texts = result["rec_texts"]
    scores = result["rec_scores"]
    print("lines found:", len(texts))
    print("-" * 40)
    index = 0
    while index < len(texts):
        print(round(scores[index], 3), "|", texts[index])
        index = index + 1
    print("-" * 40)
