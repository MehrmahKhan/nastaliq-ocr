"""Replay the held back stream split against the running API.

This is how you get real looking traffic into the review queue and into
Prometheus without waiting for users. Run it after `docker compose up`.

    python scripts/seed_stream.py 60
"""

import sys

import requests

from nastaliqops.config import Paths
from nastaliqops.data.splits_io import read_split


API_URL = "http://localhost:8000/ocr"


def main():
    limit = 50
    if len(sys.argv) > 1:
        limit = int(sys.argv[1])

    records = read_split("stream")

    sent = 0
    flagged = 0
    position = 0

    while position < len(records) and sent < limit:
        record = records[position]
        image_path = Paths.processed / "images" / record["file_name"]
        position = position + 1

        if not image_path.exists():
            continue

        handle = open(image_path, "rb")
        files = {"file": (record["file_name"], handle, "image/png")}

        try:
            response = requests.post(API_URL, files=files, timeout=120)
        except Exception as error:
            print("request failed: " + str(error))
            handle.close()
            break

        handle.close()

        if response.status_code != 200:
            print("status " + str(response.status_code))
            continue

        body = response.json()
        sent = sent + 1
        if body.get("needs_review"):
            flagged = flagged + 1

        print(str(sent) + " route=" + body.get("route", "?") +
              " conf=" + str(round(body.get("confidence", 0.0), 3)))

    print("sent " + str(sent) + " lines, " + str(flagged) + " went to review")


if __name__ == "__main__":
    main()
