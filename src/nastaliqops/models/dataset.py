"""Torch dataset that reads a split file and hands tensors to TrOCR."""

import cv2
import torch

from torch.utils.data import Dataset

from nastaliqops.config import Paths
from nastaliqops.data.splits_io import read_split


# re-exported so existing imports of read_split from this module keep working
__all__ = ["LineDataset", "read_split"]


class LineDataset(Dataset):
    def __init__(self, split_name, processor, max_target_length):
        self.records = read_split(split_name)
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        image_path = str(Paths.processed / "images" / record["file_name"])
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        encoded = self.processor(images=rgb, return_tensors="pt")
        pixel_values = encoded.pixel_values.squeeze(0)

        tokens = self.processor.tokenizer(
            record["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_target_length,
        )

        labels = []
        pad_id = self.processor.tokenizer.pad_token_id
        for token_id in tokens.input_ids:
            if token_id == pad_id:
                labels.append(-100)
            else:
                labels.append(token_id)

        item = {}
        item["pixel_values"] = pixel_values
        item["labels"] = torch.tensor(labels)
        return item
