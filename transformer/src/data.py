# pyright: reportMissingImports=false
"""
Data loading utilities for local IWSLT2017 dataset.
"""
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional

from torch.utils.data import Dataset, DataLoader


class Seq2SeqDataset(Dataset):
    """Simple sequence-to-sequence dataset backed by parallel text lists."""

    def __init__(self, src_texts, tgt_texts, src_tokenizer, tgt_tokenizer, max_len=512):
        if len(src_texts) != len(tgt_texts):
            raise ValueError("Source and target lists must be the same length.")
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]

        src_tokens = self.src_tokenizer.encode(
            src_text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).squeeze(0)

        tgt_tokens = self.tgt_tokenizer.encode(
            tgt_text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).squeeze(0)

        tgt_input = tgt_tokens[:-1]
        tgt_output = tgt_tokens[1:]

        return src_tokens, tgt_input, tgt_output


def _read_tag_file(path: str) -> List[str]:
    """Read train.tags parallel file and drop markup lines."""
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("<"):
                continue
            lines.append(line)
    return lines


def _read_parallel_tag_files(src_path: str, tgt_path: str) -> Tuple[List[str], List[str]]:
    src_texts = _read_tag_file(src_path)
    tgt_texts = _read_tag_file(tgt_path)
    if len(src_texts) != len(tgt_texts):
        raise ValueError(
            f"Parallel files have mismatched lengths: {src_path} ({len(src_texts)}) vs {tgt_path} ({len(tgt_texts)})"
        )
    return src_texts, tgt_texts


def _read_xml_segments(path: str) -> List[str]:
    """Parse XML split files and return ordered segment texts."""
    tree = ET.parse(path)
    root = tree.getroot()
    segments = []
    for seg in root.iter("seg"):
        text = (seg.text or "").strip()
        segments.append(text)
    return segments


def _read_parallel_xml_files(src_path: str, tgt_path: str) -> Tuple[List[str], List[str]]:
    src_texts = _read_xml_segments(src_path)
    tgt_texts = _read_xml_segments(tgt_path)
    if len(src_texts) != len(tgt_texts):
        raise ValueError(
            f"Parallel XML files have mismatched lengths: {src_path} ({len(src_texts)}) vs {tgt_path} ({len(tgt_texts)})"
        )
    return src_texts, tgt_texts


def _resolve_data_dir(data_dir: Optional[str]) -> str:
    if data_dir is not None:
        target_dir = data_dir
    else:
        target_dir = os.path.join(os.getcwd(), "IWSLT2017")

    if not os.path.isdir(target_dir):
        raise FileNotFoundError(
            f"IWSLT2017 data directory not found at '{target_dir}'. Please place the dataset inside the project."
        )
    return target_dir


def load_iwslt2017_local(src_tokenizer, tgt_tokenizer, max_len=512, data_dir=None):
    """Load IWSLT2017 (EN→DE) splits from local files."""
    dataset_dir = _resolve_data_dir(data_dir)

    train_src_path = os.path.join(dataset_dir, "train.tags.en-de.en")
    train_tgt_path = os.path.join(dataset_dir, "train.tags.en-de.de")
    dev_src_path = os.path.join(dataset_dir, "IWSLT17.TED.dev2010.en-de.en.xml")
    dev_tgt_path = os.path.join(dataset_dir, "IWSLT17.TED.dev2010.en-de.de.xml")
    test_src_path = os.path.join(dataset_dir, "IWSLT17.TED.tst2010.en-de.en.xml")
    test_tgt_path = os.path.join(dataset_dir, "IWSLT17.TED.tst2010.en-de.de.xml")

    for path in [
        train_src_path,
        train_tgt_path,
        dev_src_path,
        dev_tgt_path,
        test_src_path,
        test_tgt_path,
    ]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    train_src, train_tgt = _read_parallel_tag_files(train_src_path, train_tgt_path)
    val_src, val_tgt = _read_parallel_xml_files(dev_src_path, dev_tgt_path)
    test_src, test_tgt = _read_parallel_xml_files(test_src_path, test_tgt_path)

    train_dataset = Seq2SeqDataset(train_src, train_tgt, src_tokenizer, tgt_tokenizer, max_len)
    val_dataset = Seq2SeqDataset(val_src, val_tgt, src_tokenizer, tgt_tokenizer, max_len)
    test_dataset = Seq2SeqDataset(test_src, test_tgt, src_tokenizer, tgt_tokenizer, max_len)

    return train_dataset, val_dataset, test_dataset


def get_data_loaders(tokenizer, batch_size=32, max_len=512, num_workers=0, data_dir=None):
    """
    Create DataLoader objects for locally stored IWSLT2017 data.
    """
    train_dataset, val_dataset, test_dataset = load_iwslt2017_local(
        tokenizer, tokenizer, max_len=max_len, data_dir=data_dir
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader

