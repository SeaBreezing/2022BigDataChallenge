import json
import random
import zipfile
from io import BytesIO

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer

from category_id_map import category_id_to_lv2id
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold


def create_dataloaders(args, k_fold: int):
    dataset = MultiModalDataset(args, args.train_annotation, args.train_zip_feats)
    # k-fold to split dataset
    skf = StratifiedKFold(n_splits=10, random_state=args.seed, shuffle=True)
    labels = []
    for i in range(len(dataset)):
        label = category_id_to_lv2id(dataset.anns[i]["category_id"])
        labels.append(label)
    train_dataset = 0
    val_dataset = 0
    for fold_, (train_, val_) in enumerate(skf.split(dataset.anns, labels)):
        trian_medium = torch.utils.data.Subset(dataset, train_)  
        val_medium = torch.utils.data.Subset(dataset, val_)
        if fold_ == k_fold:
            train_dataset = trian_medium
            val_dataset = val_medium
            break

    train_sampler = RandomSampler(train_dataset) 
    val_sampler = SequentialSampler(val_dataset)  
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        drop_last=True,
        pin_memory=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        sampler=val_sampler,
        drop_last=False,
        pin_memory=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch,
    )
    return train_dataloader, val_dataloader


class MultiModalDataset(Dataset):
    """ A simple class that supports multi-modal inputs.

    For the visual features, this dataset class will read the pre-extracted
    features from the .npy files. For the title information, it
    uses the BERT tokenizer to tokenize. We simply ignore the ASR & OCR text in this implementation.

    Args:
        ann_path (str): annotation file path, with the '.json' suffix.
        zip_feats (str): visual feature zip file path.
        test_mode (bool): if it's for testing.
    """

    def __init__(self, args, ann_path: str, zip_feats: str, test_mode: bool = False):
        self.max_frame = args.max_frames
        self.bert_seq_length = args.bert_seq_length
        self.test_mode = test_mode

        # lazy initialization for zip_handler to avoid multiprocessing-reading error
        self.zip_feat_path = zip_feats
        self.handles = [None for _ in range(args.num_workers)]

        # load annotations
        with open(ann_path, "r", encoding="utf8") as f:
            self.anns = json.load(f)

        # initialize the text tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            args.bert_dir, use_fast=True, cache_dir=args.bert_cache
        )

    def __len__(self) -> int:
        return len(self.anns)

    def get_visual_feats(self, worker_id, idx: int) -> tuple:
        # read data from zipfile
        vid = self.anns[idx]["id"]
        if self.handles[worker_id] is None:
            self.handles[worker_id] = zipfile.ZipFile(self.zip_feat_path, "r")
        raw_feats = np.load(
            BytesIO(self.handles[worker_id].read(name=f"{vid}.npy")), allow_pickle=True
        )
        raw_feats = raw_feats.astype(np.float32)  # float16 to float32
        num_frames, feat_dim = raw_feats.shape

        feat = np.zeros((self.max_frame, feat_dim), dtype=np.float32)
        mask = np.ones((self.max_frame,), dtype=np.int32)
        if num_frames <= self.max_frame:
            feat[:num_frames] = raw_feats
            mask[num_frames:] = 0
        else:
            # if the number of frames exceeds the limitation, we need to sample
            # the frames.
            if self.test_mode:
                # uniformly sample when test mode is True
                step = num_frames // self.max_frame
                select_inds = list(range(0, num_frames, step))
                select_inds = select_inds[: self.max_frame]
            else:
                # randomly sample when test mode is False
                select_inds = list(range(num_frames))
                random.shuffle(select_inds)
                select_inds = select_inds[: self.max_frame]
                select_inds = sorted(select_inds)
            for i, j in enumerate(select_inds):
                feat[i] = raw_feats[j]
        feat = torch.FloatTensor(feat)
        mask = torch.LongTensor(mask)
        return feat, mask

    def tokenize_text(self, text: str) -> tuple:
        encoded_inputs = self.tokenizer(
            text, max_length=self.bert_seq_length, padding="max_length", truncation=True
        )

        input_ids = torch.LongTensor(encoded_inputs["input_ids"])
        mask = torch.LongTensor(encoded_inputs["attention_mask"])

        return input_ids, mask

    def __getitem__(self, idx: int) -> dict:
        # Step 1, load visual features from zipfile.
        worker_info = torch.utils.data.get_worker_info()
        frame_input, frame_mask = self.get_visual_feats(worker_info.id, idx)

        # Step 2, load title tokens
        text = self.anns[idx]["title"] + self.anns[idx]["asr"]
        for ocr in self.anns[idx]["ocr"]:
            text += ocr["text"]

        title_input, title_mask = self.tokenize_text(text)

        # Step 3, summarize into a dictionary
        data = dict(
            frame_input=frame_input,
            frame_mask=frame_mask,
            title_input=title_input,
            title_mask=title_mask,
        )

        # Step 4, load label if not test mode
        if not self.test_mode:
            label = category_id_to_lv2id(self.anns[idx]["category_id"])
            data["label"] = torch.LongTensor([label])

        return data
