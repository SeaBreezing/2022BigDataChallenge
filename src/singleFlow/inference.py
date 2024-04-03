import torch
from torch.utils.data import SequentialSampler, DataLoader

from config import parse_args
from data_helper import MultiModalDataset
from category_id_map import lv2id_to_category_id
from model import MultiModal
import numpy as np
import os


def inference():
    args = parse_args()
    # 1. load data
    dataset = MultiModalDataset(
        args, args.test_annotation, args.test_zip_feats, test_mode=True
    )
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=args.test_batch_size,
        sampler=sampler,
        drop_last=False,
        pin_memory=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch,
    )

    # 2. load model
    mmfs = []
    
    # 进行10折模型的融合
    for ckpt_file in os.listdir(args.ckpt_file):
        if ".bin" in ckpt_file:
            model_path = args.ckpt_file + ckpt_file
            model = MultiModal(args)
            checkpoint = torch.load(model_path, map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])
            model = torch.nn.parallel.DataParallel(model.cuda())
            mmfs.append(model)
            model_judge = model
    for mmf in range(len(mmfs)):
        mmfs[mmf].eval()  # 关闭梯度，将模型调整为测试模式

    # 3. inference
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            pred = []
            mix_pred = 0
            for mmf in range(len(mmfs)):
                pred_label_id = mmfs[mmf](batch, inference=True)
                pred.append(pred_label_id)

            for i in range(len(pred)):
                mix_pred += pred[i] * 0.1  # 设置模型的平均
            predictions.extend(mix_pred.cpu().numpy())
            np.save("../output/prediction_singleFlow", predictions)


if __name__ == "__main__":
    inference()
