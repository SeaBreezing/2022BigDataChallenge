import os, sys
sys.path.append("..")
from config.data_cfg import *
from config.model_cfg import *

import torch
from torch.utils.data import SequentialSampler, DataLoader

from ds_config import parse_args
from data_helper import MultiModalDataset
from category_id_map import lv2id_to_category_id
from model import MultiModal
from qqmodel.qq_uni_model import QQUniModel
BERT_PATH = os.path.join('../', BERT_PATH)
import numpy as np


def inference():
    args = parse_args()
    # 1. load data
    dataset = MultiModalDataset(args, args.test_annotation, args.test_zip_feats, test_mode=True)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=args.test_batch_size,
                            sampler=sampler,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=args.num_workers,
                            prefetch_factor=args.prefetch)

    # 2. load model
    pretrain_model = QQUniModel(MODEL_CONFIG, bert_cfg_dict=BERT_CFG_DICT, model_path=BERT_PATH)
    pretrain_model.load_state_dict(torch.load(args.pretrain_path), strict=False)
    
    prediction_pretrain = np.float32(np.full((len(dataset.anns), 200), 0.))
    for i in range(10):
        model = MultiModal(args, pretrain_model)
        checkpoint = torch.load(f'{args.ckpt_file}{i}.bin', map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        if torch.cuda.is_available():
            model = torch.nn.parallel.DataParallel(model.cuda())
        model.eval()
        # 3. inference
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                logits = model(batch, inference=True, coef=True)
                result=logits.cpu().numpy()
                predictions.extend(result)
            print(f'fold_{i}_end')
        prediction_pretrain += predictions
        
    prediction_pretrain /= 10
    np.save('../../output/prediction_pretrain', prediction_pretrain) 

if __name__ == '__main__':
    inference()
