#%%writefile pretrain.py
import os, math, random, time, sys, gc,  sys, json, psutil
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
from importlib import reload
reload(logging)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(f"train_{time.strftime('%m%d_%H%M', time.localtime())}.log"),
        logging.StreamHandler()
    ]
)

import numpy as np
import pandas as pd

from config.data_cfg import *
from config.model_cfg import *
from config.pretrain_cfg import *
from data.qq_dataset import QQDataset, create_dataloaders
from qqmodel.qq_uni_model import QQUniModel
from optim.create_optimizer import create_optimizer
from utils.utils import set_random_seed

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, ChainDataset
from transformers import AutoConfig
from transformers import get_cosine_schedule_with_warmup
from config.config import parse_args
#from apex import amp

gc.enable()
    
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
               
set_random_seed(SEED)

def get_pred_and_loss(model, item, task=None):
    """Get pred and loss for specific task"""
    video_feature = item['frame_features'].to(DEVICE)
    input_ids = item['id'].to(DEVICE)
    attention_mask = item['mask'].to(DEVICE)
    video_mask = item['frame_mask'].to(DEVICE)
    
    target = None
    if 'target' in item:
        target = item['target'].to(DEVICE)
    
    pred, emb, loss = model(video_feature, video_mask, input_ids, attention_mask, target, task)
    return pred, emb, loss

def eval(model, data_loader, get_pred_and_loss, compute_loss=True, eval_max_num=99999):
    model.eval()
    loss_l, emb_l = [], []
    with torch.no_grad():
        for batch_num, item in enumerate(data_loader):
            pred, emb, loss = get_pred_and_loss(model, item)
            if loss is not None:
                loss_l.append(loss)
                
            emb_l += emb.to("cpu").tolist()
            
    return torch.mean(torch.stack(loss_l)), np.array(emb_l)

def train(model, model_path, 
          train_loader, val_loader, 
          optimizer, get_pred_and_loss, scheduler=None, 
          num_epochs=5):
    best_val_loss, best_epoch, step = None, 0, 0
    start = time.time()

    for epoch in range(num_epochs):
        for batch_num, item in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            pred, emb, loss = get_pred_and_loss(model, item)
            #with amp.scale_loss(loss, optimizer) as scaled_loss:
                #scaled_loss.sum().backward()
            loss.sum().backward()

            optimizer.step()
            if scheduler:
                scheduler.step()

            if step == 20 or (step % 500 == 0 and step > 0):
                elapsed_seconds = time.time() - start# Evaluate the model on val_loader.

                val_loss, emb = eval(model, val_loader, get_pred_and_loss=get_pred_and_loss, eval_max_num=10000)

                improve_str = ''
                if not best_val_loss or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), model_path)
                    improve_str = f"|New best_val_loss={best_val_loss:6.4}"

                logging.info(f"Epoch={epoch + 1}/{num_epochs}|step={step:3}|val_loss={val_loss:6.4}|time={elapsed_seconds:0.3}s" + improve_str)

                start = time.time()
            step += 1
        model.load_state_dict(torch.load(model_path)) #Load best model
        val_loss, emb = eval(model, val_loader, get_pred_and_loss=get_pred_and_loss, eval_max_num=99999)
        logging.info(f"val_loss={val_loss}")

    return best_val_loss

# Show config
logging.info("Start")
for fname in ['pretrain', 'model', 'data']:
    logging.info('=' * 66)
    with open(f'config/{fname}_cfg.py') as f:
        logging.info(f"Config - {fname}:" + '\n' + f.read().strip())
    
list_val_loss = []
logging.info(f"Model_type = {MODEL_TYPE}")

for fold in range(NUM_FOLDS):
    logging.info('=' * 66)
    model_path = f"model_pretrain_{fold + 1}.pth"
    logging.info(f"Fold={fold + 1}/{NUM_FOLDS} seed={SEED+fold}")
    
    set_random_seed(SEED + fold)

    # load data into memory, need about 60-70g memory
    logging.info("Load data into memory")
    m0 = psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30
    args = parse_args()
    train_loader, val_loader = create_dataloaders(args)
    delta_mem = psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30 - m0
    logging.info(f"Dataset used memory = {delta_mem:.1f}GB")

    total_steps = NUM_EPOCHS * len(train_loader)
    
    warmup_steps = int(WARMUP_RATIO * total_steps)
    logging.info(f'Total train steps={total_steps}, warmup steps={warmup_steps}')

    # model
    model = QQUniModel(MODEL_CONFIG, bert_cfg_dict=BERT_CFG_DICT, model_path=BERT_PATH, task=PRETRAIN_TASK)
    #model.to(DEVICE)

    # optimizer
    optimizer = create_optimizer(model, model_lr=LR, layerwise_learning_rate_decay=LR_LAYER_DECAY)

    # schedueler
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_training_steps=total_steps, num_warmup_steps=warmup_steps)
    #model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    model = torch.nn.parallel.DataParallel(model.to(DEVICE))
    # train
    val_loss = train(model, model_path, train_loader, val_loader, optimizer, 
                     get_pred_and_loss=get_pred_and_loss,
                     scheduler=scheduler, num_epochs=NUM_EPOCHS)
    list_val_loss.append(val_loss)
    
    del train_dataset, val_dataset
    gc.collect()

    logging.info(f"Fold{fold} val_loss_list=" + str([round(kk, 6) for kk in list_val_loss]))

logging.info(f"Val Cv={np.mean(list_val_loss):6.4} +- {np.std(list_val_loss):6.4}")
logging.info("Train finish")
