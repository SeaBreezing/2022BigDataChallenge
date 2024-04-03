import sys
sys.path.append("..")
from config.data_cfg import *
from config.model_cfg import *
# from config.finetune_cfg import *
import logging
import os
import time
import torch

from ds_config import parse_args
from data_helper import create_dataloaders
from model import MultiModal
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
from tqdm import tqdm
from tricks import FGM, EMA
import torchcontrib

from qqmodel.qq_uni_model import QQUniModel
BERT_PATH = os.path.join('../', BERT_PATH)



def validate(model, val_dataloader):
    model.eval()
    predictions = []
    labels = []
    losses = []
    with torch.no_grad():
        for batch in val_dataloader:
            loss, _, pred_label_id, label = model(batch)
            loss = loss.mean()
            predictions.extend(pred_label_id.cpu().numpy())
            labels.extend(label.cpu().numpy())
            losses.append(loss.cpu().numpy())
    loss = sum(losses) / len(losses)
    results = evaluate(predictions, labels)

    model.train()
    return loss, results


def train_and_validate(args):
    # 1. load data
#     K_foldNum = 10  # 设置5折交叉验证
    for i in range(0, 1):
        train_dataloader, val_dataloader = create_dataloaders(args, i)
    #train_dataloader, val_dataloader = create_dataloaders(args)

    # 2. build model and optimizers
        pretrain_model = QQUniModel(MODEL_CONFIG, bert_cfg_dict=BERT_CFG_DICT, model_path=BERT_PATH)
        pretrain_model.load_state_dict(torch.load(args.pretrain_path), strict=False)
        model = MultiModal(args, pretrain_model)
    #     for n, p in model.named_parameters():
    #         print(n)
    #     print('load_success')
        optimizer, scheduler = build_optimizer(args, model)
        #optimizer = torchcontrib.optim.SWA(base_optimizer)
    #     for grouped in base_optimizer.param_groups:
    #         print(grouped['lr']) 
        if args.device == 'cuda':
            model = torch.nn.parallel.DataParallel(model.to(args.device))

        # 3. training
        step = 0
        best_score = args.best_score
        start_time = time.time()
        num_total_steps = len(train_dataloader) * args.max_epochs
        fgm = FGM(model, eps=0.5)
        ema = EMA(model, 0.9995)
        ema.register()
        valid_f1 = 0
        f = open('training log.txt', "w")
        for epoch in range(args.max_epochs):
            loop = tqdm(train_dataloader, total = len(train_dataloader))
            for batch in loop:
                model.train()
                loss, accuracy, _, _ = model(batch)
                loss = loss.mean()
                accuracy = accuracy.mean()
                loss.backward()
                fgm.attack()
                loss_adv, _, _, _ = model(batch)
                loss_adv = loss_adv.mean()
                loss_adv.backward()
                fgm.restore()
                optimizer.step()
                ema.update()
                optimizer.zero_grad()
                scheduler.step()

                loop.set_description(f'Epoch [{epoch}]')
                loop.set_postfix(loss=loss.item(), acc=accuracy.item(), valid_f1=valid_f1)
                #optimizer.bn_update(train_dataloader, model)

                step += 1
                if step % args.print_steps == 0:
                    time_per_step = (time.time() - start_time) / max(1, step)
                    remaining_time = time_per_step * (num_total_steps - step)
                    remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                    f.write(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}, accuracy {accuracy:.3f}, lr {optimizer.param_groups[0]['lr']:.3e}" + '\n')

                # 4. validation
                if step % 500 == 0:
                    ema.apply_shadow()    
                    loss, results = validate(model, val_dataloader)
                    results = {k: round(v, 4) for k, v in results.items()}
                    valid_f1 = results['mean_f1']
                    f.write(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}" + '\n')

                    # 5. save checkpoint
                    mean_f1 = results['mean_f1']
                    if mean_f1 > best_score:
                        best_score = mean_f1
                        torch.save(
                                {
                                    "Fold": i,
                                    "epoch": epoch,
                                    "model_state_dict": model.module.state_dict(),
                                    "mean_f1": mean_f1,
                                },
                                f"{args.savedmodel_path}/model_Fold_{i}.bin",
                            )
                    ema.restore()

#             if epoch > 1 and step % 3000 == 0:
#                 ema.apply_shadow()
#                 optimizer.update_swa()
#                 ema.restore()

#     print('swa_evaluate')
#     optimizer.swap_swa_sgd()
#     optimizer.bn_update(train_dataloader, model)
#     loss, results = validate(model, val_dataloader)
#     results = {k: round(v, 4) for k, v in results.items()}
#     valid_f1 = results['mean_f1']
#     f.write(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}" + '\n')
#     print(results)

#     # 5. save checkpoint
#     mean_f1 = results['mean_f1']
#     if mean_f1 > best_score:
#         best_score = mean_f1
#         torch.save(
#                 {
#                     "Fold": i,
#                     "epoch": epoch,
#                     "model_state_dict": model.module.state_dict(),
#                     "mean_f1": mean_f1,
#                 },
#                 f"{args.savedmodel_path}/epoch_{epoch}_mean_f1_{mean_f1}.bin",
#             )

def main():
    args = parse_args()
    setup_logging()
    setup_device(args)
    setup_seed(args)

    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)

    train_and_validate(args)


if __name__ == '__main__':
    main()
