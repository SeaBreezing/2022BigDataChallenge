import torch
from torch.utils.data import SequentialSampler, DataLoader

from ds_config import parse_args
from data_helper import MultiModalDataset
from category_id_map import lv2id_to_category_id
from model import MultiModal
from data_helper import create_dataloaders
from util import setup_device, setup_seed, setup_logging

from util import evaluate
from functools import partial
import numpy as np
import scipy as sp
from sklearn.metrics import f1_score
class OptimizedF1(object):
    def __init__(self):
        self.coef_ = []
        self.count = 0

    def _kappa_loss(self, coef, X, y):
        """
        y_hat = argmax(coef*X, axis=-1)
        :param coef: (1D array) weights
        :param X: (2D array)logits
        :param y: (1D array) label
        :return: -f1
        """
        X_p = np.copy(X)
        X_p = coef*X_p
        predictions = np.argmax(X_p, axis=-1)
        ll = evaluate(predictions, y)['mean_f1']
        if self.count % 500 == 0:
            print(self.count, ll)
            print(coef)
        self.count += 1
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [1.] * 200 #权重都初始化为1
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='Nelder-Mead', bounds=sp.optimize.Bounds(np.full(200, 0.9), np.full(200, 1.1)), options={'maxiter': 15000})

    def predict(self, X, y):
        X_p = np.copy(X)
        X_p = self.coef_['x'] * X_p
        predictions = np.argmax(X_p, axis=-1)
        return evaluate(predictions, y)['mean_f1']

    def coefficients(self):
        return self.coef_['x']


if __name__ == '__main__':
    args = parse_args()
    setup_logging()
    setup_device(args)
    setup_seed(args)
    train_dataloader, val_dataloader = create_dataloaders(args)
    model = MultiModal(args)
    checkpoint = torch.load(args.ckpt_file, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))
    model.eval()
    logits = []
    labels = []
    with torch.no_grad():
        for batch in val_dataloader:
            logit, label = model(batch, coef=True)
            logits.extend(logit.cpu().numpy())
            labels.extend(label.cpu().numpy())
    print('valid_end')
    op = OptimizedF1()
    op.fit(logits, labels)
    print('search_end')
    logits = op.coefficients() * logits
    predictions = np.argmax(logits, axis=-1)
    results = evaluate(predictions, labels)
    print(results)
    
    #inference
    dataset = MultiModalDataset(args, args.test_annotation, args.test_zip_feats, test_mode=True)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=args.test_batch_size,
                            sampler=sampler,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=args.num_workers,
                            prefetch_factor=args.prefetch)
    logits = []
    with torch.no_grad():
        for batch in dataloader:
            logit = model(batch, inference=True, coef=True)
            logits.extend(logit.cpu().numpy())
         
    logits = op.coefficients() * logits
    predictions = np.argmax(logits, axis=-1)
    with open(args.test_output_csv, 'w') as f:
        for pred_label_id, ann in zip(predictions, dataset.anns):
            video_id = ann['id']
            category_id = lv2id_to_category_id(pred_label_id)
            f.write(f'{video_id},{category_id}\n')
    print('save_end')