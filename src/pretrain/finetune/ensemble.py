import numpy as np  
from ds_config import parse_args
from data_helper import MultiModalDataset
from category_id_map import lv2id_to_category_id

args = parse_args()
# 1. load data
dataset = MultiModalDataset(args, args.test_annotation, args.test_zip_feats, test_mode=True)

predictions_pretrain = np.load(f'../../output/prediction_pretrain.npy')
predictions_singleFlow = np.load(f'../../output/prediction_singleFlow.npy')
predictions = (predictions_pretrain + predictions_singleFlow) / 2
    
#4. dump results
with open(args.test_output_csv, 'w') as f:
    for i in range(len(dataset.anns)):
        video_id = dataset.anns[i]['id']
        pred_label_id = np.argmax(predictions[i], axis=-1)
        category_id = lv2id_to_category_id(pred_label_id)
        f.write(f'{video_id},{category_id}\n')
