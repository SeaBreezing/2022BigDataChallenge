#!/bin/bash

mkdir src/output

cd src/singleFlow
python inference.py

cd ..
cd pretrain/finetune
python inference_kfold.py

python ensemble.py

echo "Finish"


