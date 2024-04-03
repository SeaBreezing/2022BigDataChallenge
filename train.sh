#!/bin/bash

cd src/singleFlow
python main.py

cd ..
cd pretrain/finetune
python main.py

echo "Finish"