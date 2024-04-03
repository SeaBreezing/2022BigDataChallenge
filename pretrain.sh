#!/bin/bash

cd src/pretrain
python download_pretrain_model.py
python pretrain.py

echo "Finish"