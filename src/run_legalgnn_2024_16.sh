#!/bin/sh -x

python main.py --model_name LegalGNN --gpu 2 --random_seed 2024 --batch_size 128 --eval_batch_size 8 --num_workers 5 --emb_size 64 --alpha 0.1 --fix_size 16 --trans 1 --transfer 2 --margin 1 --history_length 0 --layers '[64]' --node_dropout '[0.5,0.5]' --lr 1e-3 --l2 1e-5 --dataset 'LAW'
