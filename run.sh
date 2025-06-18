#!/bin/bash

your_python_path=/home/wanmx/anaconda3/envs/pytorch/bin/python

ds="bindingdb kiba davis"
seeds="42 84 126 168 210"

for d in $ds; do
    for s in $seeds; do
        $your_python_path train.py \
            --root "/mnt/c/Users/l/Desktop/G1work/Panpep/PanPep-main" \
            --gnn "gat_gcn gcn gin gat" \
            --dataset "$d" \
            --seed "$s"  \
            --spt 40
    done
done
