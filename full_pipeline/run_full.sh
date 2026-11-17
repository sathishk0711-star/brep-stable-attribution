#!/bin/bash
export PYTHONPATH=/home/claude/shim
cd /home/claude/pkg/ACD_Compact_Implementation_MFInstSeg_224/proposed
for s in 11 23 37; do
 python3 -W ignore train.py --allow-subset --arm full --epochs 6 --pretrain-epochs 1 --steps 8 --seeds $s --threads 2 --out /home/claude/runs
 python3 -W ignore evaluate.py /home/claude/runs/aagnet_full_seed_$s/best.pt --out /home/claude/runs/aagnet_full_seed_$s/evaluation --steps 16
done
echo ALLDONE
