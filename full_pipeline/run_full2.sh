#!/bin/bash
export PYTHONPATH=/home/claude/shim
cd /home/claude/pkg/ACD_Compact_Implementation_MFInstSeg_224/proposed
python3 -W ignore evaluate.py /home/claude/runs/aagnet_full_seed_37/best.pt --out /home/claude/runs/aagnet_full_seed_37/evaluation --steps 16
for s in 53 71; do
 python3 -W ignore train.py --allow-subset --arm full --epochs 6 --pretrain-epochs 1 --steps 8 --seeds $s --threads 2 --out /home/claude/runs
 [ -f /home/claude/runs/aagnet_full_seed_$s/best.pt ] && python3 -W ignore evaluate.py /home/claude/runs/aagnet_full_seed_$s/best.pt --out /home/claude/runs/aagnet_full_seed_$s/evaluation --steps 16
done
echo ALLDONE2
