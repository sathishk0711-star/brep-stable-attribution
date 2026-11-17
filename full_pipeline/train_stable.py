"""Repeated-seed training entry point; never interprets an epoch cap as convergence."""
import argparse,json,sys,time,random,hashlib,copy
from pathlib import Path
import numpy as np
import torch
from framework import Proposed,inputs,put
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'AAGNet'))
# shared.py is copied with its original path convention; override ROOT and P below.
sys.path.insert(0,str(ROOT/'validation'))
import shared
from models.inst_segmentors import AAGNetSegmentor

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=ROOT/'validation/subset');p.add_argument('--out',type=Path,default=ROOT/'runs');p.add_argument('--epochs',type=int,default=200);p.add_argument('--pretrain-epochs',type=int,default=50);p.add_argument('--patience',type=int,default=20);p.add_argument('--steps',type=int,default=50);p.add_argument('--seeds',type=int,nargs='+',default=[11,23,37,53,71]);p.add_argument('--arm',choices=['baseline','no_stability','no_consistency','full'],default='full');p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=4);p.add_argument('--allow-subset',action='store_true');p.add_argument('--backbone',choices=['aagnet','original_attention'],default='aagnet');args=p.parse_args()
 torch.set_num_threads(args.threads);shared.P=args.data;shared.stat=json.load(open(args.data/'attr_stat.json'))
 selection=json.load(open(args.data/'selection.json'))
 names=selection['selected']
 if not args.allow_subset and len(names['train'])<1000:raise RuntimeError('Recovered data are a small lexicographic subset. Supply full extracted data or explicitly use --allow-subset. Neither enables confirmatory claims.')
 D={k:[shared.load(n) for n in ns] for k,ns in names.items()}
 for split,rows in D.items():
  for i,(n,g,y,inst,bot) in enumerate(rows):rows[i]=(n,g.to(args.device),y.to(args.device),inst.to(args.device),bot.to(args.device))
 args.out.mkdir(parents=True,exist_ok=True)
 for seed in args.seeds:
  random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
  if args.backbone=='aagnet':
   model=AAGNetSegmentor(**shared.CFG).to(args.device)
   model.load_state_dict(torch.load(ROOT/'AAGNet/weights/weight_on_MFInstseg.pth',map_location=args.device,weights_only=False))
  else:
   from original_encoder import OriginalBackbone
   model=OriginalBackbone(D['train'][0][1].ndata['grid'].shape[1:]).to(args.device)
  net=Proposed(model,D['train'][0][1].ndata['grid'].shape[1:]).to(args.device)
  out=args.out/f'{args.backbone}_{args.arm}_seed_{seed}';out.mkdir(exist_ok=True)
  config=vars(args).copy();config={k:str(v) if isinstance(v,Path) else v for k,v in config.items()};config['seed']=seed;config['counts']={k:len(v) for k,v in D.items()};config['contribution_status']='experimental AAGNet-backbone realization; not exact original heterogeneous encoder';json.dump(config,open(out/'config.json','w'),indent=2)
  opt=torch.optim.AdamW(net.parameters(),lr=.0002,weight_decay=.0001)
  # All arms perform identical training-only masked attribute reconstruction.
  # This does not establish the original large-corpus SSL claim.
  ssl=[]
  for epoch in range(args.pretrain_epochs):
   net.eval();vals=[]
   for j in np.random.permutation(len(D['train'])):
    _,g,*_=D['train'][j];x=g.ndata['x'];mask=torch.rand(len(x),device=x.device)<.15
    if not mask.any():mask[0]=True
    with g.local_scope():
     xm=x.clone();xm[mask]=0;g.ndata['x']=xm;_,raw,_=net.encode(g)
     loss=torch.nn.functional.mse_loss(net.reconstruct(raw)[mask],x[mask]);opt.zero_grad();loss.backward();opt.step();vals.append(float(loss.detach()))
   ssl.append({'epoch':epoch+1,'loss':float(np.mean(vals))});json.dump(ssl,open(out/'ssl_history.json','w'),indent=2)
  # Reset optimizer across stages; identical train order per seed in every arm.
  opt=torch.optim.AdamW(net.parameters(),lr=.0002,weight_decay=.0001)
  best=float('inf');bad=0;history=[];start=time.perf_counter();stopped=False
  for epoch in range(args.epochs):
   net.eval();train=[]
   for j in np.random.default_rng(seed+epoch).permutation(len(D['train'])):
    _,g,y,inst,bot=D['train'][j];opt.zero_grad()
    if args.arm=='baseline':
     a,b,c=model(g);loss=torch.nn.functional.cross_entropy(a,y)+torch.nn.functional.binary_cross_entropy_with_logits(b[0],inst)+torch.nn.functional.binary_cross_entropy_with_logits(c,bot)
    else:
     # Training target uses a class present in TRAIN annotations only.
     c=int(y[y!=24][0]) if (y!=24).any() else 24
     loss,_=net.objective(g,y,inst,bot,c,args.steps,stability=0 if args.arm=='no_stability' else .5,consistency=0 if args.arm=='no_consistency' else .1)
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite training objective')
    loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();train.append(float(loss.detach()))
   val=[]
   with torch.no_grad():
    for _,g,y,_,_ in D['val']:val.append(float(torch.nn.functional.cross_entropy(model(g)[0],y)))
   v=float(np.mean(val));history.append(dict(epoch=epoch+1,train_loss=float(np.mean(train)),val_ce=v,seconds=time.perf_counter()-start));json.dump(history,open(out/'history.json','w'),indent=2)
   state=dict(model=net.state_dict(),optimizer=opt.state_dict(),epoch=epoch+1,seed=seed,torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),config=config)
   torch.save(state,out/'last.pt')
   if v<best-1e-5:best=v;bad=0;torch.save(state,out/'best.pt')
   else:bad+=1
   print(args.arm,seed,history[-1],flush=True)
   if bad>=args.patience:stopped=True;break
  json.dump({'early_stopping_triggered':stopped,'epoch_cap_reached':not stopped,'convergence_claim':'not established; review loss curves, attribution objectives and held-out performance','best_val_ce':best},open(out/'status.json','w'),indent=2)
if __name__=='__main__':main()
