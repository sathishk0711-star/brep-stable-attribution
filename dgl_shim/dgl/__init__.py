"""Minimal single-graph DGL stand-in covering only the calls used by AAGNet (sum/max PNA towers, avg pooling)."""
import contextlib, types, torch
__version__='shim-0'
class _Edges:
    def __init__(s,g,src,dst,e): s.src={k:v[src] for k,v in g.ndata.items()}; s.dst={k:v[dst] for k,v in g.ndata.items()}; s.data=e
class _Nodes:
    def __init__(s,mailbox): s.mailbox=mailbox
class DGLGraph:
    def __init__(s,src,dst,n):
        s._src=src.long(); s._dst=dst.long(); s._n=int(n); s.ndata={}; s.edata={}
    def edges(s): return s._src,s._dst
    def num_nodes(s): return s._n
    def batch_num_nodes(s): return torch.tensor([s._n])
    @contextlib.contextmanager
    def local_scope(s):
        n0,e0=dict(s.ndata),dict(s.edata)
        try: yield
        finally: s.ndata.clear(); s.ndata.update(n0); s.edata.clear(); s.edata.update(e0)
    def update_all(s,message_func,reduce_func):
        md=message_func(_Edges(s,s._src,s._dst,s.edata))
        order=torch.argsort(s._dst,stable=True); md={k:v[order] for k,v in md.items()}
        deg=torch.bincount(s._dst,minlength=s._n); start=torch.cumsum(deg,0)-deg
        out={}
        for d in torch.unique(deg[deg>0]).tolist():
            nodes=(deg==d).nonzero().flatten()
            idx=start[nodes][:,None]+torch.arange(d)[None,:]
            res=reduce_func(_Nodes({k:v[idx] for k,v in md.items()}))
            for k,v in res.items():
                if k not in out: out[k]=v.new_zeros((s._n,)+tuple(v.shape[1:]))
                out[k]=out[k].index_copy(0,nodes,v)
        for k,v in out.items(): s.ndata[k]=v
def graph(data,num_nodes=None):
    src,dst=data
    return DGLGraph(torch.as_tensor(src),torch.as_tensor(dst),num_nodes)
def unbatch(g): return [g]
class _Pool(torch.nn.Module):
    def forward(s,g,feat): return feat.mean(0,keepdim=True)
nn=types.ModuleType('dgl.nn'); nn.AvgPooling=_Pool; nn.MaxPooling=_Pool
function=types.ModuleType('dgl.function')
import sys; sys.modules['dgl.nn']=nn; sys.modules['dgl.function']=function
