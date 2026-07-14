from __future__ import annotations
import datetime, socket, torch, torch.distributed as dist, torch.multiprocessing as mp
from src.muon import MuonWithAuxAdamW

def worker(rank,world,port,queue):
    dist.init_process_group('gloo',init_method=f'tcp://127.0.0.1:{port}',rank=rank,world_size=world,timeout=datetime.timedelta(seconds=30))
    torch.manual_seed(123); p=torch.nn.Parameter(torch.randn(12,7)); local=torch.randn_like(p)+rank
    dist.all_reduce(local); local/=world; p.grad=local
    opt=MuonWithAuxAdamW([{'params':[p],'use_muon':True}],lr=3e-4,weight_decay=.1,momentum=.95,nesterov=True,ns_method='hybrid',update_rms_target=.18)
    opt.step(); gathered=[torch.empty_like(p) for _ in range(world)]; dist.all_gather(gathered,p.detach());
    if rank==0: queue.put(max((gathered[0]-x).abs().max().item() for x in gathered[1:]))
    dist.destroy_process_group()
if __name__=='__main__':
    ctx=mp.get_context('spawn'); q=ctx.SimpleQueue()
    sock=socket.socket(); sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]; sock.close()
    ps=[ctx.Process(target=worker,args=(r,2,port,q)) for r in range(2)]
    [p.start() for p in ps]; [p.join() for p in ps]
    err=q.get(); print(f'ddp_max_abs_error={err:.9g}'); raise SystemExit(0 if err<1e-7 and all(p.exitcode==0 for p in ps) else 1)
