from __future__ import annotations
import csv, json, math, os
from pathlib import Path
import torch, torch.distributed as dist
from src.muon import MuonWithAuxAdamW, newton_schulz

def main():
    rank=int(os.environ['RANK']); local=int(os.environ['LOCAL_RANK']); world=int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local); device=torch.device('cuda',local); dist.init_process_group('nccl')
    torch.manual_seed(2026)
    parameter=torch.nn.Parameter(torch.randn(128,512,device=device,dtype=torch.float32))
    local_grad=torch.randn_like(parameter) + rank/10
    dist.all_reduce(local_grad); local_grad.div_(world); parameter.grad=local_grad
    old=parameter.detach().float().clone()
    optimizer=MuonWithAuxAdamW([{'params':[parameter],'use_muon':True}],lr=3e-4,weight_decay=.1,momentum=.95,nesterov=True,ns_method='hybrid',update_rms_target=.18,metrics_interval=1)
    expected_input=local_grad.float()*1.95
    fp32_ns,_=newton_schulz(expected_input.cpu(),method='hybrid')
    fp32_scaled=fp32_ns*math.sqrt(512)*.18
    optimizer.step()
    state=optimizer.state[parameter]['momentum_buffer'].float()
    delta=parameter.detach().float()-old
    gathered=[torch.empty_like(parameter) for _ in range(world)]; dist.all_gather(gathered,parameter.detach())
    rank_error=max((gathered[0].float()-item.float()).abs().max().item() for item in gathered[1:])
    expected_update=(old*(1-3e-4*.1)-parameter.detach().float())/3e-4
    update_rms=expected_update.pow(2).mean().sqrt().item()
    bf16_rel=(expected_update.cpu()-fp32_scaled).norm().item()/fp32_scaled.norm().item()
    momentum_error=(state-local_grad.float()).abs().max().item()
    metrics=torch.tensor([rank_error,update_rms,bf16_rel,momentum_error],device=device,dtype=torch.float64)
    dist.all_reduce(metrics,op=dist.ReduceOp.MAX)
    if rank==0:
        result={'world_size':world,'rank_parameter_max_abs_error':metrics[0].item(),'scaled_update_rms':metrics[1].item(),'rms_relative_error':abs(metrics[1].item()-.18)/.18,'bf16_vs_fp32_relative_update_error':metrics[2].item(),'momentum_max_abs_error':metrics[3].item(),'status':'PASS' if metrics[0]<1e-6 and abs(metrics[1]-.18)/.18<.02 and metrics[2]<.03 else 'FAIL'}
        path=Path('audit_results/tests/gpu_nccl_muon_results.json'); path.write_text(json.dumps(result,indent=2)); print(json.dumps(result))
    dist.destroy_process_group()
if __name__=='__main__': main()
