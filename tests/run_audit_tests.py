from __future__ import annotations
import copy, csv, json, math, time
from pathlib import Path
import torch, yaml
from src.modeling_v3 import DeepSeekV3LikeLM
from src.muon import MuonWithAuxAdamW, newton_schulz
from src.train import build_optimizer

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'audit_results'

def write_csv(path, rows):
    path=OUT/path
    if not rows: return
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def polar_metrics(matrix, output):
    if not torch.isfinite(output).all(): return float('nan'),float('nan'),float('nan'),True
    u,_,vh=torch.linalg.svd(matrix.float(),full_matrices=False); ref=u@vh
    rel=torch.linalg.vector_norm(output.float()-ref)/torch.linalg.vector_norm(ref).clamp_min(1e-30)
    gram=output.float()@output.float().mT if output.shape[0]<=output.shape[1] else output.float().mT@output.float()
    orth=torch.linalg.vector_norm(gram-torch.eye(gram.shape[0]))/math.sqrt(gram.shape[0])
    sv=torch.linalg.svdvals(output.float()); dev=(sv-1).abs().max()
    return rel.item(),orth.item(),dev.item(),False

def legacy_ns(matrix, method, iterations=8, eps=1e-7):
    x=matrix.float(); trans=x.shape[0]>x.shape[1]
    if trans:x=x.mT
    x=x/x.norm().clamp_min(eps)
    for i in range(iterations):
        gram=x@x.mT
        if method=='standard' or i>=min(3,iterations): x=1.5*x-.5*(gram@x)
        else: x=3.4445*x+(-4.775*gram+2.0315*(gram@gram))@x
    return x.mT if trans else x

def numerical_tests():
    torch.manual_seed(2026); cases=[]
    specs=[('square_gaussian',32,32,None),('tall_gaussian',64,16,None),('wide_gaussian',16,64,None),
           ('condition_1e3',32,32,1e3),('near_rank_deficient',32,24,1e8),('zero',16,16,'zero'),
           ('tiny_norm',24,16,'tiny'),('large_norm',24,16,'large')]
    for name,r,c,kind in specs:
        if kind=='zero': matrix=torch.zeros(r,c)
        elif kind in ('tiny','large'): matrix=torch.randn(r,c)*(1e-30 if kind=='tiny' else 1e20)
        elif isinstance(kind,float):
            u,_=torch.linalg.qr(torch.randn(r,min(r,c))); v,_=torch.linalg.qr(torch.randn(c,min(r,c)))
            s=torch.logspace(0,-math.log10(kind),min(r,c)); matrix=(u*s)@v.mT
        else: matrix=torch.randn(r,c)
        for method,label,call in [
            ('legacy_v4_8','current_pre_fix_hybrid',lambda:legacy_ns(matrix,'v4',8)),
            ('hybrid','report_hybrid_8_plus_2',lambda:newton_schulz(matrix,method='hybrid')[0]),
            ('standard','standard_10',lambda:newton_schulz(matrix,method='standard')[0])]:
            start=time.perf_counter()
            try: out=call(); elapsed=time.perf_counter()-start; rel,orth,dev,bad=polar_metrics(matrix,out)
            except Exception: elapsed=time.perf_counter()-start; rel=orth=dev=float('nan'); bad=True
            cases.append(dict(case=name,shape=f'{r}x{c}',implementation=label,relative_polar_factor_error=rel,
                orthogonality_error=orth,max_singular_value_deviation=dev,nan_or_inf=bad,runtime_seconds=elapsed))
    write_csv('ns_numerical_tests.csv',cases)

def rms_tests():
    rows=[]; torch.manual_seed(7)
    for r,c in [(64,64),(128,512),(512,128),(768,3072),(3072,768)]:
        matrix=torch.randn(r,c); out,_=newton_schulz(matrix,method='hybrid')
        before=out.float().pow(2).mean().sqrt().item(); factor=math.sqrt(max(r,c)); after=before*factor*.18
        legacy=before*math.sqrt(max(1,r/c))
        rows.append(dict(module='synthetic',shape=f'{r}x{c}',gradient_rms=matrix.pow(2).mean().sqrt().item(),momentum_rms='',
            nesterov_input_rms='',ns_output_rms=before,scaling_factor=factor,gamma=.18,scaled_update_rms=after,
            legacy_scaled_update_rms=legacy,relative_error_to_0_18=abs(after-.18)/.18,final_delta_rms=after*3e-4,
            parameter_rms='',delta_rms_over_parameter_rms='',frobenius_update_to_weight_ratio='',update_spectral_norm=torch.linalg.matrix_norm(out*factor*.18,2).item()))
    write_csv('update_scale_analysis.csv',rows)

def one_step_tests():
    rows=[]; torch.manual_seed(11); mu=.95; lr=3e-4; wd=.1
    p=torch.nn.Parameter(torch.randn(8,5)); old=p.detach().clone(); prior=torch.randn_like(p); grad=torch.randn_like(p)
    opt=MuonWithAuxAdamW([{'params':[p],'use_muon':True}],lr=lr,weight_decay=wd,momentum=mu,nesterov=True,
        ns_method='hybrid',ns_iterations=10,update_rms_target=.18,metrics_interval=1)
    opt.state[p]['momentum_buffer']=prior.clone(); p.grad=grad.clone()
    expected_m=mu*prior+grad; expected_n=mu*expected_m+grad
    expected_o,_=newton_schulz(expected_n,method='hybrid'); expected_o*=math.sqrt(max(p.shape))*.18
    expected_new=old*(1-lr*wd)-lr*expected_o
    opt.step(); state=opt.state[p]['momentum_buffer']; delta=p.detach()-old
    checks=[('momentum',state,expected_m),('nesterov_input_formula',expected_n,mu*state+grad),('final_weight',p.detach(),expected_new),
            ('decoupled_weight_decay_delta',delta,expected_new-old)]
    for name,obs,exp in checks:
        err=(obs-exp).abs().max().item(); rows.append(dict(check=name,shape=list(obs.shape),dtype=str(obs.dtype),device=str(obs.device),
            observed_rms=obs.float().pow(2).mean().sqrt().item(),expected_rms=exp.float().pow(2).mean().sqrt().item(),max_abs_error=err,
            relative_error=err/exp.abs().max().clamp_min(1e-30).item(),status='PASS' if err<2e-6 else 'FAIL'))
    write_csv('one_step_reference_check.csv',rows)

def parameter_groups():
    cfg=yaml.safe_load((ROOT/'configs/rerun_v4_muon_hybrid.yaml').read_text()); model=DeepSeekV3LikeLM(cfg); opt=build_optimizer(model,cfg['train'],False)
    assignments={}; group_by={}
    for gid,g in enumerate(opt.param_groups):
        for p in g['params']: assignments.setdefault(id(p),[]).append('Muon' if g.get('use_muon') else 'AdamW'); group_by[id(p)]=(gid,g)
    modules=dict(model.named_modules()); rows=[]
    for name,p in model.named_parameters():
        module_name=name.rsplit('.',1)[0] if '.' in name else ''; assigned=assignments.get(id(p),[]); gid,g=group_by.get(id(p),('',{}))
        low=name.lower(); expected='AdamW' if ('embed_tokens' in low or 'lm_head' in low or 'norm' in low or '.router.' in low or '.ffn.gate.' in low or p.ndim<2) else ('Muon' if p.ndim in {2,3} else 'UNVERIFIABLE')
        actual=assigned[0] if len(assigned)==1 else ('MISSING' if not assigned else 'DUPLICATE')
        status='PASS' if actual==expected or expected=='UNVERIFIABLE' else 'FAIL'
        reason='report semantic exclusion' if expected=='AdamW' else ('logical matrix or packed logical matrices' if expected=='Muon' else 'report does not disclose tensor-rank handling')
        rows.append(dict(parameter_name=name,module_name=module_name,shape='x'.join(map(str,p.shape)),ndim=p.ndim,numel=p.numel(),requires_grad=p.requires_grad,
            assigned_optimizer=actual,optimizer_group_id=gid,learning_rate=g.get('lr',''),weight_decay=g.get('weight_decay',''),momentum=g.get('momentum',''),
            beta1=(g.get('betas') or ('',''))[0],beta2=(g.get('betas') or ('',''))[1],epsilon=g.get('eps',''),nesterov=g.get('nesterov',''),
            use_newton_schulz=bool(g.get('use_muon')),ns_method=g.get('ns_method',''),ns_iterations=(g.get('ns_first_stage_steps',0)+g.get('ns_second_stage_steps',0) if g.get('ns_method') in {'v4','hybrid'} else g.get('ns_standard_steps','')),
            update_rms_target=g.get('update_rms_target',''),reason_for_assignment=reason,expected_optimizer_from_report=expected,compliance_status=status))
    write_csv('optimizer_parameter_groups.csv',rows)
    summary={'total_numel':sum(r['numel'] for r in rows),'muon_numel':sum(r['numel'] for r in rows if r['assigned_optimizer']=='Muon'),
      'adamw_numel':sum(r['numel'] for r in rows if r['assigned_optimizer']=='AdamW'),'duplicates':sum(r['assigned_optimizer']=='DUPLICATE' for r in rows),
      'missing':sum(r['assigned_optimizer']=='MISSING' for r in rows),'noncompliant':sum(r['compliance_status']=='FAIL' for r in rows),
      'unverifiable':sum(r['expected_optimizer_from_report']=='UNVERIFIABLE' for r in rows)}
    (OUT/'parameter_group_summary.json').write_text(json.dumps(summary,indent=2))

def main():
    numerical_tests(); rms_tests(); one_step_tests(); parameter_groups(); print('audit tests completed')
if __name__=='__main__': main()
