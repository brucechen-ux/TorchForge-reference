from pathlib import Path
import copy,sys,yaml
root=Path(__file__).resolve().parents[1]
b=yaml.safe_load((root/'configs/rerun_v4_muon_hybrid.yaml').read_text()); c=yaml.safe_load((root/'configs/rerun_muon_standard_ns.yaml').read_text())
for cfg in (b,c):
 cfg['model']['name']='SAME'; cfg['train']['output_dir']='SAME'; cfg['train']['tensorboard_dir']='SAME'; cfg['train']['optimizer']['newton_schulz']='SAME'
if b!=c:
 print('FAIL: B/C differ outside allowed identity and NS method fields'); sys.exit(1)
print('PASS: B/C differ only in run identity and Newton-Schulz method')
