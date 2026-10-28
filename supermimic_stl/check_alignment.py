"""Compare parameter/buffer layouts with an actual vanilla SAPG checkpoint."""
import argparse,json
from pathlib import Path
import torch
from .aligned import make_model


def main():
    p=argparse.ArgumentParser();p.add_argument('--sapg-checkpoint',type=Path,required=True)
    p.add_argument('--supermimic-checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    sm=torch.load(a.supermimic_checkpoint,map_location='cpu',weights_only=False)
    sapg=torch.load(a.sapg_checkpoint,map_location='cpu',weights_only=False)
    sapg=sapg.get(0,sapg)
    weights=sm['learner']['model'];schema=sm['metadata']['schema'];spec=sm['metadata']['core_cfg']['native_spec']
    actor_dim=schema['obs_dim'];critic_end=max(v[1] for k,v in schema['slices'].items() if k.startswith('critic.'))
    rows={}
    for name,dim in [('actor',actor_dim),('critic',critic_end-actor_dim)]:
        original=sapg['model'] if name=='actor' else {k.removeprefix('model.'):v for k,v in sapg['assymetric_vf_nets'].items() if k.startswith('model.')}
        prefix=f'{name}.model.'
        actual={k[len(prefix):]:v for k,v in weights.items() if k.startswith(prefix)}
        shape=lambda d:{k:list(v.shape) for k,v in d.items()}
        assert shape(original)==shape(actual),f'{name} layout differs from actual SAPG checkpoint'
        model=make_model(spec,dim,sm['metadata']['action_dim'],critic=name=='critic')
        model.load_state_dict(original,strict=True)
        rows[name]=dict(parameter_count=sum(p.numel() for p in model.parameters()),
                        parameter_and_buffer_shapes=shape(actual),matches_actual_sapg_checkpoint=True)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(rows,indent=2));print(json.dumps(rows),flush=True)

if __name__=='__main__': main()
