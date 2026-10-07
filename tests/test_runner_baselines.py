import numpy as np
import torch
from ucpa_fl.risk_control import evaluate_beta_on_records


def test_local_beta_zero_has_zero_excess_risk():
    class M(torch.nn.Module):
        def forward(self,x):
            z=x.flatten(1).sum(1)
            return torch.stack([z,-z],dim=1)
    model=M().eval()
    local=np.linspace(1,0,16); local=local/local.sum()
    rec={'x':np.ones((1,4,4),np.float32),'pred':0,'local_map':local,'oracle_map':local.copy()}
    prior=np.stack([local,local])
    out=evaluate_beta_on_records(model,[rec],prior,beta=0.0,device=torch.device('cpu'),steps=4,topk=4)
    assert abs(out['excess_fidelity_risk']) < 1e-12
    assert abs(out['sample_fidelity_jsd']) < 1e-12
