"""Faithful *style* APGF v6 gate generalized to CIFAR spatial fields.
Uses 36 fitting + 12 private teacher-validation records per client, five
quantized peer weights, reserved validation never used to fit any candidates.
"""
from __future__ import annotations
import numpy as np
from study_v7.codec import fit_fields
from study_v7.baselines import static_map
from study_v7.metrics import jsd

def choose_apgf(teachers,hw,weights=(0.,.1,.2,.4,.8),nfit=36,margin=.001,min_win_fraction=.6,power=1.2):
    n=len(teachers)
    if n<2 or any(len(z['x'])!=48 or len(z['ig'])!=48 for z in teachers):
        raise ValueError('Generalized APGF requires 48 original IG teachers/client')
    xs=[z['x'][:nfit] for z in teachers];ys=[z['ig'][:nfit] for z in teachers]
    fields={};up=None;one_down=None
    for w in weights:
        fit,wire=fit_fields(xs,ys,hw,peer_weight=w,power=power)
        fields[w]=fit['private'] if w==0 else fit['fixed']
        if up is None:up=wire['static_uplink_bytes_per_client'];one_down=wire['field_bytes']
    chosen=[];decisions=[]
    for i,z in enumerate(teachers):
        xv=z['x'][nfit:];yv=z['ig'][nfit:]
        scores={w:np.array([jsd(a,b) for a,b in zip(static_map(xv,fields[w][i],power),yv)]) for w in weights}
        base=scores[0.];best=0.;best_gain=0.
        for w in weights:
            if w==0:continue
            delta=base-scores[w]
            gain=float(delta.mean());win=float((delta>0).mean())
            if gain>=margin and win>=min_win_fraction and gain>best_gain:
                best=w;best_gain=gain
        chosen.append(fields[best][i])
        decisions.append({'client':i,'chosen_peer_weight':best,'heldout_val_gain':best_gain,
                          'n_validation':len(xv),'selection_used_final_evaluation':False})
    ledger={'per_client_uplink_bytes':up,'per_client_downlink_bytes':[one_down*len(weights)]*n,
        'per_client_total_bytes':[u+one_down*len(weights) for u in up],
        'candidate_weights':list(weights),'all_candidates_delivered_to_client_for_private_gate':True}
    return np.stack(chosen),decisions,ledger
