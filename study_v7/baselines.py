"""Comparison controls; no official xFedAlign reproduction is claimed."""
from __future__ import annotations
import numpy as np
from study_v7.codec import norm

def static_map(x,field,power=1.2):
    return norm(np.maximum(np.asarray(x),0).reshape(len(x),-1)**power * np.asarray(field).reshape(1,-1))

def fedattr_class_prior(teachers,nclasses=10):
    """Explicit nonofficial FedAttr class-template control.
    Each client sends quantized positive per-class IG summaries; high byte cost.
    Prediction uses target-model predicted class only. No evaluation IG is sent.
    """
    from study_v7.codec import field_encode,field_decode
    d=teachers[0]['ig'].shape[1]
    maps=[[] for _ in range(nclasses)];wire=0
    for data in teachers:
        for cl in range(nclasses):
            idx=np.flatnonzero(data['pred']==cl)
            if len(idx):
                payload=field_encode(data['ig'][idx].mean(axis=0))
                wire+=len(payload)
                maps[cl].append((field_decode(payload,d),len(idx)))
    field=[]
    for cl in range(nclasses):
        if maps[cl]:field.append(norm(np.average(np.stack([p[0] for p in maps[cl]]),axis=0,weights=[p[1] for p in maps[cl]])[None])[0])
        else:field.append(np.ones(d)/d)
    return np.stack(field),{'uplink_total_all_clients_bytes':wire,
             'downlink_per_client_bytes':d*nclasses*1+2*nclasses,
             'type':'Nonofficial classwise FedAttr proxy, not xFedAlign'}
