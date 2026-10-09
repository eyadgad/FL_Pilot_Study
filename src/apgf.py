"""APGF v6: Attribution-Predictive Gated Federation (experimental).

All feature gains are computed from real Integrated Gradients teacher records.
No transmission of raw images/oracles. Server accesses only quantized sufficient-statistic packets.
Calibration-only selection: validation teacher records are never used to fit fields,
and final evaluation never selects a candidate. No privacy or generalization guarantee.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter

# Imports are resolved from the supplied v5 implementation, not rewritten.
from ucpa_fl.fagc import encode_stat, decode_stat, encode_gain_full, decode_gain_full, fagc_map
from ucpa_fl.metrics import normalize_np
from ucpa_fl.alignment import jsd

POWER=1.2


def quantized_teacher_stats(records, power=POWER):
    if not records: raise ValueError('No genuine IG teachers')
    A=np.zeros(784);B=np.zeros(784)
    for r in records:
        x=np.asarray(r['x'],float).reshape(-1)
        y=normalize_np(np.asarray(r['oracle_map'],float).reshape(1,-1))[0]
        if x.size!=784 or y.size!=784 or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Invalid teacher record')
        A+=y;B+=np.maximum(x,0)**power
    return encode_stat(A,len(records)),encode_stat(B,len(records))


def _decoded_stats(packet_pair):
    aa,na=decode_stat(packet_pair[0]); bb,nb=decode_stat(packet_pair[1])
    if na!=nb:raise ValueError('Teacher count mismatch')
    return aa,bb,na


def _field(A,B,n,blur=1.):
    g=A/(B+0.01*n)
    if blur:g=gaussian_filter(g.reshape(28,28),blur).reshape(-1)
    return decode_gain_full(encode_gain_full(g))


def candidate_fields(packets, weights=(0.,.1,.2,.4,.8),blur=1.):
    """Only decoded quantized packets enter the server calculations.

    lambda=0 yields the local *quantized* private field with no peer correction.
    lambda>0 pools peer stats (excluding self) and transmits quantized gain maps.
    Each candidate has one 786-byte packet; all candidates must travel downlink for a private validation-based choice.
    """
    decoded=[_decoded_stats(p) for p in packets]
    out=[]
    for i,(a,b,n) in enumerate(decoded):
        peer_a=sum(row[0] for j,row in enumerate(decoded) if j!=i)
        peer_b=sum(row[1] for j,row in enumerate(decoded) if j!=i)
        peer_n=sum(row[2] for j,row in enumerate(decoded) if j!=i)
        c={}
        for w in weights:
            w=float(w)
            f=_field(a+w*peer_a,b+w*peer_b,n+w*peer_n,blur)
            c[w]=f
        out.append(c)
    ledger={'per_client_uplink_bytes':[sum(map(len,p)) for p in packets],
            'per_client_downlink_bytes':len(weights)*len(encode_gain_full(np.ones(784))),
            'candidate_fields_sent_per_client':len(weights),
            'why_all_candidates_sent':'Client keeps validation IG private and therefore must receive all candidate maps to select peer weight',
            'transport':'quantized uint16 support plus two uint8 784-pixel stats; uint8 downlink',
            'peer_only_excludes_self':True,
            'privacy_guarantee':False}
    return out,ledger


@dataclass
class GateDecision:
    selected_weight:float
    val_gain:float
    n_val:int
    private_validation_jsd:float
    selected_validation_jsd:float
    reason:str
    weight_validated:bool


def choose_weight(candidates, validation_records, improvement_margin=0.001,
                  min_pair_win_fraction=0.6, gate_candidates=(0.,.1,.2,.4,.8), power=POWER):
    """Conservative paired JSD gate; weight 0 is always a valid fallback.

    Note: selection optimism remains possible; this is a validation heuristic,
    NOT a distribution-free statistical guarantee.
    """
    vals=list(validation_records)
    if len(vals)<4:raise ValueError('Insufficient held-out IG calibration for safe gating')
    by={}
    for w in gate_candidates:
        pred=[fagc_map(r['x'],np.ones(784),candidates[float(w)],power=power,class_exponent=0) for r in vals]
        by[float(w)]=np.asarray([jsd(p,np.asarray(r['oracle_map'])) for p,r in zip(pred,vals)])
    base=by[0.]; best_w=0.;best_gain=0.;reason='NO_VALIDATED_PEER_GAIN'
    for w in gate_candidates:
        if w==0:continue
        improvement=base-by[float(w)]
        m=float(np.mean(improvement))
        if m<improvement_margin or np.mean(improvement>0)<min_pair_win_fraction:continue
        if m>best_gain:
            best_gain=m;best_w=float(w);reason='PEER_GAIN_ON_RESERVED_TEACHERS'
    return GateDecision(best_w,best_gain,len(vals),float(base.mean()),float(by[best_w].mean()),reason,best_w>0)


def fit_gated_fields(fit_teachers, val_teachers, weights=(0.,.1,.2,.4,.8),
                     improvement_margin=.001,min_pair_win_fraction=.6,power=POWER):
    if len(fit_teachers)!=len(val_teachers):raise ValueError('Client count mismatch')
    packets=[quantized_teacher_stats(t,power=power) for t in fit_teachers]
    cand,ledger=candidate_fields(packets,weights)
    decisions=[choose_weight(c,v,improvement_margin,min_pair_win_fraction,weights,power) for c,v in zip(cand,val_teachers)]
    selected=np.stack([c[d.selected_weight] for c,d in zip(cand,decisions)])
    private=np.stack([c[0.] for c in cand])
    fixed=np.stack([c[.2] for c in cand])
    return {'selected':selected,'private':private,'fixed_peer_0p2':fixed,
            'decisions':[vars(d) for d in decisions], 'ledger':ledger,
            'weights':list(weights)}
