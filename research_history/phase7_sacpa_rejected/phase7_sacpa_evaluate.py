#!/usr/bin/env python3
from pathlib import Path
import glob,json
import pandas as pd, numpy as np
ROOT=Path(__file__).resolve().parent;RES=ROOT/'results'
files=sorted(glob.glob(str(RES/'P7_sacpa_confirm_70[1-5].jsonl')))
if len(files)!=5: raise SystemExit(f'need 5 files, found {len(files)}')
df=pd.concat([pd.read_json(f,lines=True) for f in files],ignore_index=True)
methods=['SACPA','Local','GlobalMedianOracle','ClusterOracle','SACPA_no_corroboration']
# task sanity unique scenario records
A=bool((df.test_acc>=.90).all())
# helpers
shared={}
for dom in ['patch','rotation']:
 x=df[(df.domain==dom)&(df.scenario=='shared')]
 m=x.groupby('method').oracle_jsd.mean()
 shared[dom]={k:float(m[k]) for k in methods if k in m}
shared_checks={dom:{
 'local_improvement':1-shared[dom]['SACPA']/shared[dom]['Local'],
 'global_ratio':shared[dom]['SACPA']/shared[dom]['GlobalMedianOracle']
} for dom in shared}
B=all(v['local_improvement']>=.50 and v['global_ratio']<=1.10 for v in shared_checks.values())
# family map
het=df[df.scenario!='shared'].copy()
het['family']=het.domain+'_'+het.scenario
fam=het.groupby(['family','method']).oracle_jsd.mean().unstack('method')
# equal-family aggregates
agg=fam.mean(axis=0)
C_metrics={
 'vs_local':1-agg['SACPA']/agg['Local'],
 'vs_global':1-agg['SACPA']/agg['GlobalMedianOracle'],
 'vs_cluster':1-agg['SACPA']/agg['ClusterOracle']}
C=C_metrics['vs_local']>=.20 and C_metrics['vs_global']>=.08 and C_metrics['vs_cluster']>=.08
# D family noninferiority and 3/4 wins
ratios=(fam['SACPA']/fam['GlobalMedianOracle']).to_dict(); wins=int((fam['SACPA']<fam['GlobalMedianOracle']).sum())
D1=all(v<=1.02 for v in ratios.values()) and wins>=3
# patch strength-specific
ps=het[het.domain=='patch'].groupby(['scenario','strength','method']).oracle_jsd.mean().unstack('method')
psratio=(ps['SACPA']/ps['GlobalMedianOracle']).to_dict();D2=all(v<=1.02 for v in psratio.values())
D=D1 and D2
# E seed equal-family means
seedfam=het.groupby(['seed','family','method']).oracle_jsd.mean().unstack('method')
seedavg=seedfam.groupby('seed').mean()
seedwins={b:int((seedavg['SACPA']<seedavg[b]).sum()) for b in ['Local','GlobalMedianOracle','ClusterOracle']}
E=seedwins['Local']==5 and seedwins['GlobalMedianOracle']>=4 and seedwins['ClusterOracle']>=4
# F corroboration
pp=het[(het.domain=='patch')&(het.scenario=='personal')].groupby('method').oracle_jsd.mean()
pers_impr=1-pp['SACPA']/pp['SACPA_no_corroboration']
other=het[~((het.domain=='patch')&(het.scenario=='personal'))].groupby('method').oracle_jsd.mean()
other_worsen=other['SACPA']/other['SACPA_no_corroboration']-1
F=pers_impr>=.20 and other_worsen<=.07
# G active peer mass
peer=float(df[df.method=='SACPA'].peer_mass_mean.mean())
G=peer>=1.5
report={
 'status':'SMOKE_TEST_APPROVED_FOR_CONTINUATION' if all(bool(x) for x in [A,B,C,D,E,F,G]) else 'NOT_APPROVED',
 'gates':{'A_task_sanity':bool(A),'B_homogeneous':bool(B),'C_heterogeneous_aggregate':bool(C),'D_no_family_collapse':bool(D),'E_seed_robustness':bool(E),'F_corroboration':bool(F),'G_peer_active':bool(G)},
 'shared':shared,'shared_checks':shared_checks,'family_means':{str(k):{str(kk):float(vv) for kk,vv in v.items()} for k,v in fam.to_dict(orient='index').items()},
 'aggregate_means':{str(k):float(v) for k,v in agg.to_dict().items()},'aggregate_improvements':{str(k):float(v) for k,v in C_metrics.items()},
 'family_sacpa_over_global':{str(k):float(v) for k,v in ratios.items()},'family_wins_vs_global':wins,
 'patch_strength_sacpa_over_global':{str(k):float(v) for k,v in psratio.items()},
 'seed_equal_family_means':{str(k):{str(kk):float(vv) for kk,vv in v.items()} for k,v in seedavg.to_dict(orient='index').items()},'seed_wins':seedwins,
 'corroboration_personal_patch_improvement':float(pers_impr),'corroboration_other_worsen':float(other_worsen),
 'peer_mass_mean':peer,'n_rows':len(df),'seeds':sorted(df.seed.unique().tolist())}
out=RES/'P7_sacpa_gate_report.json';out.write_text(json.dumps(report,indent=2,sort_keys=True))
print(json.dumps(report,indent=2,sort_keys=True))
