#!/usr/bin/env python3
from pathlib import Path
import glob,json
import pandas as pd
ROOT=Path(__file__).resolve().parent;RES=ROOT/'results'
files=sorted(glob.glob(str(RES/'P8_ncsacpa_confirm_80[1-5].jsonl')))
if len(files)!=5: raise SystemExit(f'expected 5 files, got {len(files)}')
df=pd.concat([pd.read_json(f,lines=True) for f in files],ignore_index=True)
A=bool((df.test_acc>=.90).all())
# shared
shared={}
for dom in ['patch','rotation']:
 m=df[(df.domain==dom)&(df.scenario=='shared')].groupby('method').oracle_jsd.mean()
 shared[dom]={k:float(v) for k,v in m.items()}
shared_checks={d:{'local_improvement':1-v['NC_SACPA']/v['Local'],'beats_global':bool(v['NC_SACPA']<v['GlobalMedianOracle'])} for d,v in shared.items()}
B=all(v['local_improvement']>=.35 and v['beats_global'] for v in shared_checks.values())
# heterogeneous families
het=df[df.scenario!='shared'].copy();het['family']=het.domain+'_'+het.scenario
fam=het.groupby(['family','method']).oracle_jsd.mean().unstack('method');agg=fam.mean(0)
Cimp={'vs_local':1-agg['NC_SACPA']/agg['Local'],'vs_global':1-agg['NC_SACPA']/agg['GlobalMedianOracle'],'vs_cluster':1-agg['NC_SACPA']/agg['ClusterOracle']}
C=Cimp['vs_local']>=.30 and Cimp['vs_global']>=.10 and Cimp['vs_cluster']>=.10
family_wins={f:bool(r.NC_SACPA<r.GlobalMedianOracle) for f,r in fam.iterrows()}
ps=het[het.domain=='patch'].groupby(['scenario','strength','method']).oracle_jsd.mean().unstack('method')
strength_wins={str(k):bool(r.NC_SACPA<r.GlobalMedianOracle) for k,r in ps.iterrows()}
D=all(family_wins.values()) and all(strength_wins.values())
# seed robustness
seedfam=het.groupby(['seed','family','method']).oracle_jsd.mean().unstack('method');seedavg=seedfam.groupby('seed').mean()
seedwins={b:int((seedavg.NC_SACPA<seedavg[b]).sum()) for b in ['Local','GlobalMedianOracle','ClusterOracle']}
E=all(v==5 for v in seedwins.values())
# mechanism
sg=het[(het.domain=='patch')&(het.scenario=='grouped')&(het.strength=='strong')].groupby('method').oracle_jsd.mean()
strong_impr=1-sg.NC_SACPA/sg.SACPA_global_count
other=het[~((het.domain=='patch')&(het.scenario=='grouped')&(het.strength=='strong'))].groupby('method').oracle_jsd.mean()
other_worsen=other.NC_SACPA/other.SACPA_global_count-1
F=strong_impr>=.15 and other_worsen<=.20
peer=float(df[df.method=='NC_SACPA'].peer_mass_mean.mean());G=peer>=1.5
passes=[A,B,C,D,E,F,G]
report={
 'status':'SMOKE_TEST_APPROVED_FOR_CONTINUATION' if all(bool(x) for x in passes) else 'NOT_APPROVED',
 'gates':{'A_task_sanity':bool(A),'B_shared':bool(B),'C_heterogeneous_aggregate':bool(C),'D_no_family_strength_collapse':bool(D),'E_seed_robustness':bool(E),'F_revision_mechanism':bool(F),'G_active_federation':bool(G)},
 'shared':shared,'shared_checks':shared_checks,
 'family_means':{str(k):{str(kk):float(vv) for kk,vv in v.items()} for k,v in fam.to_dict('index').items()},
 'aggregate_means':{str(k):float(v) for k,v in agg.items()},'aggregate_improvements':{k:float(v) for k,v in Cimp.items()},
 'family_wins_vs_global':family_wins,'patch_strength_wins_vs_global':strength_wins,
 'seed_equal_family_means':{str(k):{str(kk):float(vv) for kk,vv in v.items()} for k,v in seedavg.to_dict('index').items()},'seed_wins':seedwins,
 'strong_grouped_improvement_vs_predecessor':float(strong_impr),'other_families_worsen_vs_predecessor':float(other_worsen),
 'peer_mass_mean':peer,'n_rows':int(len(df)),'seeds':sorted(int(x) for x in df.seed.unique())}
path=RES/'P8_ncsacpa_gate_report.json';path.write_text(json.dumps(report,indent=2,sort_keys=True));print(json.dumps(report,indent=2,sort_keys=True))
