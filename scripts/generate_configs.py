#!/usr/bin/env python3
from pathlib import Path
from copy import deepcopy
import yaml

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'configs'
SEEDS=[6101,6102,6103,6104,6105]

def base(name,dataset='mnist'):
    is_cifar=dataset=='cifar10'
    return {
      'experiment_name':name,'seeds':SEEDS,'device':'auto','deterministic':True,'num_workers':2,
      'dataset':{'name':dataset,'root':'./data','download':True},
      'federation':{'n_clients':8,'rounds':15,'local_epochs':1,'batch_size':64,'lr':0.01,'momentum':0.9,'weight_decay':1e-4,
                    'participation_rate':1.0,'partition':'dirichlet','dirichlet_alpha':0.1,'min_client_samples':40,'sample_size_sigma':0.0},
      'shift':{'kind':'none','strength':1.0,'rotation_max_deg':25.0,'patch_size':4 if not is_cifar else 5,'patch_value':1.0,'drift_start_round':5,'drift_end_round':15},
      'surrogate':{'kind':'linear','source':'linear_surrogate','epochs':3,'lr':0.1,'temperature':3.0,'l1':1e-4,'hidden_dim':128,
                   'max_samples_per_class':32,'eval_samples_per_class':24,'ig_steps':20 if not is_cifar else 16,'artifact_samples_per_class':24},
      'artifact':{'topk':128 if not is_cifar else 256,'clip_radius':5.0,'quant_bits':8,'dp_sigma':0.1,'missing_variance_scale':4.0,
                  'send_variance':True,'sparse_intersection_only':True},
      'alignment':{'beta':0.2,'methods':['local','fedattr_mean','xfedalign_median','ucpa','ucpa_whole_only','ucpa_coord_only','cluster'],
                   'ucpa_z':2.0,'ucpa_h':0.048,'ucpa_variance_floor_fraction':0.05,'cluster_k':3},
      'evaluation':{'deletion_steps':20,'eval_batch_size':128,'max_eval_samples':256,'compute_oracle_local_fidelity':True,
                    'oracle_samples_per_class':64,'topk_overlap_k':128 if not is_cifar else 256,'bootstrap_samples':5000,
                    'deletion_insertion_samples_per_client':24,'primary_metric':'artifact_fidelity_jsd'},
      'attack':{'enabled':False,'fraction':0.25,'strength':0.3,'kind':'artifact_shift'},
      'logging':{'output_root':'./outputs','save_checkpoints':True,'save_artifacts':True,'log_every_round':True},
      'tags':{'status':'confirmatory','frozen_ucpa':'z=2,h=0.048,beta=0.2','suite_version':'1.0'}
    }

def dump(cfg,fn):
    (OUT/fn).write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')

configs=[]
def add(fn,cfg,tier,role):
    dump(cfg,fn); configs.append({'config':fn,'tier':tier,'role':role,'experiment_name':cfg['experiment_name']})

# Core confirmation: ordinary non-IID and two different forms of explanation heterogeneity.
for ds in ['mnist','cifar10']:
    c=base(f'core_{ds}_label_skew',ds); add(f'core_{ds}_label_skew.yaml',c,'A','ordinary non-IID control')
    c=base(f'core_{ds}_rotation',ds); c['shift']['kind']='rotation'; c['shift']['rotation_max_deg']=30.0; add(f'core_{ds}_rotation.yaml',c,'A','coherent covariate/explanation shift')
    if ds=='mnist':
        c=base('core_mnist_patch',ds); c['shift']['kind']='patch'; add('core_mnist_patch.yaml',c,'A','sparse feature heterogeneity')
    else:
        c=base('core_cifar10_color',ds); c['shift']['kind']='color'; add('core_cifar10_color.yaml',c,'A','client-specific photometric shift')

# Heteroskedastic client sample counts: tests the uncertainty term.
c=base('stress_mnist_rotation_unequal_sizes'); c['shift']['kind']='rotation'; c['federation']['sample_size_sigma']=1.0; add('stress_mnist_rotation_unequal_sizes.yaml',c,'B','uncertainty/heteroskedasticity stress')
c=base('stress_cifar10_rotation_unequal_sizes','cifar10'); c['shift']['kind']='rotation'; c['federation']['sample_size_sigma']=1.0; add('stress_cifar10_rotation_unequal_sizes.yaml',c,'B','uncertainty/heteroskedasticity stress')

# Task participation stress. Explanation artifacts are still evaluated at final deployment snapshot; label this honestly.
c=base('stress_mnist_rotation_partial_task_participation'); c['shift']['kind']='rotation'; c['federation']['participation_rate']=0.5; c['tags']['scope_note']='partial participation applies to FedAvg task training; explanation snapshot includes all clients'; add('stress_mnist_rotation_partial_task_participation.yaml',c,'B','partial task participation')

# Drift stress: training environment shifts over rounds, evaluation at final round.
c=base('stress_mnist_drift_rotation'); c['shift']['kind']='drift_rotation'; c['shift']['rotation_max_deg']=35.0; c['tags']['scope_note']='final-snapshot explanation evaluation after time-varying training shift'; add('stress_mnist_drift_rotation.yaml',c,'B','concept/covariate drift')

# Explainer-source robustness: coordinate the task model's direct IG artifacts instead of a distilled linear surrogate.
for ds in ['mnist','cifar10']:
    c=base(f'stress_{ds}_rotation_task_ig',ds); c['shift']['kind']='rotation'; c['surrogate']['source']='task_ig'; c['tags']['status']='stress'; c['tags']['scope_note']='tests coordination-rule dependence on surrogate choice'; add(f'stress_{ds}_rotation_task_ig.yaml',c,'B','explainer-source robustness')

# Robustness: paired clean and attacked results are emitted from each run.
for kind in ['artifact_shift','random_support']:
    c=base(f'robust_mnist_rotation_{kind}'); c['shift']['kind']='rotation'; c['attack'].update({'enabled':True,'fraction':0.25,'strength':0.3,'kind':kind}); add(f'robust_mnist_rotation_{kind}.yaml',c,'B','attribution poisoning')

# Communication/privacy sweeps. Five seeds retained; these are one-factor-at-a-time.
for k in [32,64,128,256]:
    c=base(f'sweep_mnist_topk_{k}'); c['shift']['kind']='rotation'; c['artifact']['topk']=k; c['evaluation']['topk_overlap_k']=k; c['tags']['status']='sensitivity'; add(f'sweep_mnist_topk_{k}.yaml',c,'C','communication sparsity sensitivity')
for sigma in [0.0,0.05,0.1,0.2]:
    tok=str(sigma).replace('.','p')
    c=base(f'sweep_mnist_dp_{tok}'); c['shift']['kind']='rotation'; c['artifact']['dp_sigma']=sigma; c['tags']['status']='sensitivity'; add(f'sweep_mnist_dp_{tok}.yaml',c,'C','artifact noise sensitivity')

# UCPA sensitivity around the frozen smoke-test values. Mark exploratory; core claims use frozen values only.
for z,h in [(1.0,.048),(4.0,.048),(2.0,.024),(2.0,.096)]:
    tok=f'z{z:g}_h{h:g}'.replace('.','p')
    c=base(f'sensitivity_mnist_{tok}'); c['shift']['kind']='rotation'; c['alignment']['ucpa_z']=z; c['alignment']['ucpa_h']=h; c['tags']['status']='exploratory_sensitivity'; add(f'sensitivity_mnist_{tok}.yaml',c,'C','UCPA parameter sensitivity')

manifest={'suite_version':'1.0','frozen_confirmatory_seeds':SEEDS,'tiers':{'A':'required core evidence','B':'stress/robustness','C':'sensitivity/communication'},'configs':configs}
(OUT/'suite_manifest.yaml').write_text(yaml.safe_dump(manifest,sort_keys=False),encoding='utf-8')
print(f'wrote {len(configs)} configs')
