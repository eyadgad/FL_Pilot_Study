# Published baseline registry and scientific comparison contract

Do **not** conflate (i) task-model optimizers, (ii) published explanation methods, and (iii) personalized federated explanation-alignment systems. Numerical dominance over a simplistic baseline does not establish dominance over a competing FL-XAI system.

| Name | Published reference | In this code | Publication use |
|---|---|---|---|
| FedAvg | McMahan et al., AISTATS 2017: *Communication-Efficient Learning of Deep Networks from Decentralized Data* | `ucpa_fl/federated.py:train_fedavg`; task-training baseline | Independent implementation; compare task accuracies under paired data partitions/seeds |
| FedProx | Li et al., MLSys 2020: *Federated Optimization in Heterogeneous Networks* | `ucpa_fl/federated.py:train_fedprox`, fixed µ=0.01; separate task-training baseline | Independent implementation, not official source; not an explanation method |
| Vanilla gradient saliency | Simonyan et al., ICLR Workshop 2014 (arXiv 2013): *Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps* | `study_v7/published_baselines.py:saliency` | Independent implementation, true held-out CNN gradient; no explanation coordination |
| Grad-CAM | Selvaraju et al., ICCV 2017: *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization* | `study_v7/published_baselines.py:gradcam` | Independent implementation, true CNN last-convolution attribution |
| Integrated Gradients | Sundararajan et al., ICML 2017: *Axiomatic Attribution for Deep Networks* | `ucpa_fl/explain.py`; held-out reference/oracle, 8 steps in full study | **Not independent evidence of explanation faithfulness** merely by reproducing it |
| xFedAlign | Wasif et al., ICML 2026: *Explainable Federated Learning via Global–Local Attribution Alignment* | **NOT integrated**; official authors' code: https://github.com/dawoodwasif/xFedAlign | **Required paired official-source port or independently audited reproduction before any SOTA superiority claim** |
| FedAttr-Agg, Fed-XAI | Baseline families documented within the xFedAlign (ICML 2026) benchmark | **NOT integrated**. `fedattr_class_template_proxy` is a custom template proxy, not FedAttr-Agg | Do not label the proxy as the published baseline |
| QF-SCAD v7 | Experimental in-house research candidate | Same original 77-param neural corrector, real-MNIST verification | Candidate; not a published SOTA baseline |
| APGF v6 | Earlier exploratory candidate | Matched 36-fit / 12-validation gated field | Internal historical control; not published method |
| Strong private convolutional corrector | Architecture-matched private 77-param CNN, all 48 teachers | `private_conv_48`, same optimizer budget | Crucial lower bound on federated-only contribution |

**Important**: a paper asserting novelty and competitive superiority MUST integrate and rerun the author-released xFedAlign baseline on the same frozen task checkpoints, real dataset partitions, teacher budgets, explanation references, and communication accounting. The official xFedAlign scripts contain their own task-training/evaluation procedures, so running the public scripts independently and copying their numbers is not an apples-to-apples paired comparison. This v8 package intentionally FAILS CLOSED on the publication-level SOTA promotion gate until that has been done; results from the built-in methods are *not* a completed published-baseline benchmark.

### Bibliography

- xFedAlign paper: https://proceedings.mlr.press/v306/wasif26a.html
- Author source: https://github.com/dawoodwasif/xFedAlign
- FedAvg: https://proceedings.mlr.press/v54/mcmahan17a.html
- FedProx: https://proceedings.mlsys.org/paper_files/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html
- Grad-CAM: https://openaccess.thecvf.com/content_ICCV_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html
- Integrated Gradients: https://proceedings.mlr.press/v70/sundararajan17a.html
- Simonyan saliency: https://arxiv.org/abs/1312.6034
