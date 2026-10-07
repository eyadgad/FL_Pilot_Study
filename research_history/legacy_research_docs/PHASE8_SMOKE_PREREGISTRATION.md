# Phase VIII pre-registration: NC-SACPA smoke test

Frozen before running seeds **801--805**.

## Candidate

**NC-SACPA — Neighborhood-Corroborated Scale-Adaptive Peer Alignment**.

For client `i`, compute pairwise JSD distances `d_ik` between sparse explanation artifacts and local scale

`σ_i = median_{r != i} d_ir`.

Use the symmetric self-tuning kernel

`g_ik = exp(-(d_ik / sqrt(σ_i σ_k))^2)`.

The target client is excluded from its peer prior. For coordinates already reported by the target, the peer prior is the `g_ik`-weighted mean over reporting peers. For a coordinate absent from the target, define the target's **local corroboration neighborhood** as its three highest-weight peers. The missing coordinate may be imported only if at least **two of those three peers** report it; if imported, only those local-neighborhood peers contribute to that coordinate. If no peer supports a coordinate, the prior falls back to the target's own value.

Final alignment:

`A_i = normalize(0.2 E_i + 0.8 P_i)`.

Frozen settings:
- self-tuning squared-distance kernel (`p=2`);
- local scale = median peer JSD;
- local corroboration neighborhood size = 3 peers;
- missing-coordinate corroboration = at least 2 of 3 neighbors;
- `beta = 0.8`;
- top-k artifact = 128;
- smoke artifact Gaussian noise = 0.10 at the same scale used in Phase VI;
- no variance channel.

Selected only using development seeds 701 and 702 after Phase-VII SACPA failed its frozen strong-grouped-patch gate. Seeds 801--805 are untouched.

## Baselines

All methods see exactly the same sparse/noised artifacts.

- Local / self-prior.
- Oracle-tuned GlobalMedian over beta `{0.1,0.2,0.4,0.6,0.8,1.0}`.
- Oracle-tuned Cluster over `k={2,3,4,6}` and the same beta grid.
- **SACPA-global-count**: Phase-VII method (`p=2, beta=.6`) using raw global two-peer support counting rather than the local-neighborhood corroboration rule. This is the direct predecessor/mechanism ablation.

GlobalMedian and Cluster are deliberately oracle-tuned on confirmation data, advantaging them.

## Fresh confirmation data

Binary MNIST 3-vs-8 controlled explanation study with top-128 sparse artifacts.

Families:
- patch shared;
- patch grouped at strong `(0.90,0.10)` and moderate `(0.75,0.25)` marker strengths;
- patch personal at both strengths;
- rotation shared;
- rotation grouped;
- rotation personal.

Explanation sample counts `{8,16,32,64}`.

Fresh seeds: **801, 802, 803, 804, 805**.

## Primary metric

Mean JSD to an independent high-sample oracle explanation (`oracle_jsd`), lower is better.

## Approval gate

NC-SACPA is **SMOKE-TEST APPROVED FOR CONTINUATION** only if every condition passes.

### A. Task sanity
All seed/scenario task test accuracies >= 0.90.

### B. Shared-case denoising
Separately for patch-shared and rotation-shared:
- improve over Local by >= 35%; and
- strictly beat oracle-tuned GlobalMedian.

### C. Heterogeneous aggregate superiority
Equal-weight the four heterogeneous families. NC-SACPA must improve mean `oracle_jsd` by:
- >= 30% versus Local;
- >= 10% versus oracle-tuned GlobalMedian;
- >= 10% versus oracle-tuned Cluster.

### D. No family or strength collapse
NC-SACPA must strictly beat oracle-tuned GlobalMedian in each of the four heterogeneous families.

For patch grouped/personal, it must also strictly beat GlobalMedian separately at both strong and moderate marker strengths.

### E. Fresh-seed robustness
On the equal-family heterogeneous average, NC-SACPA must beat Local, GlobalMedian, and Cluster in **5/5 seeds**.

### F. Revision mechanism
On strong grouped-patch heterogeneity, NC-SACPA must improve over the frozen Phase-VII `SACPA-global-count` predecessor by >= 15% in mean `oracle_jsd`.

Across the remaining three heterogeneous families combined, NC-SACPA may be at most 20% worse than SACPA-global-count. This prevents fixing grouped patches by destroying the other cases.

### G. Active federation
The mean peer mass before beta mixing must be >= 1.5 across heterogeneous confirmation cases.

## Failure rule

Any failed gate means the method is not approved. No parameters may be changed using seeds 801--805.
