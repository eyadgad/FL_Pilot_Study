# RPGA smoke confirmation

RPGA was frozen before seeds 1201–1205. It passed all four preregistered smoke gates across client-specific rotation and sparse-patch families.

Key confirmation results:
- deletion AUC, insertion AUC, and local top-k support were exactly invariant in all 10 seed-family runs;
- pairwise EDI fell from 0.12414 to 0.10448 on rotation (15.8%) and from 0.10588 to 0.09044 on patch (14.6%);
- RPGA won EDI in 5/5 rotation seeds and 4/5 patch seeds;
- pooled oracle JSD was 0.12056 vs 0.12069 Local and 0.13093 xFedAlign(beta=.2);
- 9/10 seed-family pairs met the preregistered Pareto condition.

The saved gate report and raw seed JSON files are the source of truth.
