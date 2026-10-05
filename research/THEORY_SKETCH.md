# Theory sketch: NC-SACPA as self-tuned sparse peer regression

Let client `i` have a latent class-conditional explanation distribution `mu_i` and a sparse estimate `E_i` whose reported support is `M_i`.

NC-SACPA can be viewed as a Nadaraya-Watson-style peer smoother on explanation space, but with two additions motivated by failures of earlier candidates:

1. **Self-tuned metric scale.** The bandwidth for client `i` is its median distance to other clients, and pair bandwidth is the geometric mean of two local scales. Thus multiplying all comparable distances by a common factor leaves the kernel ratios unchanged. This directly targets UCPA v1's fixed-bandwidth scale collapse.
2. **Local support identifiability.** A coordinate not observed in the target is not treated as a true zero. It may enter the peer prior only when multiple *nearby* clients report it. This is a local support-consensus assumption, not a claim that absence from top-k means irrelevance.

Under a stylized model with latent client groups separated in explanation distribution and sparse support observations with independent reporting noise, one would expect:

- within-group self-tuned weights to dominate cross-group weights when the distance gap is sufficiently large;
- false support import probability to fall with the number of required local corroborators;
- true shared support recovery to increase with neighborhood size and artifact sample count;
- a bias/variance trade-off controlled by `beta` and the collaboration graph.

These are mechanistic hypotheses, **not proved theorems**. If the full preregistered study passes, the next theory phase should formalize conditions for (a) scale invariance, (b) support-import error, and (c) excess explanation-estimation risk relative to local and global pooling. Recent personalized-FL work on adaptive collaboration and collaboration geometry is relevant mathematical prior art; any theorem must be positioned against it rather than presented as a generic novelty of similarity weighting.
