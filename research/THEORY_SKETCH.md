# Theory sketch: UCPA as heteroskedastic product-kernel smoothing

Let a client's true class-conditional explanation vector be `mu_i`, and suppose its transmitted estimate is `E_i = mu_i + eps_i`, with coordinatewise sampling variance approximately `V_i`. UCPA is a Nadaraya-Watson-style peer smoother with a product kernel:

- a semantic kernel on the full explanation distribution, `K_s(i,k)=exp(-JSD(E_i,E_k)/h)`;
- a heteroskedastic coordinate kernel, `K_u(i,k,j)=exp(-0.5 zscore(i,k,j)^2)`.

The self term prevents complete replacement of a client's estimate. Informally, if `mu_i=mu_k`, increased sample size should make the compatibility statistic concentrate near 1 and pooling should reduce variance. If `|mu_ij-mu_kj|` stays separated while `V_i+V_k -> 0`, coordinate compatibility goes to zero, so asymptotically UCPA should stop averaging that genuinely different coordinate. Whole-explanation similarity provides an additional guard when many coordinates shift coherently.

This sketch motivates the two ablations and unequal-sample experiment. It is **not a theorem**. If full results support the mechanism, a paper-quality analysis should state explicit noise assumptions, derive bias/variance, and characterize how sparse top-k masking alters consistency.
