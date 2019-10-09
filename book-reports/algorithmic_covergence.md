# Measuring the Algorithmic Convergence of Randomized Ensembles: The Regression Setting
Paper: https://export.arxiv.org/pdf/1908.01251

## Summary

*How many trees is enough?* 

The perfect ensemble would have infinitely many trees. In practical settings, we want a rigorous guarantee the that a given finite ensemble will perform nearly as well as an infinite ensemble.

This paper provides such a guarantee.


## Review

Recall that RF trains each tree on a *resampled* set of the training data. 
Resampled means the training data is randomly sampled with replacement.


## Definition of algorithmic convergence

Define MSE_t to be the expected mean-squared error of an ensemble of size t. Then MSE_infinity is the limit of MSE_t as t goes to infinity.

We are interested in quantifying the Pr[(MSE_t - MSE_infinity) < q]. This is the same as the differential quantiles of MSE_t.


## Main result

For a fixed dataset D, mse_t − mse_infinity ≤ q_{1−α}(t)

q_{1−α}(t) is estimated using bootstrap methods
