# Summary of _Decision Forests for Classification, Regression, Density Estimation, Manifold Learning and Semi-Supervised Learning_

(link to paper)[https://www.microsoft.com/en-us/research/publication/decision-forests-for-classification-regression-density-estimation-manifold-learning-and-semi-supervised-learning/]

## Density forests (Section 5)
Similar to quantile regression, estimate posterior distribution F(y|X) from the leaf nodes. This paper recommends estimating the distribution by fitting a multivariate gaussian to each distribution.

This section also talks about what to do if the data is unlabelled. 

## Manifold forests (Section 6)
Problem statement: learn a mapping from R^d to a space of much lower dimensionality that preserves relative geodesic distances. I guess that "relative distance" can be interpreted on a local scale.

## Biggest take-away
Think about all random forest tasks as posterior density estimates!

- Classification: learn a discrete posterior
- Regression: learn a continuous posterior
- Density estimation: it's in the name

This unified model for decision forests helped marry a few disperate concepts in my mind

## Background
- Assume your data comes in the form _(x_i, y_i)_ for each patient
- The posterior distribution is p_{y|x} (the probability of observing a given _y_ from input data _x_)

## Drawbacks

From talking with jovo, hayden, and ronak this week, I've learned the estimation of high-dimension densities is not easy to do

Oftentimes, you can circumvent the hard step of estimating these densities by taking a shortcut directly to regression or classification

So this framework is useful conceptually, but not practically
