# Summary of _Decision Forests for Classification, Regression, Density Estimation, Manifold Learning and Semi-Supervised Learning_

(link to paper)[https://www.microsoft.com/en-us/research/publication/decision-forests-for-classification-regression-density-estimation-manifold-learning-and-semi-supervised-learning/]

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
