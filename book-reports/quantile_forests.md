# Quantile Regression Forests
http://jmlr.org/papers/volume7/meinshausen06a/meinshausen06a.pdf

## Introduction
- Assume random forests work in the classic Brieman way (sample split criteria, tree structure, etc)
- Quantile regression is an extension to FR
- It occurs at the leaf nodes and provides a more holistic continuous prediction than the conditional mean

## Algorithm
- When splitting the feature space, save the values of y in each leaf node
- In standard RF, the prediction is the average of all y's in the leaf (ie conditional mean)
- In quantile regression RFs, you instead estimate the __whole conditional distribution__ from the values in the leaf node

## Applications
- Outlier detection: given a point (X_i, y_i), obtain P(y|X_i) from the quantile forest. Then observe P(y=y_i|X_i) according to the estimated distribution!
- Estimating confidence intervals
- Uncertainty estimation

## Outstanding questions
- How do you average estimated distributions from each tree?
