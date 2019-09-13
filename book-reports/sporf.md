# Summary of _Sparse Projection Oblique Randomer Forests_

## High-level points

- Decision forests (ensembles of decision trees) work well in many scenarios
- Trees usually recursively split the input space with boundaries parallel to the input axes. Whenever classes are not seperable along any single dimension, decision boundaries become more complicated.
- "Oblique" ensembles address this problem by splitting on linear combinations of features
- SPORF adapts RF by splitting data along linear combinations of a small number of features

Just a quick aside: _oblique_ means neither perpendicular nor parallel, so in this setting, we are looking for splits that are neither parallel nor perpendicular to the coordinate axes

## Methods


## Needs

- SPORF for regression is not implemented
- Seems like no one knows how to do multivariate regression with RF and its varients

## Outstanding questions

- How does SPORF maintain interpretability if splits are obique?
- Is SPORF consistently an improvement over RF?
