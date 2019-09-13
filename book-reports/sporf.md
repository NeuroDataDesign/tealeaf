# Summary of _Sparse Projection Oblique Randomer Forests_

## High-level points

- Decision forests (ensembles of decision trees) work well in many scenarios
- Trees usually recursively split the input space with boundaries parallel to the input axes. Whenever classes are not seperable along any single dimension, decision boundaries become more complicated.
- "Oblique" ensembles address this problem by splitting on linear combinations of features
- SPORF adapts RF by splitting data along linear combinations of a small number of features

Just a quick aside: _oblique_ means neither perpendicular nor parallel, so in this setting, we are looking for splits that are neither parallel nor perpendicular to the coordinate axes

## Methods

## Drawbacks / areas for improvement

- SPORF (and all oblique ensembles) forfeit 

## Needs

- SPORF for regression (uni- and multi-variate) is not implemented
