# Summary of _Sparse Projection Oblique Randomer Forests_

## High-level points

- Decision forests (ensembles of decision trees) work well in many scenarios
- Trees usually recursively split the input space with boundaries parallel to the input axes. Whenever classes are not seperable along any single dimension, decision boundaries become more complicated (see [taxicab geometry](https://en.wikipedia.org/wiki/Taxicab_geometry#/media/File:Manhattan_distance.svg)).
- "Oblique" ensembles address this problem by splitting on linear combinations of features
- SPORF adapts RF by splitting data along linear combinations of a small number of features

Just a quick aside: _oblique_ means neither perpendicular nor parallel, so in this setting, we are looking for splits that are neither parallel nor perpendicular to the coordinate axes

## Methods

### All oblique methods split along different projections of the input data
- A generalized model for oblique ensembles is to think about searching over the space of all possible projection matrices
- If the data matrix $X$ is in R^{n x p} (_n_ subjects and _p_ features), then all projection matrices live in R^{n x d}
- _d_ is the dimensionality of the projected space

### SPORF considers sparse projection matrices
- The potential projection matrices __A__ considered by SPORF are _sparse_
- (_sparse_ means most entries are 0)
- Why sparse matrices?
> Li et al. [8] demonstrates that very sparse random projections, in which a large fraction of entries in A are zero, can maintain high accuracy and significantly speed up the matrix multiplication by a factor of sqrt(p) or more.
- See Section 3.1 for the distribution for a sparse matrix

### A quick aside about MORF
- Manifold Forests (MORF) are a _further_ extension of SPORF
- MORF operates on structured data (ie data where the indices encode important important information)
- MORF uses the structure of the data to create data-minded projection matrices
- The projections considered by MORF are a subset of SPORF

## Needs

- SPORF for regression is not implemented
- Seems like no one knows how to do multivariate regression with RF and its varients

## Outstanding questions

- In SPORF, is each tree in an ensemble trained on the same projection of the data?
- How does SPORF maintain interpretability if splits are obique?
- Is SPORF consistently an improvement over RF?
