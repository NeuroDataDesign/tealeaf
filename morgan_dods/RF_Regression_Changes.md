* Must evaluate forest with mean squared error instead of accuracy
* Select best split based off RSS instead of gini
terminal node value should be average of outcomes instead of max appearance
* Make sure to call new split criteria and terminal node function when making splits
* When building tree, be sure to call correct splitting function
* Bagging predict should predict average prediction across trees (or should this be max)
* When creating random forest, be sure to make the correct trees

# Links
* https://github.com/neurodata/SPORF/blob/77d4ba13f47923c965e496714109dd2da73f9db8/packedForest/src/forestTypes/basicForests/rerf/splitRerF.h#L68-L107
* https://github.com/neurodata/SPORF/blob/77d4ba13f47923c965e496714109dd2da73f9db8/packedForest/src/forestTypes/basicForests/rerf/unprocessedRerFNode.h
* https://github.com/neurodata/SPORF/blob/77d4ba13f47923c965e496714109dd2da73f9db8/packedForest/src/forestTypes/binnedTree/binStruct.h
* https://github.com/neurodata/SPORF/blob/77d4ba13f47923c965e496714109dd2da73f9db8/packedForest/src/forestTypes/binnedTree/binnedBase.h
* https://github.com/neurodata/SPORF/blob/77d4ba13f47923c965e496714109dd2da73f9db8/packedForest/src/forestTypes/binnedTree/inNodeClassTotals.h
* https://github.com/neurodata/SPORF/blob/77d4ba13f47923c965e496714109dd2da73f9db8/packedForest/src/forestTypes/basicForests/rerf/rerfTree.h
