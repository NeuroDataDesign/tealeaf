# Tour of the SPORF Repository: MORF

## Ran the Following Demos
* [Circle Experiment](https://nbviewer.jupyter.org/github/NeuroDataDesign/tealeaf/blob/master/morgan_dods/MORF_demos/Circle_experiment_using_S-RerF.ipynb)
* [Impulse Classification](https://nbviewer.jupyter.org/github/NeuroDataDesign/tealeaf/blob/master/morgan_dods/MORF_demos/Impulse_Classification_using_S-RerF.ipynb)
* [MNIST Classification](https://nbviewer.jupyter.org/github/NeuroDataDesign/tealeaf/blob/master/morgan_dods/MORF_demos/MNIST_classification_using_structured_RerF.ipynb)
  * There was a small error in this demo. clf.fit() threw an error because the labels were strings instead of integers. I fixed this in my version.
* [Digits Classification](https://nbviewer.jupyter.org/github/NeuroDataDesign/tealeaf/blob/master/morgan_dods/MORF_demos/plot_digits_classification.ipynb)
* [hvbar demo](https://nbviewer.jupyter.org/github/NeuroDataDesign/tealeaf/blob/master/morgan_dods/MORF_demos/sRerF-hvbar-demo.ipynb)

## Important Files Related to MORF
* Python/src/packedForest.cpp
  * This file seems to contain a c++ base for building a forest. 
* Python/rerf/rerfClassifier.py (lines 233 - 280)
  * import pyfp (see .cpp file above)
  * self.forest_ = pyfp.fpForest() (happens in all forests)
  * elif self.projection_matrix == "S-RerF" (This block seems important for distinguishing s-RerF from RerF)
* packedForest/src/ (seems to contain lots of c++ code that may relate to MORF)
  * packedForest/src/forestTypes/binnedTree/processingNodeBin.h (lines 160 - 176)
* [Relevant commit](https://github.com/neurodata/SPORF/commit/ef9ade7a28a14a8c7d6db413874bcf2628084726#diff-2a9622f4ae137df85699310ebd93744b)
