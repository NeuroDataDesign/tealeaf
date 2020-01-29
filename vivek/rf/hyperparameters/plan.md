# Sprint 1 (Suyeon and Celina)

## Goal
We want to recreate this figure from the SPORF paper, with three important modifications.

**Make sure you understand the meaning of `rank` here!**

![table][figure6]


## Modifications

### Task 1
*Compare [RandomForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) 
to [ExtraTrees](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html).*

To accomplish this (in the two hyperparameter case), we make one table for each algorithm and put them side-by-side.

### Task 2
**Use the Open-ML data sets instead of the UCI data sets.**

We're going to spend some time and figure out exactly why this script cannot run on MARCC. 
In theory, there shouldn't be any issue, but we'll get to the bottom of it!

### Task 3
**Figure out the hyperparameters we're using.**

For each pair of hyperparameters, we make another table. Therefore, for simplicity's sake, 
it'd be better to minimize the number of hyperparameters we optimize over (this will also help the program
run faster on MARCC!).

[figure6]: ./figure.png
