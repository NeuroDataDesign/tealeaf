# Automatic epileptic seizure prediction based on scalp EEG and ECG signals
  
paper link: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7743357
    

## Abstract
* Epilepsy: a neurological disease caused by a neuronal electric activity inbalance in any side of the brain (the epileptic focus), characterized by sudden, recurrent seizures.
* 50% of patients can feel a seizure coming on
* Goal: non-invasive seizure prediction to reduce injuries
* Data: EEG and ECG from 7 focal epilepsy patients (15 minutes before seizure and baseline times)
* Task: Recognize patterns in EEG and ECG before seizures 
* How?: Generate features with Sequential Forward Selection using linear-Bayes and KNN classifiers for cost calculation. Found which features have relevant information.

## Introduction
* Epilepsy affects more than 50 million people worldwide
* Seizures: brief episodes of involuntary movement of all or part of body, sometimes accompanied by loss of consciousness
* Scalp EEG is subject to lots of noise, so intercranial EEG was shown to predict seizures previously.
* ECG was also examined in this study because it is thought that both are affected before a seizure.
* Previous methods struggle with the tradeoff between high sensitivity and low false positive rate as well as invasiveness, noise, and computational costs.
* Features related to seizure forecasting:
    * Linear: autoregressive coefficient, accumulated energy, maximum linear cross correlation
    * Nonlinear: correlation dimension, largest Lypunov exponent, dynamical similarity measure based on zero-crossing intervals, phase synchronization, heart rate variability

## Feature Extraction

* Univariate measures: computed on each EEG channel separately
* Multivariate measures: quantify relationship between between 2+ channels
* Use wavelet transform to analyze properties
* R-wave detection for ECG

## Dimensionality Reduction
* 10-20 international system of electrode placement for EEG recording --> 21 channels
* Solution: remove redundant/irrelevant features

## Methods
* Select dataset, filter sEEG and ECG, compute features from biosignals, outlier removal, feature selection, binary classification (seizure prediction)
* NOTE: They visualize and analyze data with Matlab
* Preprocessed EEG with elliptical, low-pass and high-pass filters
* Preprocessed ECG with wavelets
* Extract features from EEG:
    * segment each channel into non-overlapping 5s windows and compute Discrete Wavelet Transform for each window
* Extract features from ECG using sliding window with one heartbeat overlapping width (3 minutes each)
    * time domain features and frequency domain features
* Feature selection for EEG based on Pearson's Correlation < 0.95 and SFS
* sEEG classified with a linear Bayes classifier and ECG with KNN classifier
* Results: Average accuracy (0.94), sensitivity (0.93), FPR (0.051)
