# Summary: BOHB Robust and Efficient Hyperparameter Optimization at Scale
URL: http://proceedings.mlr.press/v80/falkner18a/falkner18a.pdf
### 0. Desiderata
* Strong anytime performace
* Strong final performance
* Effective use of parallel resource
* Scalability
* Robustness & Flexibility

### 1. HPO Problem
* f: X --> R where hyperparameters are x in set X
* x* = argmin f(x)

### 2. Bayesian optimization
* Gaussian is most commonly used probabilitistic model in BO
* Don't fulfill scalability and flexibility
* Uses acquisition function a: X --> R that trades off exporation and exploitation
* Three steps:
  * select point maximizes the aquisition function 
  * evaluate objective function
  * augment data and refit model
* Tree Parzen Estimator
  * Uses kernel densitiy estimator to model densities
  
### 3. Hyperband
* Allocates resources to set of random configurations, uses successive halving to stop poor configs
* Weaknesses: only samples configurations randomly, does not learn from previously sampled configurations
* Quality of model increases with b (budget)
* Repeatedly call SuccessiveHalving to identify best out of n randomly sampled configs by evluating bmax
* Because it uses randomly drawn configs, with large budgets, advantage over random search diminishes

### 4. BOHB
* BO handled by varint of TPE with product kernel
* HB determins how many configs to evaluate with budget
  * Replaces random selection with model-based search (model trained based on configs evaluated so far)
* 20x speedup over random search and standard BO
* Applications
  * Optimization of Bayesian neural network
  * Optimization of reinforcement learning agest
  * Optimization of SVM
* Limits:
  * Need to define meaningful budgets.  
  * If evaluation on small budgets is too noisy, BOHB is k times slower than BO
 
  
  
  
