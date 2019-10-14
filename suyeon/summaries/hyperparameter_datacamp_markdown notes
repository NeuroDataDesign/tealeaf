# Hyperparameter Optimization in Machine Learning Models
- machine learning models are parameterized so that their behavior can be tuned for a given problem


### What is a parameter in a machine-learning learning model?
- model parameter: configuration variable that is internal to the model and whose value can be estimated from the data
    - required by model when making predictions
    - values define the skill of the model on your problem
    - estimated or learned from data
    - often not set manually by the practitioner
    - often saved as part of the learned model
- whether a model has a fixed or variable number of parameters determines whether it may be referred to as "parametric" or "nonparametric"
- examples of model parameters:
    - weights in an artifical neural network
    - support vectors in a support vector machine
    - coefficients in a linear regression or logistic regression


### What is a hyperparameter in a machine-learning learning model?
- model hyperparameter: configuration that is external to the model and whose value cannot be estimated from the data
    - often used in processes to help estimate model parameters
    - often specified by the practitioner
    - can often be set using heuristics
    - often tuned for a given predictive modeling problem
- you cannot know the best value for a model hyperparameter on a given problem
- when a machine learning algorithm is tuned for a specific problem, then essentially you are tuning the hyperparameters of the model to discover the parameters of the model that result in the most skillful predictions
- "If you have to specify a model parameter manually, then it is probably a model hyperparameter"
- examples of model hyperparameters:
    - learning rate for training a neural network
    - C and sigma hyperparameters for support vector machines
    - k in k-nearest neighbors
    

### Importance of the right set of hyperparameter values in a machine learning model
- best way to think about hyperparameters is like the settings of an algorithm that can be adjusted to optimize performance
- ask the machine to perform the exploration of the optimal model architecture for a given model


### Two simple strategies to optimize/tune the hyperparameters
#### 1. Grid search
- methodically builds and evaluates a model for each combination of algorithm parameters specified in a grid
- computationally very expensive: goes through all the intermediate combinations of parameters

#### 2. Random search
- no longer provide a discrete set of values to explore for each hyperparameter; instead, you provide a statistical disttribution for each hyperparameter from which values may be randomly sampled
    - distribution (statistical definition): an arrangement of values of a variable showing their observed or theoretical frequency of occurrence
    - sampling (statistical definition): process of choosing a representative sample from a target population and collecting data from that sample in order to understand something about the population as a whole
- random search concept:
    - define a sampling distribution for each hyperparameter
    - define how many iterations you'd like to build when searching for the optimal model
    - in each iteration, hyperparameter values of the model are set by sampling the defined distributions
- for cases where the hyperparameter being studied has little effect on the resulting model score, grid search results in wasted effort
- random search has much improved exploratory power and can focus on finding the optimal value for the critical hyperparameter
