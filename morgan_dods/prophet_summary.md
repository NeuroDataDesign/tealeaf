# Forecasting at Scale
paper link: [https://peerj.com/preprints/3190/](https://)
## Main features of Prophet:
* Modular regression model
* Interpretable parameters
* Main goals of the package: 
    * Enable a large number of people to make forecasts regardless of amount of training in time-series methods (Accessible) 
    * Enable a large variety of forecasting problems to be solved (with unique features) (Generalizable)
    * Create an efficient way to evaluate and compare forecasts and detect when they are performing poorly
* Prophet does NOT explore computational scale

## Motivation:
* Automatic forecasting techniques can be difficult to tune and not flexible enough to incorporate useful assumptions and heuristics
* Many people who are experts in their field are not also experts at time-series forecasting

## References to Other Time-Series Forecasting Models
* Random walk models
* Exponential smoothing models
* ARIMA model
* Generalized additive models
* TBATS models

## Prophet Method and Results
* Prophet uses a curve-fitting approach rather than focusing on structure of data
* The algorithm seems to be fairly resistant to overfitting, yet produces fairly complex predictions
* The paper talks about automatically evaluating forecasts to flag potentially problematic predictions.
