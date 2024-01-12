# Some objective of ML 
* Inpterpretation,
    -  model is train with the goal of finding insights from the data
    - $y_p = f(\omega, x)$, interpretation uses $\omega$ to give us insight into the system (feature importance)
* Prediction
    - Focus is on making the best prediction. Thus performance metrics to measure the quality of the model's prediction
    - performance metrics is a quantitative measure of the closeness between y_p and our predicted values and our actual values. Without focusing on interpretation we risk having a black-box model

interpretation can actually provide insights into improving our prediction and vice versa. Not all ML models will allow for a balance beteen interpretation and prediction. 

# Supervised Learning
Two types: classifiation (predicts a categorical output) and regression (predicts a numeric outcome or variable is continous)
# Linear regression
A modeling best practice for linear regression is:

* Use cost function to fit the linear regression model
* Develop multiple models
* Compare the results and choose the one that fits your data and whether you are using your model for prediction or interpretation. 

Three common measures of error for linear regressions are:
$$sum of square error (SSE) = \sum_{1}^{m} (y - y_{predicted})^2$$
$$total sum of square (TSS = \sum_{1}^{m} (y_{average} - y_{predicted})^2$$
$$coefficient of determination (R^2) = 1 - \frac{SSE}{TSS}$$

Training error is the loss (one example) or cost (all examples) function. The cost function needs to be minimize

## Polynomial Regression.
Polynomial features can be use to capture non-linear effects

What if your data is more complex than a straight line? Surprisingly, you can use a linear model to fit nonlinear data. A simple way to do this is to add powers of each feature as new features, then train a linear model on this extended set of features. This technique is called Polynomial Regression.

There are two factors when determining model performance: overfitting and underfitting. Overfitting is when the model is too complex and does well on the training data but not on the test data. Underfitting is when the model is too simple and performs poorly on the training and testing data sets.

Overfitting is simple to deal with, using methods like regularization. To deal with underfitting, we can build a more complex model using methods like polynomial regression.

## Cross validation
This is calculating the error on multiple test splits making the performance metrics more statistically significant. This is splitting the data into several training and validation sets so that there is no overlap. The model is trained on the training set and tested on the validation set. The average of the error metrics on the validation set gives the overall performance of the model. By doing this, we prevent a model from performing well out of sheer luck.

## Hyperparameter tuning
These are parameters we have to tweek ourselves

## Bias and Variance
Bias is a tendency to miss.
Variance is a tendency to be inconsistent

There are 3 main sources of model errors:
1. being wrong - bias. Cause by missing information or overly-simplistic assumptions. Miss real patterns (underfit)
2. being unstable - variance. Characterize by sensitivity of output to small changes in the input data. Due to overly complex or poorly-fit model
3. Unavoidable randomness - irreducible error

bias-variance trade-off - model adjustments that decrease bias often increase variance and vice versa. This is similar to complexity trade-off. Finding the best model means finding the right level of complexity.

## Regularization
Lasso L1 regularization is slower to converge than Ridge L2 regression. Lasso gives a better interpretability but less computationally efficient. Elastic net, an alternative hybrid approach which combines penalties from both Ridge and Lasso regression

## Recursive Feature Elimination (RFE)
it repeatedly applies the model, measures feature importance and recursively removes less important features