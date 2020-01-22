# predictive_model

This repository is an implementation of a predictive model. The model is sklearn compatible and has passed sklearn estimator check.
It can be used together with all other functionalities in sklearn like GridSearch, get_params, set_params and others. 

To use this Estimator first clone this repo via 

```
git clone https://github.com/engmubarak48/predictive_model.git
```
The code expects that you have numpy and sklearn installed. 

After cloning, go to the directory of the repo in your computer. From the terminal run the below command to make sure everything is working as expected in your computer. 

```
python HOWTOUSE.py
```
If everything worked correctly you should get an output close to one below. Though it might little bit vary depending on Initialization and version. 

```
 this code was tested in sklearn version '0.21.2' & your sk.version is '0.21.2'
-------The estimator has passed all the checks-------
Train R2_score for randomly generated data:  0.8756649842429656
---- Using Datasets availlable in sklearn like BOSTON dataset for housing price prediction---
Train set R2_score for BOSTON data:  0.737389131707604
----using sklearn packages like GridSearch to search for best parameters---
Train set R2_score for BOSTON data with gridsearch:  0.737389131707604
Best parameters chose by GridSearch:  {'num_iterations': 1000, 'p_norm': 2}
---Using sklearn-openML interface to evaluate the model on other datasets.---
Train set R2_score for BOSTON data fetched from openML:  0.7353366403676359

```
If you get the above output, then you can easily use the package just like any sklearn package. Since, this is a regression model it can only be used on regression data (i.e. univariate & Multivariate regression data).

Use the below script to use the model. 

```
import numpy as np
from Estimator import PnormRegressor
from sklearn.datasets import fetch_openml

X,y = fetch_openml(name='boston', return_X_y=True)

# Normalizing the data
X = (X - X.mean(axis=0))/X.std(axis=0)

regressor  = PnormRegressor()

# Train on train data
regressor.fit(X, y)
# Predict on train data
predictions = regressor.predict(X) 
```


The model has three hyperparameters; the learning_rate, num_iterations and p_norm. You can either initialize these parameters in PnormRegressor() initialization or simply use sklearn's GridSearch to find the best parameters among the ones you provide. 

In my analysis, I realized if "p_norm" is greater than 2, it leads the loss to go infinity. For now, I would highly recommend using value between [1,3).

For further details of how to use this repo, please read through the "HOWTOUSE.py" file or "HOW_TO_USE" jupyter notebook file.

If you have any further questions, please don't hesitate to ask. 

