# Supervised Learning

# and Linear Regression

```
IN5148: Statistics and Data Science with Applications in
Engineering
```
```
Alan R. Vazquez
Department of Industrial Engineering
```

### Agenda

1. Introduction to Supervised Learning
2. Linear Regression Model
3. K-fold Cross Validation


# Introduction to

# Supervised Learning


### Supervised Learning

Includes algorithms that learn by example. The user provides
the supervised algorithm with a known data set that includes
the corresponding known inputs and outputs. The algorithm
must find a method to determine how to reach those inputs
and outputs.

While the user knows the correct answers to the problem, the
algorithm identifies patterns in the data, learns from
observations, and makes predictions.

The algorithm makes predictions that can be corrected by the
user, and this process continues until the algorithm reaches a
high level of accuracy and performance.



### Supervised learning problems

**Regression Problems**. The response is numerical. For example,
a person’s income, the value of a house, or a patient’s blood
pressure.

**Classification Problems**. The response is categorical and
involves K diff erent categories. For example, the brand of a
product purchased (A, B, C) or whether a person defaults on a
debt (yes or no).

#### The predictors ( X ) can be numerical or categorical.


### Supervised learning problems

**Regression Problems**. The response is numerical. For example,
a person’s income, the value of a house, or a patient’s blood
pressure.

**Classification Problems**. The response is categorical and
involves K diff erent categories. For example, the brand of a
product purchased (A, B, C) or whether a person defaults on a
debt (yes or no).

#### The predictors ( X ) can be numerical or categorical.


### Regression problem

**Goal** : Find the best function of the predictors
that describes the response.

In mathematical terms, we want to establish the following
relationship:

```
Where is a natural (random) error.
```
#### f ( X )

#### X = ( X 1 , ..., Xp ) Y

#### Y = f ( X ) + ε

#### ε


### How to find the shape of?

Using training data.

## f ( X )


### How to find the shape of?

Using training data.

## f ( X )


### How to evaluate the quality of the

### candidate function?

Using validation data.

## f ^( X )


### How to evaluate the quality of the

### candidate function?

Using validation data.

## f ^( X )



### Moreover...

We can use **test data** for
a final evaluation of the
model.

Test data is data
obtained from the
process that generated
the training data.

Test data is independent
of the training data.


# Linear Regression Model


### Linear Regression Model

A common candidate function for predicting a response is the
linear regression model. It has the mathematical form:

```
Where is the index of the training data.
is the prediction of the actual value of the response
associated with values of predictors denoted by
.
```
```
The values , , ..., are the coefficients of the model.
```
#### Y ^ i = f ^( X i ) = β ^ 0 + β ^ 1 Xi 1 + ⋯ + β ^ pXip.

#### i = 1 , ..., nt nt

#### Y ^ i Yi

#### p

#### X i = ( Xi 1 , ..., Xip )

#### β ^ 0 β ^ 1 β ^ p


### Interpretation of coefficients

where the unknown parameter is called the “intercept,” and

```
is the “coefficient” of the j-th predictor.
```
#### Y ^ i = β ^ 0 + β ^ 1 Xi 1 + ⋯ + β ^ pXip ,

#### β ^ 0

#### β ^ j

For the j-th predictor, we have that:

#### β ^ j = 0 implies no dependence.

#### β ^ j > 0 implies positive dependence.

#### β ^ j < 0 implies negative dependence.


**Interpretation** :

```
is the average response when all predictors equal 0.
```
```
is the amount of increase in the average response by a 1
unit increase in the -th predictor , when all other
predictors are fixed to an arbitrary value.
```
#### Y ^ i = β ^ 0 + β ^ 1 Xi 1 + ⋯ + β ^ pXip ,

#### β ^ 0 Xj

#### β ^ j

#### j Xij


### Estimation of Coefficients

The values of , , ..., are obtained from the training data
using method of **least squares**.

This method finds the coefficient values that minimize the

error made by the model when trying to predict the
responses in the training set:

where means residual sum of squares.

#### β ^ 0 β ^ 1 β ^ p

#### f ^( Xi )

#### RSS =

```
nt
```
#### ∑

```
i = 1
```
#### ( Yi − ( β ^ 0 + β ^ 1 Xi 1 + ⋯ + β ^ pXip ))^2

#### RSS


### The idea in two dimensions


### Load the libraries

Before we see an example, let’s import the data science
libraries into Python.

Here, we use specific functions from the **pandas** , **matplotlib** ,
**seaborn** and **sklearn** libraries in Python.

```
1 import numpy as np
2 import pandas as pd
3 from sklearn.model_selection import train_test_split, cross_val_score
4 from sklearn.linear_model import LinearRegression
5 from sklearn.metrics import mean_squared_error, root_mean_squared_error
6 from sklearn.metrics import r2_score
```

### Example 1

We used the dataset called “Advertising.xlsx” in Canvas.

```
TV: Money spent on TV ads for a product (in hundreds of $).
Radio: Money spent on Radio ads for a product (in hundreds
of $).
Newspaper: Money spent on Newspaper ads for a product
(in hundreds of $).
Sales: Sales generated from the product ( in thousands of $).
200 markets
1 # Load the data into Python
2 Ads_data = pd.read_excel('Advertising.xlsx')
```

### Notation

For our linear regression model:

```
is the -th observation for Sales.
is the -th observation for TV.
is the -th observation for Radio.
is the -th observation for Newspaper.
```
#### Yi i

#### Xi 1 i

#### Xi 2 i

#### Xi 3 i


```
TV Radio Newspaper Sales
0 230.1 37.8 69.2 22.1
1 44.5 39.3 45.1 10.4
2 17.2 45.9 69.3 9.3
3 151.5 41.3 58.5 18.5
4 180.8 10.8 58.4 12.9
```
1 Ads_data.head()


Now, let’s set the predictors and response. In the definition of
X_full, the double bracket in [] is important because it
allows us to have a pandas DataFrame as output. This makes it
easier to fit the linear regression model with **scikit-learn**.
1 # Chose the predictor.
2 X_full = Ads_data.filter(['TV', 'Radio', 'Newspaper'])
3
4 # Set the response.
5 Y_full = Ads_data.filter(['Sales'])


### Create training and validation data

To evaluate a model’s performance on unobserved data, we
split the current dataset into a training dataset and a
validation dataset. To do this, we use the scikit-learn
train_test_split() function.

We use 75% of the data for training and the rest for validation.

```
1 X_train, X_valid, Y_train, Y_valid = train_test_split(X_full, Y_full,
2 test_size = 0.25,
3 random_state = 301655
```

### Fit a linear regression model in

### Python

In Python, we use the LinearRegression() and fit()
functions from the **scikit-learn** to fit a linear regression model.

```
1 # 1. Create the linear regression model
2 LRmodel = LinearRegression()
3
4 # 2. Fit the model.
5 LRmodel.fit(X_train, Y_train)
```

The following commands allow you to show the estimated
coefficients of the model.

We can also show the estimated intercept.

The estimated model thus is

```
1 print("Coefficients:", LRmodel.coef_)
Coefficients: [[ 0.04620676 0.18682436 -0.0059342 ]]
```
```
1 print("Intercept:", LRmodel.intercept_)
Intercept: [3.25814952]
```
#### Y ^ i = 3. 258 + 0. 046 Xi 1 + 0. 186 Xi 2 − 0. 005 Xi 3.


### Interpretation

```
The average sales are 3.258 thousands of dollars when the
advertising budgets for TV, Radio, and Newspaper are 0 dollars.
Increasing the advertising TV by 1 hundreds of dollars increases the
average sales by 0.046 thousands of dollars.
Increasing the advertising Radio by 1 hundreds of dollars increases
the average sales by 0.186 thousands of dollars.
Increasing the advertising Newspaper by 1 hundreds of dollars
reduces the average sales by 0.005 thousands of dollars.
```
#### Y ^ i = 3. 258 + 0. 046 Xi 1 + 0. 186 Xi 2 − 0. 005 Xi 3.


### Prediction error

After estimating and validating the linear regression model, we
can check the quality of its predictions on **unobserved** data.
That is, on the data in the validation set.

One metric for this is the **mean prediction error** (MSE ):

```
For , the validation data!
```
The smaller , the better the predictions.

```
v
```
```
MSE v =
```
```
∑ ni = v 1 ( Yi − ( β ^ 0 + β ^ 1 Xi 1 + ⋯+ β ^ pXip ))^2
nv
```
#### nv

#### MSE v


In practice, the square root of the mean prediction error is
used:

The advantage of is that it can be interpreted as:

For example, if , then a prediction of will
have an (average) error rate of.

#### RMSE v = √MSE v.

#### RMSE v

```
The average variability of a model prediction.
```
#### RMSE v = 1 Y ^ = 5

#### ± 1


### In Python

To evaluate the model’s performance, we use the validation
dataset. Specifically, we use the predictor matrix stored in
X_valid.

In Python, we make the prediction using the pre-trained
LRmodel.
1 Y_pred = LRmodel.predict(X_valid)


To evaluate the model, we use the function
mean_squared_error() from **scikit-learn**. Recall that the
responses from the validation dataset are in Y_valid, and the
model predictions are in Y_pred.

To obtain the **root mean squared error (RMSE)** , we use
root_mean_squared_error() instead.

```
1 mse = mean_squared_error(Y_valid, Y_pred) # Mean Squared Error (MSE)
2 print(round(mse, 2 ))
5.09
```
```
1 rmse = root_mean_squared_error(Y_valid, Y_pred)
2 print(round(rmse, 2 ))
2.26
```

### Another Metric:

```
In the context of Data Science, can be interpreted as the
squared correlation between the actual responses and those
predicted by the model.
The higher the correlation, the better the agreement
between the predicted and actual responses.
```
We compute in Python as follows:

## R^2

#### R^2

#### R^2

```
1 rtwo_sc = r2_score(Y_valid, Y_pred) # Rsquared
2 print(round(rtwo_sc, 2 ))
0.78
```

# K-fold Cross Validation


### Limitations of a validation set

```
By using a training data set that is much smaller than our
actual data, the estimated model will be less good
than if we used the full training data. That is, more likely for
predictions to be far from actual values.
```
#### f ^( X )

```
Thus, the validation MSE is likely to be bigger than had we
(a) used the full data set and (b) fit the correct model.
In other words, using less than all data results in a
that is not a good representation of the predictive
performance of.
```
#### MSE v

#### f ^( X )


### -fold Cross-Validation

**Basic Idea:** Divide the training data into equally-sized
divisions or folds.

## K

#### K


### -fold Cross-Validation

**Basic Idea:** Divide the training data into equally-sized
divisions or folds ( here).

## K

#### K

#### K = 5


### -fold Cross-Validation

**Basic Idea:** Divide the training data into equally-sized
divisions or folds ( here).

## K

#### K

#### K = 5

#### Using training, construct f ^(−^1 ).

```
Using validation, calculate
```
#### CV 1 ( f ^(−^1 )) = n^11 ∑ i ∈ F 1 ( Yi − f ^(−^1 )( X i ))

```
Call it the Fold-Based Error Estimate
```

### -fold Cross-Validation

**Basic Idea:** Divide the training data into equally-sized
divisions or folds ( here).

```
Using training, construct.
Using validation, calculate
```
## K

#### K

#### K = 5

#### f ^(−^2 )

#### CV 1 ( f ^(−^2 )) = n^12 ∑ i ∈ F 2 ( Yi − f ^(−^2 )( X i ))


### -fold Cross-Validation

**Basic Idea:** Divide the training data into equally-sized
divisions or folds ( here).

```
Using training, construct.
Using validation, calculate
```
## K

#### K

#### K = 5

#### f ^(−^5 )

#### CV 1 ( f ^(−^5 )) = n^15 ∑ i ∈ F 5 ( Yi − f ^(−^5 )( X i ))


We average these fold-based error estimates to yield an
evaluation metric:

```
Called the -fold cross-validation estimate.
Here, we used but another popular choice is
.
```
#### CV ( f ^) =

#### 1

#### K

```
K
```
#### ∑

```
k = 1
```
#### CVk ( f ^(− k )).

#### K

#### K = 5

#### K = 10


### -fold CV in Python

In Python, we apply -fold cross validation (CV) using the
function cross_val_score() from **scikit-learn**. The argument
cv sets the number of folds to use, and scoring sets the
evaluation metric to compute on the folds.

## K

#### K

```
1 # 1. Define a new linear regression model
2 LRmodel_cv = LinearRegression()
3
4 # 2. Apply 5-fold CV
5 neg_cv_MSEs = cross_val_score(LRmodel_cv, X_full, Y_full, cv = 5 ,
6 scoring = "neg_mean_squared_error")
7
8 # 3. Show scores
9 print(neg_cv_MSEs)
[-3.1365399 -2.42566776 -1.58522508 -5.42615506 -2.79114519]
```

Unfortunately, cross_val_score() outputs negative scores.
We simply turn them into positive by multiplying them by -1 or
adding a - symbol.

After that, we average the values using .mean() to obtain a the
-fold CV estimate.

```
1 cv_MSEs = - neg_cv_MSEs
```
#### 5

```
1 MSE_cv = cv_MSEs.mean()
2 print(round(MSE_cv, 3 ))
3.073
```

Note that we can compute a -fold CV estimate for any
evaluation metric including the. To this end, we set
scoring = "r2".

#### K

#### R^2

```
1 # 1. Define a new linear regression model
2 LRmodel_cv = LinearRegression()
3
4 # 2. Apply 5-fold CV
5 cv_Rsq = cross_val_score(LRmodel_cv, X_full, Y_full, cv = 5 ,
6 scoring = "r2")
7
8 # 3. Compute CV estimate
9 Rsq_cv = cv_Rsq.mean()
10 print(round(Rsq_cv, 3 ))
0.887
```

We can also compute a -fold CV estimate for the root mean
squared error (RMSE).

#### K

```
1 LRmodel_cv = LinearRegression()
2 neg_cv_RMSEs = cross_val_score(LRmodel_cv, X_full, Y_full, cv = 5 ,
3 scoring = "neg_root_mean_squared_error")
4 cv_RMSEs = - neg_cv_RMSEs
5 RMSE_cv = cv_RMSEs.mean()
6 print(round(RMSE_cv, 3 ))
1.718
```

### Final remarks

```
-fold CV provides a robust estimate of prediction error by
averaging performance across multiple data splits.
It improves data efficiency by allowing all observations to be
used for both training and validation.
It is widely used for model comparison and hyperparameter
tuning.
It can be extended to generalized -fold cross-validation
(e.g., stratified, grouped, or time-series folds) to respect data
structure.
```
#### K

#### K


# Return to main page


