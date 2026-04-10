### Model Evaluation and

### Inference

```
IN5148: Statistics and Data Science with Applications in
Engineering
```
```
Alan R. Vazquez
Department of Industrial Engineering
```

##### Agenda

1. Residual analysis

###### 2. Inference about individual β ’s using t-tests


##### Load the libraries

Let’s import the required libraries into Python together with
other relevant libraries.

We will introduce the new library **statsmodels.api** later.

```
1 import pandas as pd
2 import seaborn as sns
3 import matplotlib.pyplot as plt
4 from sklearn.linear_model import LinearRegression
5 import statsmodels.api as sm
```

### Inference about

### individual ’s using t-

### tests

## β


##### Linear Regression Model

Recall that the linear regression model is a common candidate
function for predicting a response:

```
Where is the index of the training data.
is the prediction of the actual value of the response
associated with values of predictors denoted by
.
```
```
The values , , ..., are the coefficients of the model.
```
###### Y ^ i = f ^( X i ) = β ^ 0 + β ^ 1 Xi 1 + ⋯ + β ^ pXip.

###### i = 1 , ..., nt nt

###### Y ^ i Yi

###### p

###### X i = ( Xi 1 , ..., Xip )

###### β ^ 0 β ^ 1 β ^ p


##### Example 1

Let’s consider the “auto.xlsx” dataset.

Let’s assume that we want to train the following model.

where is the predicted mpg, is the weight, and is
the acceleration of the -th car,.

Here, we will use the entire dataset for training and ignore -
fold cross-validation to illustrate the concepts.

```
1 auto_data = pd.read_excel('auto.xlsx')
```
###### Y ^ i = β ^ 0 + β ^ 1 Xi 1 + β ^ 2 Xi 2 ,

###### Y ^ i Xi 1 Xi 2

###### i i = 1 , ..., 392

###### K


##### Research question

```
Do the weight and acceleration have a significant
association with the mpg of a car?
```

##### The two cultures of statistical models

```
Inference : develop a model that fits the data well. Then
make inferences about the data-generating process based
on the structure of such model.
Prediction : Silent about the underlying mechanism
generating the data and allow for many predictive
algorithms, which only care about accuracy of predictions.
```
They overlap very often.


The least squares estimates are subject to
uncertainty, since they are calculated based on a random
sample of data. That is, an instance of training data.

Therefore, assessing the amount of the uncertainty in these
estimates is important. To this end, we use hypothesis tests on
individual coefficients.

###### β ^ 0 , β ^ 1 , ..., β ^ p


##### Hypothesis test

```
A statistical hypothesis is a statement about the
coefficients of a model.
```
In our case, we are interested in testing:

###### H 0 : βj = 0 v.s. H 1 : βj ≠ 0 (Two-tailed Test)

###### βj is the true (unknown) coefficient of the predictor.

```
Rejecting (in favor of ) implies that the predictor has a
significant association with the response.
```
###### H 0 H 1

```
Not rejecting implies that the predictor does not have a
significant association with the response.
```
###### H 0


```
v.s. (Two-tailed Test)
```
Testing this hypothesis consists of the following steps:

###### H 0 : βj = 0 H 1 : βj ≠ 0

1. Take a random sample ( **your training data** ).
2. Compute the appropriate test statistic.
3. Reject or fail to reject the null hypothesis based on a
    computed p-value.


##### Step 1. Random sample

The random sample is the training data we use to train or fit
the model.

As mentioned before, we will use the entire dataset for
training.

```
1 # Dataset with predictors.
2 auto_X_train = auto_data[['weight', 'acceleration']]
3
4 # Dataset with the response only.
5 auto_Y_train = auto_data['mpg']
```

##### Step 2. Test statistic

The test statistic is

```
is the least squares estimate of the true coefficient.
```
```
is the standard error of the estimate calculated
using Python.
```
###### t 0 =

###### β ^ j

###### √ v ^ jj

###### β ^ j βj

###### √ v ^ jj β ^ j


Recall that we obtain the least squares estimates ( ) in
Python using:

Later, we will see how to obtain the standard error of the

estimate.

###### β ^ j

```
1 # 1. Create the linear regression model
2 LRmodel = LinearRegression()
3
4 # 2. Fit the model.
5 LRmodel.fit(auto_X_train, auto_Y_train)
6
7 # 3. Show estimated coefficients.
8 print("Intercept:", LRmodel.intercept_)
9 print("Coefficients:", LRmodel.coef_)
Intercept: 41.
Coefficients: [-0.0072931 0.2616504]
```
###### β ^ j


```
We like this statistic because it follows a
well-known distribution.
If the null hypothesis ( ) is
true, the statistic follows a
distribution with degrees of
freedom
Remember that is the number of
observations and the number of
predictors.
```
###### t 0 =

###### β ^ j

###### √^ vjj

###### H 0 : βj = 0

###### t 0 t

###### n − p − 1

###### n

###### p


##### t distribution

This distribution is also known as the
student’s t-distribution.

It was invented by when he
worked at the Guinness Brewery in Ireland.

It has one parameter which generally
equals a number of degrees of freedom.

The parameter controls the shape of the
distribution.

```
William Gosset
```
###### ν

###### ν

```
William Gosset
```

The t-distribution resembles a standard normal distribution

###### when ν goes to infinity.


##### Step 3. Calculate the p-value

The **p-value** is the probability that the test statistic will get a
value that is at least as extreme as the observed value when
the null hypothesis ( ) is true.

###### t

###### t 0

###### H 0


For v.s. , the p-value is calculated
using both tails of the distribution. It is the blue area under
the curve below.

We use the two tails because includes the possibility that
the value of is positive or negative.

###### H 0 : βj = 0 H 1 : βj ≠ 0

###### t

###### H 1

###### βj


##### The p-value is not the probability that

##### is true

Since the p-value is a probability, and since small p-values
indicate that is unlikely to be true, it is tempting to think
that the p-value represents the probability that is true.

###### This is not the case!

Remember that the **p-value** is the probability that the test
statistic will get a value that is at least as extreme as the
observed value **when** is true.

#### H 0

###### H 0

###### H 0

###### t

###### t 0 H 0


##### How small must the p-value be to

#### reject H 0?


**Answer** : Very small!

But, how small?

**Answer** : Very small!

But, tell me, how small?

**Answer** : Okay, it must be less than
a value called which is usually
0.1, 0.05, or 0.01.

###### α

Thank you!

```
Sir Ronald Fisher
```

##### Decision

For a significance level of :

```
If the p-value is smaller than , we reject in
favor of.
If the p-value is larger than , we do not reject.
```
No scientific basis for this advice. In practice, report the p-
value and explore the data using descriptive statistics.

###### α = 0. 05

###### α H 0 : βj = 0

###### H 1 : βj ≠ 0

###### α H 0 : βj = 0



##### t-tests in Python

Unfortunately, there is no native function in **scikit-learn** to
compute all elements of t-tests for a linear regression model.

The easiest and safest way to do this is to train a model from
scratch using the **statsmodels** library in Python.

🙁


##### statsmodels library

```
statsmodels is a powerful Python library for statistical
modeling, data analysis, and hypothesis testing.
It provides classes and functions for estimating statistical
models.
It is built on top of libraries such as NumPy , SciPy , and
pandas
https://www.statsmodels.org/stable/index.html
```
# statsmodels

```
Tecnologico de Monterrey
```

An issue with **statsmodels** is that we must add the intercept to
our predictor matrix using the add_constant() function.

We fit or train a linear regression model using the .OLS() and
.fit() functions.

```
1 auto_X_train_with_intercept = sm.add_constant(auto_X_train)
```
```
1 # Create linear regression object in statsmodel
2 regr = sm.OLS(auto_Y_train, auto_X_train_with_intercept)
3
4 # Train the model using the training sets
5 linear_model = regr.fit()
```

We access the information on the t-tests using the .summary()
function.
1 model_summary = linear_model.summary()
2 print(model_summary)
OLS Regression Results
==============================================================================
Dep. Variable: mpg R-squared: 0.700
Model: OLS Adj. R-squared: 0.698
Method: Least Squares F-statistic: 453.2
Date: Thu, 19 Mar 2026 Prob (F-statistic): 2.43e-102
Time: 18:09:57 Log-Likelihood: -1125.4
No. Observations: 392 AIC: 2257.
Df Residuals: 389 BIC: 2269.
Df Model: 2
Covariance Type: nonrobust
===============================================================================
coef std err t P>|t| [0.025
0.975]


##### Table of t-tests

```
coef std
err
```
```
t P>
```
```
const 41.0953 1.868 21.999 0.000 37.423 44.768
weight -0.0073 0.000 -25.966 0.000 -0.008 -0.007
acceleration 0.2617 0.086 3.026 0.003 0.092 0.432
```
The column **std err** contains the values of the estimated

standard error ( ) of the estimates. They represent the

variability in the estimates.

###### | t | [ 0. 025 0. 975

###### √ v ^ jj β ^ j


```
coef std
err
```
```
t P>
```
```
const 41.0953 1.868 21.999 0.000 37.423 44.768
weight -0.0073 0.000 -25.966 0.000 -0.008 -0.007
acceleration 0.2617 0.086 3.026 0.003 0.092 0.432
```
The column **t** contains the values of the observed statistic
for the hypothesis tests.

###### | t | [ 0. 025 0. 975

###### t 0


```
coef std
err
```
```
t P>
```
```
const 41.0953 1.868 21.999 0.000 37.423 44.768
weight -0.0073 0.000 -25.966 0.000 -0.008 -0.007
acceleration 0.2617 0.086 3.026 0.003 0.092 0.432
```
The column **P>|t|** contains the p-values for the hypothesis
tests.

Since the p-value is smaller than , we reject for
acceleration and weight. Therefore, **weight and acceleration
have a significant association with the mpg of a car**.

###### | t | [ 0. 025 0. 975

###### α = 0. 05 H 0


```
coef std
err
```
```
t P>
```
```
const 41.0953 1.868 21.999 0.000 37.423 44.768
weight -0.0073 0.000 -25.966 0.000 -0.008 -0.007
acceleration 0.2617 0.086 3.026 0.003 0.092 0.432
```
The columns and show the limits of a 95%
confidence interval on each true coefficient. This interval
contains the true parameter value with a confidence of 95%.

###### | t | [ 0. 025 0. 975

###### [ 0. 025 0. 975 ]

###### βj


### Residual analysis


##### Residuals

The errors of the estimated model are called **residuals**

```
,
```
In linear regression, we can validate the adequacy of the model
by checking whether the residuals satisfy three
(or four) assumptions.

###### ε ^ i = Yi − Y ^ i i = 1 ,... , n.

###### ^ ε 1 , ^ ε 2 , ..., ε ^ n


##### Assumptions

###### The residuals ^ εi ’s must then satisfy the following assumptions:

1. On average, they are close to zero for any value of the

###### predictors Xj.

2. For any value of the predictor , the dispersion or variance
    is roughly the same.

###### Xj

###### 3. The εi ’s are independent from each other.

###### 4. The ^ εi ’s follow normal distribution.

To evaluate the assumption of a linear regression model, we
use a **Residual analysis**.


##### Residual Analysis

To check the validity of these assumptions, we will follow a
graphical approach. Specifically, we will construct three
informative plots of the residuals.

1. **Residuals vs Fitted Values** Plot. To assess the structure of
    the model and check for constant variance
2. **Residuals Vs Time** Plot. To check independence.
3. **Normal Quantile-Quantile** Plot. To assess if the residuals
    follow a normal distribution


##### Example 2

```
This example is inspired by Foster, Stine and Waterman
(1997, pages 191–199).
The data are in the form of the time taken (in minutes) for a
production run, , and the number of items produced, ,
for 20 randomly selected orders as supervised by a manager.
We wish to develop an equation to model the relationship
between the run time ( ) and the run size ( ).
```
###### Y X

###### Y X


##### Dataset

The dataset is in the file “production.xlsx”.

```
Case RunTime RunSize
0 1 195 175
1 2 215 189
2 3 243 344
3 4 162 88
4 5 185 114
```
```
1 production_data = pd.read_excel('production.xlsx')
2 production_data.head()
```

##### Linear regression model

where is the run size and is the predicted run time for the
-th observation,.

###### Y ^ i = β ^ 0 + β ^ 1 Xi

###### Xi Y ^ i

###### i i = 1 , ..., 20

We fit the model in **scikit-learn** using the entire data for
training.

```
1 # Defining the predictor (X) and the response variable (Y).
2 prod_Y_train = production_data['RunTime']
3 prod_X_train = production_data[['RunSize']]
4
5 # Fitting the simple linear regression model.
6 LRmodelProd = LinearRegression()
7 LRmodelProd.fit(prod_X_train, prod_Y_train)
```

##### Estimated model

The estimated coefficients are

So, the fitted model is

where is the predicted or fitted value of run time for the -th
observation,.

The **residuals** of this estimated model are.

```
1 print("Intercept:", LRmodel.intercept_)
2 print("Coefficients:", LRmodel.coef_)
```
###### Y ^ i = β ^ 0 + β ^ 1 Xi = 149. 75 + 0. 26 Xi ,

###### Y ^ i i

###### i = 1 , ..., 20

###### ^ εi = Yi − Y ^ i


##### Calculation of fitted values and

##### residuals

Recall that we can calculate the predicted values and residuals
as follows.

The extra 0*prod_Y_train above allows us to have the
correct Python for the predictions from **scikit-learn** functions.

```
1 # Make predictions using the the model
2 prod_Y_pred = LRmodelProd.predict(prod_X_train) + 0 *prod_Y_train
3
4 # Calculate residuals.
5 residuals = prod_Y_train - prod_Y_pred
```

### Residuals vs Fitted

### Values Plot


##### Intuition behind ...

the Residuals versus Fitted Values plot.


##### Residuals vs Fitted Values

```
Code Code
```

##### Residuals vs Fitted Values

```
If there is a trend, the
model is misspecified.
A “funnel” shape
indicates that the
assumption of
constant variance is
not met.
```

Examples of plots that do not support the conclusion of
constant variance.


Another example.

The phenomenon of non-constant variance is called
**heteroscedasticity**.


### Residuals vs Time Plot


##### Residuals vs Time Plot

```
By “time,” we mean that time the observation was taken or
the order in which it was taken. The plot should not show
any structure or pattern in the residuals.
Dependence on time is a common source of lack of
independence, but other plots might also detect lack of
independence.
Ideally, we plot the residuals versus each variable of interest
we could think of, either included or excluded in the model.
Assessing the assumption of independence is hard in
practice.
```

Code


Examples of plots
that do and do
not support the
independence
assumption.


### Normal Quantile-

### Quantile Plot


##### Checking for normality

This assumption is generally checked by looking at the
distribution of the residuals.

Two plots:

```
Histogram.
Normal Quantile-Quantile Plot (also called normal
probability plot).
```

##### Histogram

Ideally, the histogram resembles a normal distribution around

0. If the number of observations is small, the histogram may
not be an effective visualization.

```
Code
```

##### Normal Quantile-Quantile (QQ) Plot

A normal QQ plot is helpful for deciding whether a sample was
drawn from a distribution that is approximately normal.

1. First, let be the residuals ranked in an
    increasing order, where is the minimum and is the
    maximum. These points define the **sample** percentiles (or
    quantiles) of the distribution of the residuals.

###### ^ ε [ 1 ],^ ε [ 2 ], ..., ^ ε [ n ]

###### ^ ε [ 1 ] ^ ε [ n ]

2. Next, calculate the **theoretical** percentiles of a (standard)
    Normal distribution using Python.


The normal QQ plot displays the (sample) percentiles of the
residuals versus the percentiles of a normal distribution.

If these percentiles agree with each other, then they would
approximate a straight line.

The straight line is usually determined visually, with
emphasis on the central values rather than the extremes.

For a nice explanation, see this YouTube video.


##### QQ plot in Python

To construct a QQ plot, we use the function qqplot() from the
**statsmodels** library.
1 # QQ plot to assess normality of residuals
2 plt.figure(figsize=( 5 , 3 ))
3 sm.qqplot(residuals, line = 'q')
4 plt.title('QQ Plot of Residuals')
5 plt.show()


Substantial departures from a straight line indicate that the
distribution is not normal.
<Figure size 672x384 with 0 Axes>


This plot suggests that the residuals are consistent with a
Normal curve.
<Figure size 672x384 with 0 Axes>


##### Comments

These data are truly Normally distributed. But note that we
still see deviations. These are entirely due to chance.

When is relatively small, you tend to see deviations,
particularly in the tails.

###### n


**Normal probability plots for data sets following various
distributions.** 100 observations in each data set.


##### Consequences of faulty assumptions

1. If the model structure is incorrect, the estimated coefficients

###### β ^ j will be incorrect and the predictions Y ^ i will be inaccurate.

2. If the residuals do not follow a normal distribution, we have
    two cases:

```
If sample size is large, we still get accurate p-values for the t-
tests for the coefficients thanks to the Central Limit Theorem.
Otherwiese, the t-tests and p-values are unreliable.
```

3. If the residuals do not have constant variance, then the
    linear model is incorrect and everything falls apart!
4. If the residuals are dependent, then the linear model is
    incorrect and everything falls apart!

There are methods to correct the faulty assumptions, but we
will not discus them here.


### Return to main page


