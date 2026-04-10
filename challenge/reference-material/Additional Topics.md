## Additional Topics

IN5148: Statistics and Data Science with Applications in
Engineering

```
Alan R. Vazquez
Department of Industrial Engineering
```

### Agenda

1. Linear Models with Categorical Predictors
2. Another Example


### Load the libraries

Let’s import **scikit-learn** into Python together with the other
relevant libraries.

We will not use all the functions from the **scikit-learn** library.
Instead, we will use specific functions from the sub-libraries
**preprocessing** , **model_selection** , and **metrics**.

```
1 import pandas as pd
2 import seaborn as sns
3 import matplotlib.pyplot as plt
4 import statsmodels.api as sm
5 from sklearn.linear_model import LinearRegression
6 from sklearn.preprocessing import StandardScaler
7 from sklearn.model_selection import train_test_split
8 from sklearn.metrics import root_mean_squared_error
```

## Linear Models with

## Categorical Predictors


### Categorical predictors

```
A categorical predictor takes on values that are categories,
say, names or labels.
Their use in regression requires dummy variables, which are
quantitative variables.
When a categorical predictor has more than two levels, a
single dummy variable cannot represent all possible
categories.
In general, a categorical predictor with categories requires
dummy variables.
```
#### k

#### k − 1


### Dummy coding

```
Traditionally, dummy variables are binary variables which
can only take the values 0 and 1.
This approach implies a reference category. Specifically, the
category that results when all dummy variables equal 0.
This coding impacts the interpretation of the model
coefficients:
is the mean response under the reference category.
is the amount of increase in the mean response when
we change from the reference category to another
category.
```
#### β 0

#### βj


### Example 1

The auto data set includes a categorical variable “Origin”
which shows the origin of each car.

In this section, we will consider the auto_data as our **training**
dataset.

```
1 # Load the Excel file into a pandas DataFrame.
2 auto_data = pd.read_excel("auto.xlsx")
3
4 # Set categorical variables.
5 auto_data['origin'] = pd.Categorical(auto_data['origin'])
```

### Dataset

```
mpg origin
0 18.0 American
1 15.0 American
2 18.0 American
3 16.0 American
4 17.0 American
5 15.0 American
```
```
1 (auto_data
2 .filter(['mpg', 'origin'])
3 ).head( 6 )
```

### Dummy variables

Origin has 3 categories: European, American, Japanese.

So, 2 dummy variables are required:

“American” acts as the reference category.

#### d 1 = {^1 if car is European and

#### 0  if car is not European

#### d 2 = {.

#### 1  if car is Japanese

#### 0  if car is not Japanese


The dataset with the dummy variables looks like this:

```
origin
American 0 0
American 0 0
European 1 0
European 1 0
American 0 0
Japanese 0 1
```
#### d 1 d 2

#### ⋮ ⋮ ⋮


### Linear regression model

```
is the i-th predicted response.
is 1 if the i-th observation is from a European car, and 0
otherwise.
is 1 if the i-th observation is from a Japanese car, and 0
otherwise.
```
#### Y ^ i = β ^ 0 + β ^ 1 d 1 i + β ^ 2 d 2 i for i = 1 , ..., n.

#### Y ^ i

#### d 1 i

#### d 2 i


### Model coefficients

#### Y ^ i = β ^ 0 + β ^ 1 d 1 i + β ^ 2 d 2 i

#### β ^ 0 is the mean response (mpg) for American cars.

```
is the amount of increase in the mean response when
changing from an American to a European car.
```
#### β ^ 1

```
is the amount of increase in the mean response when
changing from an American to a Japanese car.
```
#### β ^ 2


Alternatively, we can write the regression model as:

Given this model representation:

#### Y ^ i = β ^ 0 + β ^ 1 d 1 i + β ^ 2 d 2 i =

#### ⎧

# ⎪

#### ⎨

# ⎪

#### ⎩

#### β ^ 0 + β ^ 1 + β ^ i if car is European

#### β ^ 0 + β ^ 2  if car is Japanese

#### β ^ 0 if car is American

```
Tecnologico de Monterrey
```

### In Python

We follow three steps to fit a linear model with a categorical
predictor. First, we compute the dummy variables.

Next, we construct the response column.

```
1 # Create linear regression object
2 dummy_X_train = pd.get_dummies(auto_data['origin'], drop_first = True,
3 dtype = 'int')
```
```
1 # Create response vector
2 Y_train = auto_data['mpg']
```

Finally, we fit the model using **scikit-learn**.
1 # 1. Create the linear regression model
2 LRmodel = LinearRegression()
3
4 # 2. Fit the model.
5 LRmodel.fit(dummy_X_train, Y_train)
6
7 # 3. Show estimated coefficients.
8 print("Intercept:", LRmodel.intercept_)
9 print("Coefficients:", LRmodel.coef_)
Intercept: 20.
Coefficients: [ 7.56947179 10.41716352]


### Analysis of covariance

Models that mix categorical and numerical predictors are
sometimes referred to as analysis of covariance (ANCOVA)
models.

**Example** (cont): Consider the predictor weight ( ).

where denotes the i-th observed value of weight and is
the coefficient of this predictor.

#### X

#### Y ^ i = β ^ 0 + β ^ 1 d 1 i + β ^ 2 d 2 i + β ^ 3 Xi ,

#### Xi β ^ 3


### ANCOVA model

The components of the ANCOVA model are individual functions
of the coefficients.

To gain insight into the model, we write it as follows:

The models have different intercepts but the same slope.

#### Y ^ i = β ^ 0 + β ^ 1 d 1 i + β ^ 2 d 2 i + β ^ 3 Xi

#### =.

#### ⎧

# ⎪

#### ⎨

# ⎪

#### ⎩

#### ( β ^ 0 + β ^ 1 ) + β ^ 3 Xi if car is European

#### ( β ^ 0 + β ^ 2 ) + β ^ 3 Xi if car is Japanese

#### β ^ 0 + β ^ 3 Xi if car is American

```
Tecnologico de Monterrey
```

To estimate , , and , we use least squares.

We could make individual inferences on and using t-
tests in the **statsmodels** library.

However, better tests are possible such as overall and partial
F-tests (not discussed here).

#### β ^ 0 β ^ 1 β ^ 2 β ^ 3

#### β 1 β 2


### In Python

To fit an ANCOVA model, we use similar steps as before. The
only extra step is to concatenate the data with the dummy
variables and the numerical predictor using the function
concat() from **pandas**.
1 # Concatenate the two data sets.
2 X_train = pd.concat([dummy_X_train, auto_data['weight']], axis = 1 )
3
4 # 1. Create the linear regression model
5 LRmodel_ancova = LinearRegression()
6
7 # 2. Fit the model.
8 LRmodel_ancova.fit(X_train, Y_train)


The estimated model is shown below.
1 # 3. Show estimated coefficients.
2 print("Intercept:", LRmodel_ancova.intercept_)
3 print("Coefficients:", LRmodel_ancova.coef_)
Intercept: 43.
Coefficients: [ 0.97090559 2.3271499 -0.00702708]


## Another Example


### Example 2

The “BostonHousing.xlsx” contains data collected by the US
Bureau of the Census concerning housing in the area of
Boston, Massachusetts. The dataset includes data on 506
census housing tracts in the Boston area in 1970s.

The goal is to predict the median house price in new tracts
based on information such as crime rate, pollution, and
number of rooms.

The response is the median value of owner-occupied homes in
$1000s, contained in the column MEDV.


### The predictors

```
CRIM: per capita crime rate by town.
ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
INDUS: proportion of non-retail business acres per town.
CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
NOX: nitrogen oxides concentration (parts per 10 million).
RM: average number of rooms per dwelling.
AGE: proportion of owner-occupied units built prior to 1940.
DIS: weighted mean of distances to five Boston employment centers
RAD: index of accessibility to radial highways.
TAX: full-value property-tax rate per $10,000.
PTRATIO: pupil-teacher ratio by town.
LSTAT: lower status of the population (percent).
```

### Read the dataset

We read the dataset and set the variable CHAS as categorical.

##### CRIM ZN INDUS CHAS NOX RM AGE

```
0 0.00632 18.0 2.31 No 0.538 6.575 65.2 4.0
1 0.02731 0.0 7.07 No 0.469 6.421 78.9 4.9
2 0.02729 0.0 7.07 No 0.469 7.185 61.1 4.9
```
```
1 # Load Excel file (make sure the file is in your Colab)
2 Boston_data = pd.read_excel('BostonHousing.xlsx')
3
4 # Drop the categorical variable.
5 Boston_data['CHAS'] = pd.Categorical(Boston_data['CHAS'])
6 Boston_data['RAD'] = pd.Categorical(Boston_data['RAD'])
7
8 # Preview the dataset.
9 Boston_data.head( 3 )
```

### Train and validation data

We split the dataset into a training and a validation dataset
using the function train_test_split() from **scikit-learn**.

The parameter test_size sets the portion of the dataset that
will go to the validation set. It is usually 0.3 or 0.2.

```
1 # Set full matrix of predictors.
2 X_full = Boston_data.drop(columns = ['MEDV'])
3
4 # Set full matrix of responses.
5 Y_full = Boston_data['MEDV']
6
7 # Split the dataset into training and validation.
8 X_train, X_valid, Y_train, Y_valid = train_test_split(X_full, Y_full,
9 test_size=0.3,
10 random_state = 59227 )
```

### Transform the categorical predictors

Before we continue, let’s turn the categorical predictors into
dummy variables

##### CRIM ZN INDUS NOX RM AGE DIS

##### 452 5.09017 0.0 18.10 0.713 6.297 91.8 2.3682

##### 317 0.24522 0.0 9.90 0.544 5.782 71.7 4.0317

##### 352 0.07244 60.0 1.69 0.411 5.884 18.5 10.7103

```
1 # Transform categorical predictors into dummy variables
2 Boston_X_train = pd.get_dummies(X_train, drop_first = True, dtype = 'int')
3 Boston_X_train.head( 3 )
```

### Fit a model using training data

We train a linear regression model to predict the MEDV in terms
of the 12 predictors using functions from **scikit-learn**.
1 # 1. Create the linear regression model
2 LRmodel = LinearRegression()
3
4 # 2. Fit the model.
5 LRmodel.fit(Boston_X_train, Y_train)
6
7 # 3. Show estimated coefficients.
8 print("Intercept:", LRmodel.intercept_)
9 print("Coefficients:", LRmodel.coef_)
Intercept: 37.516763561108434
Coefficients: [-5.31702151e-02 4.42408175e-02 8.46402739e-02 -1.19782458e+01
3.82114415e+00 1.77709868e-02 -1.25752076e+00 -7.33041781e-03
-9.14794029e-01 -7.17544266e-01 5.25221827e+00 -2.04793579e+00
-2.47739812e+00]


### Brief Residual Analysis

We evaluate the model using a “Residual versus Fitted Values”
plot. The plot does not show concerning patterns in the
residuals. So, we assume that the model satisfies the
assumption of constant variance.


### Validation Mean Squared Error

When the response is numeric, the most common evaluation
metric is the validation **Root Mean Squared Error** ( **RMSE** ):

where are the observations in
the validation dataset, and

is the model
prediction of the i-th response.

#### (

#### 1

#### nv

```
nv
```
#### ∑

```
i = 1
```
#### ( Yi − Y ^ i )

```
2
```
#### )

```
1 / 2
```
#### ( Y 1 , X 1 ),..., ( Ynv , X nv ) nv

#### Y ^ i = β ^ 0 + β ^ 1 Xi 1 + β ^ 2 Xi 2 + ⋯ + β ^ pXip


### Recall that ...

```
The RMSE is in the same units as the response.
The RMSE value is interpreted as either how far (on average)
the residuals are from zero.
It can also be interpreted as the average distance between
the observed response values and the model predictions.
```

### In Python

We first compute the predictions of our model on the
validation dataset. To this end, we create the dummy variables
of the categorical variables using the validation dataset.

##### CRIM ZN INDUS NOX RM AGE DIS

##### 482 5.73116 0.0 18.1 0.532 7.061 77.0 3.4106

##### 367 13.52220 0.0 18.1 0.631 3.863 100.0 1.5106

##### 454 9.51363 0.0 18.1 0.713 6.728 94.1 2.4961

```
1 Boston_X_val = pd.get_dummies(X_valid, drop_first = True, dtype = 'int')
2 Boston_X_val.head( 3 )
```

Now, we use the values of the predictors in this dataset and
use it as input to our model. Our model then computes the
prediction of the response for each combination of values of
the predictors.

We compute the validation RMSE using **scikit-learn**.

The lower the validation RMSE, the more accurate our model.

Interpretation: On average, our predictions are off by 5,981
dollars.

```
1 # Predict responses using validation data.
2 predicted_medv_val = LRmodel.predict(Boston_X_val)
```
```
1 rmse = root_mean_squared_error(Y_valid, predicted_medv_val)
2 print( round(rmse, 3 ) )
6.02
```
#### ±


## Return to main page


