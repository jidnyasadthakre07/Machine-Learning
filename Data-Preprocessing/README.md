# Data Preprocessing

## Overview

Data preprocessing is the process of cleaning, transforming, and
preparing raw data before training a machine learning model.

It helps handle missing values, duplicate records, outliers,
inconsistent data types, different feature scales, and skewed
distributions.

## Dataset

This project uses a water-quality dataset with numerical features such
as:

-   `ph`
-   `Hardness`
-   `Solids`
-   `Chloramines`
-   `Sulfate`
-   `Conductivity`
-   `Organic_carbon`
-   `Trihalomethanes`
-   `Turbidity`

The target column is `Potability`.

-   `0` = Not potable
-   `1` = Potable

Always verify the exact column names using:

``` python
df.columns
```

## 1. Import Libraries

``` python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.model_selection import train_test_split
```

## 2. Load and Understand Data

``` python
df = pd.read_csv("water_potability.csv")

df.head()
df.tail()
df.info()
df.shape
df.columns
df.describe()
df.dtypes
```

## 3. Missing Values

A missing value is an unavailable value, commonly represented by `NaN`.

### Check missing values

``` python
df.isnull().sum()
```

### Check whether any null value exists

``` python
df.isnull().values.any()
```

### Missing-value percentage

``` python
df.isnull().mean() * 100
```

Formula:

\[ `\text{Missing Percentage}`{=tex} =
`\frac{\text{Number of Missing Values}}`{=tex}
{`\text{Total Number of Values}`{=tex}} `\times 100`{=tex} \]

### Display rows containing missing values

``` python
df[df.isnull().any(axis=1)]
```

### Remove missing rows

``` python
df = df.dropna()
```

### Fill with mean

``` python
df["ph"] = df["ph"].fillna(df["ph"].mean())
```

### Fill with median

``` python
df["ph"] = df["ph"].fillna(df["ph"].median())
```

Mean formula:

\[ `\bar`{=tex}{x} = `\frac{\sum x_i}{n}`{=tex} \]

Median is the middle value after sorting and is less affected by
outliers.

## 4. Duplicate Values

Duplicate rows are repeated records.

### Count duplicates

``` python
df.duplicated().sum()
```

### Display duplicates

``` python
df[df.duplicated()]
```

### Remove duplicates

``` python
df = df.drop_duplicates()
```

## 5. Outliers

An outlier is an unusually high or low value.

Example:

``` text
10, 12, 15, 18, 1000
```

### Detect outliers using a boxplot

``` python
sns.boxplot(data=df, x="ph")
plt.show()
```

### Interquartile Range

\[ IQR = Q_3 - Q_1 \]

Lower bound:

\[ Q_1 - 1.5 `\times `{=tex}IQR \]

Upper bound:

\[ Q_3 + 1.5 `\times `{=tex}IQR \]

Values outside these limits are potential outliers. Do not remove them
automatically; first determine whether they are valid observations.

## 6. Train-Test Split

Separate features and target:

``` python
X = df.drop("Potability", axis=1)
y = df["Potability"]
```

Split the data:

``` python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

-   `X` = input features
-   `y` = target
-   `test_size=0.2` = 20% test data
-   `random_state=42` = reproducible split

Split before fitting a scaler to prevent data leakage.

## 7. Min-Max Scaling

Min-Max Scaling converts values to a fixed range, usually 0 to 1.

### Formula

\[ x\_{scaled} = `\frac{x-x_{min}}{x_{max}-x_{min}}`{=tex} \]

### Example

If the minimum is 10, maximum is 50, and the value is 30:

\[ `\frac{30-10}{50-10}`{=tex}=0.5 \]

### Python

``` python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train_minmax = scaler.fit_transform(X_train)
X_test_minmax = scaler.transform(X_test)

X_train_minmax = pd.DataFrame(
    X_train_minmax,
    columns=X_train.columns
)
```

Min-Max Scaling is sensitive to outliers because it uses the minimum and
maximum values.

## 8. Standardization

Standardization, also called Z-score normalization, transforms a feature
to approximately mean 0 and standard deviation 1.

### Formula

\[ z = `\frac{x-\mu}{\sigma}`{=tex} \]

Where:

-   \(x\) = original value
-   (`\mu`{=tex}) = mean
-   (`\sigma`{=tex}) = standard deviation

### Example

If the mean is 50, standard deviation is 10, and the value is 70:

\[ z = `\frac{70-50}{10}`{=tex}=2 \]

### Python

``` python
from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()

X_train_std = scalar.fit_transform(X_train)
X_test_std = scalar.transform(X_test)

X_train_std = pd.DataFrame(
    X_train_std,
    columns=X_train.columns
)

X_train_std.head()
```

Use `fit_transform()` only on training data. Use `transform()` on test
data with the same fitted scaler.

## 9. Robust Scaling

RobustScaler uses the median and interquartile range, so it is less
affected by outliers.

### Formula

\[ x\_{scaled} = `\frac{x-\text{Median}}{Q_3-Q_1}`{=tex} \]

### Example

If the median is 50, Q1 is 30, Q3 is 70, and the value is 90:

\[ IQR = 70-30=40 \]

\[ x\_{scaled}=`\frac{90-50}{40}`{=tex}=1 \]

### Python

``` python
from sklearn.preprocessing import RobustScaler

scalar = RobustScaler()

X_train_rob = scalar.fit_transform(X_train)
X_test_rob = scalar.transform(X_test)

X_train_rob = pd.DataFrame(
    X_train_rob,
    columns=X_train.columns
)

X_train_rob.head()
```

### Custom quantile range

``` python
scalar = RobustScaler(quantile_range=(20.0, 80.0))
```

Formula for this setting:

\[ x\_{scaled} = `\frac{x-\text{Median}}{P_{80}-P_{20}}`{=tex} \]

RobustScaler changes the scale; it does not remove outliers.

## 10. Comparison of Scaling Methods

  -----------------------------------------------------------------------------
  Method            Formula               Main Use            Outlier
                                                              Sensitivity
  ----------------- --------------------- ------------------- -----------------
  Min-Max Scaling   `(x-min)/(max-min)`   Fixed range         High

  Standardization   `(x-mean)/std`        Mean 0 and standard Sensitive
                                          deviation 1         

  Robust Scaling    `(x-median)/IQR`      Outlier-resistant   Lower
                                          scaling             
  -----------------------------------------------------------------------------

## 11. Data Transformations

### Box-Cox Transformation

Box-Cox is a power transformation used to reduce skewness.

For (`\lambda `{=tex} e 0):

\[ y = `\frac{x^\lambda-1}{\lambda}`{=tex} \]

For (`\lambda=0`{=tex}):

\[ y=`\ln`{=tex}(x) \]

Box-Cox generally requires positive values.

``` python
df["ph_boxcox"], fitted_lambda = stats.boxcox(df["ph"])
```

### Square Root Transformation

\[ y=`\sqrt{x}`{=tex} \]

``` python
df["ph_sqrt"] = np.sqrt(df["ph"])
```

It can reduce right skewness and compress large values.

### Log Transformation

\[ y=`\ln`{=tex}(x) \]

``` python
df["ph_log"] = np.log(df["ph"])
```

For non-negative data containing zero:

``` python
df["column_log"] = np.log1p(df["column"])
```

`log1p(x)` calculates (`\ln`{=tex}(1+x)).

## 12. KDE Plot

KDE means Kernel Density Estimation. It is used to visualize the
distribution of numerical data.

It helps identify:

-   Distribution shape
-   Skewness
-   Spread
-   Possible outliers
-   Changes after scaling or transformation

### Plot one feature

``` python
sns.kdeplot(data=X_train_std, x="ph", fill=True)
plt.show()
```

### Plot multiple features

``` python
sns.kdeplot(data=X_train_std, x="ph", fill=True)
sns.kdeplot(data=X_train_std, x="Hardness", fill=True)
sns.kdeplot(data=X_train_std, x="Solids", fill=True)
plt.show()
```

The name passed to `x` must exactly match a column in the DataFrame.

Check column names:

``` python
X_train_std.columns
```

For example, `Chloramines` and `chlorine` are different names.

## 13. Recommended Workflow

``` text
Load Data
    ↓
Understand Data
    ↓
Check Data Types
    ↓
Check Missing Values
    ↓
Check Duplicate Values
    ↓
Handle Missing Values
    ↓
Inspect Outliers
    ↓
Separate Features and Target
    ↓
Train-Test Split
    ↓
Fit Scaler on Training Data
    ↓
Transform Training Data
    ↓
Transform Testing Data
    ↓
Train Model
    ↓
Evaluate Model
```

## 14. Complete Basic Example

``` python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("water_potability.csv")

# Check missing values and duplicates
print(df.isnull().sum())
print(df.duplicated().sum())

# Separate features and target
X = df.drop("Potability", axis=1)
y = df["Potability"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Fit scaler only on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data using the same scaler
X_test_scaled = scaler.transform(X_test)
```

## 15. Questions

### What is data preprocessing?

It is the process of cleaning and transforming raw data before model
training.

### Why do we scale numerical data?

To bring features to comparable scales and prevent large-valued features
from dominating distance-based or optimization-based algorithms.

### Why is RobustScaler used?

It uses the median and IQR, so it is less sensitive to outliers.

### Does scaling remove outliers?

No. Scaling changes numerical values but does not remove outliers.

### Why use `fit_transform()` on training data?

The scaler learns its parameters from the training data and transforms
it.

### Why use `transform()` on test data?

The test data must use the same parameters learned from the training
data.

### Why use KDE plots?

To visualize the distribution and shape of numerical features.

## Conclusion

Data preprocessing improves the quality and consistency of input data.
The correct method depends on the feature distribution, missing values,
outliers, and the machine learning algorithm.

Always inspect the dataset before selecting a preprocessing technique.

