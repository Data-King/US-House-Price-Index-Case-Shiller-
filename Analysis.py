import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
import datetime

# Load the data
df = pd.read_csv('cities-month.csv')

# Basic information and summary statistics
print(df.info())
print(df.describe())

# Check for missing values and duplicates
print(df.isnull().sum())
print(f"Duplicate rows: {df.duplicated().sum()}")

# Data visualization
plt.figure(figsize=(12, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Time series analysis (assuming 'date' column exists)
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.resample('Y').mean().plot(figsize=(12, 6))
    plt.title('Yearly Average Trends')
    plt.show()

# Distribution of numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns
df[numerical_cols].hist(figsize=(15, 10), bins=30)
plt.tight_layout()
plt.show()

# Top 5 cities by a specific metric (e.g., population)
if 'city' in df.columns and 'population' in df.columns:
    print(df.groupby('city')['population'].mean().nlargest(5))

# Basic statistical tests
for col in numerical_cols:
    stat, p = stats.normaltest(df[col].dropna())
    print(f"{col}: p-value = {p:.4f}")




# New analysis sections:

# Boxplots for numerical columns
plt.figure(figsize=(15, 10))
df[numerical_cols].boxplot()
plt.title('Boxplots of Numerical Columns')
plt.xticks(rotation=45)
plt.show()

# Pairplot for key variables
key_vars = ['population', 'temperature', 'humidity']  # Adjust based on your data
sns.pairplot(df[key_vars])
plt.suptitle('Pairplot of Key Variables', y=1.02)
plt.show()

# Time-based analysis (if date column exists)
if 'date' in df.columns:
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    
    # Monthly trends
    monthly_avg = df.groupby('month')[numerical_cols].mean()
    monthly_avg.plot(figsize=(12, 6))
    plt.title('Monthly Trends')
    plt.show()
    
    # Yearly trends
    yearly_avg = df.groupby('year')[numerical_cols].mean()
    yearly_avg.plot(figsize=(12, 6))
    plt.title('Yearly Trends')
    plt.show()

# Correlation analysis
correlation_matrix = df[numerical_cols].corr()
high_correlations = correlation_matrix[abs(correlation_matrix) > 0.7].stack().reset_index()
high_correlations = high_correlations[high_correlations['level_0'] != high_correlations['level_1']]
print("High correlations (>0.7 or <-0.7):")
print(high_correlations)

# Outlier detection
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"Outliers in {col}: {len(outliers)}")

# Basic summary statistics by category (if applicable)
if 'city' in df.columns:
    print(df.groupby('city')[numerical_cols].agg(['mean', 'median', 'std']))

# Skewness and Kurtosis
for col in numerical_cols:
    skewness = df[col].skew()
    kurtosis = df[col].kurtosis()
    print(f"{col} - Skewness: {skewness:.2f}, Kurtosis: {kurtosis:.2f}")
