# Example 8.3
# Load Packages
import pandas as pd
df = pd.read_csv('driving_test_measurements.csv')
df.drop('User ID',axis=1,inplace=True)
total = df.isnull().sum().sort_values(ascending = False)
print(total)
# Impute missing values of numeric data with mean
cols = list(df)
for column in cols:
   if column in df.select_dtypes(include = np.number).columns:
      df[column].fillna(df[column].mean(), inplace = True)

# Checking for Outliers
cols = list(df)
outlier_data = []
for column in cols:
   if column in df.select_dtypes(include = np.number).columns:
      q1 = df[column].quantile(0.25)
      # first quartile (Q1)
      q3 = df[column].quantile(0.75)
      # third quartile (Q3)
      iqr = q3 - q1
      upper_limit = q3 + (1.5*iqr)
      lower_limit = q1 - (1.5*iqr) # Define lower_limit
      num_outliers = df.loc[(df[column] < lower_limit) | (df[column] > upper_limit)].shape[0]
      outlier_data.append({'Column': column, 'Number of Outliers': num_outliers})

outliers = pd.DataFrame(outlier_data)
print(outliers)
