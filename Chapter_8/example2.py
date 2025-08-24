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
outliers = pd.DataFrame(columns=['Column','Number of Outliers'])
for column in cols:
   if column in df.select_dtypes(include = np.number).columns:
      q1 = df[column].quantile(0.25)
      # first quartile (Q1)
      q3 = df[column].quantile(0.75)
      # third quartile (Q3)
      iqr = q3 - q1
      upper_limit = q3 + (1.5*iqr)
      outliers = outliers.append(
                  {'Column':column,'Number of Outliers':
                    df.loc[(df[column] < lower_limit) |
                          df[column] > upper_limit)].shape[0]}, ignore_index = True)
                                                                                                   
      print(outliers)