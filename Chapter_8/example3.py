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

# Tackling Outliers through Winsorization
from scipy.stats.mstats import winsorize
cols = list(df)
for col in cols:
   if col in df.select_dtypes(include=np.number).columns:
     df[col] = winsorize(df[col],limits=    
                             [0.05, 0.1],inclusive=(True, True))

# Checking for class Imbalance
class_percentages = (df['Category'].value_counts()
                     / df['Category'].value_counts().sum())*100
print(class_percentages)

from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_sm, y_sm = smote.fit_resample(df[['concentration_score_1',
                     'concentration_score_2', 'concentration_score_3',
         'response_time_score_1', 'response_time_score_2',
         'response_time_score_3', 'response_time_score_4',
          'keep_cool_score_1', 'keep_cool_score_2',
          'keep_cool_score_3']], df['Category'])

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
model1 = LogisticRegression()
model1.fit(X,y)
y_pred = model1.predict(X)
print(classification_report(y, y_pred))
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
model2 = DecisionTreeClassifier()
X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]
model2.fit(X,y)
y_pred = model2.predict(X)

print(classification_report(y, y_pred))
