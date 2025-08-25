# Example 10.1
# Load Packages
import pandas as pd
df = pd.read_csv('BankCustomers.csv',delimiter=',')
df.drop('Id',axis=1,inplace=True)
df[column_name].fillna(df[column_name].mean( ),inplace=True)
df = df.dropna(axis=0)
total = df.isnull().sum().sort_values(ascending = False)
print(total)

# Checking for Outliers
cols = list(df)
outliers = pd.DataFrame(columns=['Column', 'Number of Outliers'])
for column in cols:
        if column in df.select_dtypes(include=np.number).columns:
            q1 = df[column].quantile(0.25) # first quartile (Q1)
            q3 = df[column].quantile(0.75) # third quartile (Q3)
            iqr = q3 - q1                  # Inter Quartile Range(IQR)
            lower_limit = q1 - (1.5 * iqr)
            upper_limit = q3 + (1.5 * iqr)
            outliers = outliers.append(
             				        
                    	{'Column':column,'Number of Outliers':
              					   
                           df.loc[(df[column] < lower_limit) | 
           						      
                           (df[column] > upper_limit)].shape[0]},
                                               ignore_index=True)
print(outliers)

# Bar Charts
categorical_columns = df.select_dtypes(include= ['object']).columns
for i in range(0,len(categorical_columns),2):
			if len(categorical_columns) > i+1:  
     	   plt.figure(figsize=(10,4))
			   plt.subplot(121)
						     
         df[categorical_columns[i]].value_counts(normalize=True).plot(kind='bar')
 		     plt.title(categorical_columns[i])
         plt.subplot(122)     
         df[categorical_columns[i+1]].value_counts(normalize=True).plot(kind='bar') 
   		   plt.title(categorical_columns[i+1])
  			 plt.tight_layout()
         plt.show()
		else:
       				    
         df[categorical_columns[i]].value_counts(normalize=True).plot(kind='bar')
         plt.title(categorical_columns[i])

#	Impute mising values('unknown', 'NaN') of 
# Categorical data with mode
cols = list(df)
	for column in cols:
    if column in df.select_dtypes(exclude=np.number).columns:
			     df[column]=df[column].str.replace 				          
                                ('unknown',df[column].mode()[0])
 			     df[column]=df[column].str.replace('NaN',df[column].mode()[0])

# Univariate Analysis of Continuous Variables
# Histograms and Box Plots
numeric_columns = df.select_dtypes(include= ['number']).columns.tolist()
for i in range(0,len(numeric_columns),2):
	if len(numeric_columns) > i+1:
 		plt.figure(figsize=(10,4))
		plt.subplot(121)
		sns.distplot(df[numeric_columns[i]], kde=False)
		plt.subplot(122)            
    sns.distplot(df[numeric_columns[i+1]], kde=False)
		plt.tight_layout()          
		plt.show()
	else:
		sns.distplot(df[numeric_columns[i]], kde=False)

# Code to generate Box Plot
for I in range (0, len(numeric_columns), 2):
  if len(numeric_columns) > i+1 :
    plt.figure(figsize = (10, 4))  
    plt.subplot(121)
		sns.boxplot(df[numeric_columns[i]])
		plt.subplot(122)            
		sns.boxplot(df[numeric_columns[i+1]])
		plt.tight_layout()
		plt.show()
	  else:
  		sns.boxplot(df[numeric_columns[i]])
		df.drop(['pdays','previous'],1, inplace=True)

from scipy.stats.mstats import winsorize
cols = list(df)
for column in cols:
	   if column in df.select_dtypes (include = np.number).columns:
		   df[column] = winsorize (df[col], limits = [0.05, 0.10],inclusive = (True, True))

# Bivariate Analysis of Continuous Variables
# Categorical Variables are plotted against the target variable
%matplotlib inline
categorical_columns = df.select_dtypes(exclude = np.number).columns
for i in range(0,len(categorical_columns),2):
  if len(categorical_columns) > i+1:
    plt.figure(figsize=(15,5))
    plt.subplot(121)
    sns.countplot(x=df[categorical_columns[i]], hue = df['y'],data=df)	  plt.xticks(rotation=90)
    plt.subplot(122)            
    sns.countplot(df[categorical_columns[i+1]], hue = df['y'],data=df)	  plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

# Code to perform Label Encoding of Categorical Variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
categorical_columns = list(df.select_dtypes(include = ['category','object']))
for column in categorical_columns:
    try:
        df[column] = le.fit_transform(df[column])
    except:
        print('Error encoding '+ column)

# Code for Logistic Regression Model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, roc_curve, confusion_matrix
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
model1 = LogisticRegression()
model1.fit(X,y)
y_pred = model1.predict(X)
auc = roc_auc_score(y, y_pred)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y, y_pred)
print('ROC_AUC_SCORE is', auc)
plt.plot(false_positive_rate, true_positive_rate)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.show()


# Feature Selection using RFE for Random Forest Technique
from sklearn.feature_selection import RFE 
X = df.iloc[:, 0:-1]
y = df.iloc[:,-1]
model6 = RandomForestClassifier()
rfe = RFE(model6, 8)
rfe = rfe.fit(X,y)
feature_ranking = pd.Series(rfe.ranking, index = X.columns)
plt.show()
print('Features to be selected for Random Forest Model')
print(feature_ranking[feature_ranking.values == 1].index.tolist())
print("----" * 30)

# Random Forest Technique
model8 = RandomForestClassifier()
param_grid = { 'max_features' : ['auto', 'sqrt', 'log2'],
               'max_depth'  : [4, 5, 6, 7, 8],
               'criterion'  : ['gini', 'entropy']
             }
grid_search_model = GridSearchCV(model8, param_grid = param_grid)
grid_search_model.fit(X,y)
print('Best Parameters are:', grid_search_model.best_params_)

# Applying best parameters & best features for 
# Random Forest Model
# SMOTE(Synthetic Minority Oversampling Technique)
# is applied to handle class imbalance
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, roc_curve, confusion_matrix 
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
smote = SMOTE(kind = 'regular')
X_sm, y_sm = smote.fit_sample(X[['duration',
            'euribor3m','age','nr.employed','job',
         		'day_of_week','campaign','education']], y)
model9 = RandomForestClassifier(n_estimators = 8, 
            max_features = 'auto', max_depth = 8, criterion = 'gini')
model9.fit(X_sm,y_sm)
y_pred = model9.predict(X[['duration','euribor3m', 'pdays','nr.employed',
			'emp.var.rate', 'poutcome', 'cons.price.idx', 'cons.conf.idx']])
false_positive_rate, true_positive_rate, thresholds = roc_curve(y,y_pred)
print('ROC_AUC_SCORE is', roc_auc_score(y, y_pred))
print(classification_report(y, y_pred))
plt.clf()
plt.plot(false_positive_rate, true_positive_rate)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.show()
