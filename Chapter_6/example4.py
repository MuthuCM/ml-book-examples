# Example 6.4
# Load Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
# Load Data
data = pd.read_csv ("bmi.csv")
data.head()
data.info()

# Perform Preprocessing
data.isnull().sum()
if data.duplicated().any():
	   print("There are duplicates")
else:
   print("There are no duplicates")
# print("Number of duplicates: " + str(data.duplicated().sum()))
data.drop_duplicates(keep = False, inplace = True)
print("DataFrame shape: " , data.shape)

# Do Label Encoding
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
objList data.select_dtypes(include = "object").columns
for obj in objList:
   data[obj] = lb.fit_transform(data[obj].astype(str))
data.head()

# Visualize bivariate relationships
# !pip install -q plotly pandas
# !pip install-U kaleido
import plotly.express as px
% matplotlib inline
sns.color_palette("PiYG")
sns.set_style("whitegrid")
# Scatter Diagram
fig = px.scatter( data, x = 'Height', y = 'Weight', color = 'Index',
                  title = 'Height vs Weight with index',
                  labels={'Height':'Height(cm)','Weight':'Weight (kg)'},
                  category_orders = {'Index':['0','1','2','3','4','5']},
                  hover_name = 'Index',
                  hover_data = {'Height': True, 'Weight': True},)
fig.update_layout( showlegend = True, legend_title_text = 'Index',
                   xaxis = dict(gridcolor='lightgray'),
                   yaxis=dict(gridcolor='lightgray'),
                   paper_bgcolor='lightgray',plot_bgcolor='white',)
fig.show()

# Define Independent and Dependent Variables
X = data.drop('Index', axis = 1).values
y = data['Index'].values
# Splitting into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                    test_size = 0.2,
                                    random_state = 42)

# Fit Logistic Regression Model
log_model = LogisticRegression()
log_model.fit(X_train,y_train)
y_pred1 = log_model.predict(X_test)
acc = accuracy_score(y_test, y_pred1)
print('Accuracy for Logistic Regression Model is',acc)

# Fit KNN Classification Model
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred2 = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred2)
print('Accuracy for K Nearest Neighbor Model is',accuracy)

# Fit KNN Model with 6 as the value of n_neighbors parameter
Knn_best = KNeighborsClassifier(n_neighbors = 6)
Knn_best.fit(X_train, y_train)
# Calculate Accuracy
y_pred_test = knn_best.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
print('Test Set Accuracy is', test_accuracy)

# Do Prediction
prediction = knn_best.predict([[1, 150, 70]])
print(prediction)

# Install Gradio Library
# !pip install gradio -q
import gradio as gr
# Define function to predict salary
def predict_obesity(prompt1, prompt2, prompt3):
   model = 'Obesity Status Prediction'
   input_value1 = prompt1
   if input_value1 == 'Male':
      input_value1 = 1
   else:
      input_value1 = 0
   input_value2 = int(prompt2)
   input_value3 = int(prompt3)
   input_value1 = sc.transform(input_value1)
   prediction = knn_best.predict([[input_value1,input_value2, input_value3]])
   obesity_status = int(prediction[0])
   return obesity_status
# Create Gradio interface
with gr.Blocks() as ObesityPredictor:
   gr.Markdown("## Obesity Predictor")
   prompt_input1 = gr.Textbox(label="Enter Your Gender",
                              placeholder = "e.g. Male")
   prompt_input2 = gr.Textbox(label="Enter Your Height",
                              placeholder = "e.g. 174")
   prompt_input3 = gr.Textbox(label="Enter Your Weight", 
                             placeholder = "e.g. 96")
   style_input = gr.Dropdown(choices =["watercolor", "photorealistic"], 
                              label = "Choose a style" ) 
   output_value = gr.Textbox(label="Predicted Obesity Status")
   generate_btn = gr.Button("Predict Obesity Status")
   # Set the function to be called on button click
   generate_btn.click(fn = predict_obesity,inputs = [prompt_input1,
                           prompt_input2,prompt_input3],
                           outputs = output_value)
# Launch the Gradio interface
ObesityPredictor.launch()