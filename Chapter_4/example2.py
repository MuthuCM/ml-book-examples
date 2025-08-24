# Example 4.2
# Classification Example 2 - Logistic Regression
# Import the libraries
import numpy as np
import pandas as pd
# Read data from a CSV file
df = pd.read_csv('VehicleData1.csv')
X = df["salary"].values
y = df["vehicle_type"].values
X = X.reshape(-1, 1)
# Do Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
# Fit Logistic Regression Model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X, y)
# Find Accuracy Score
y_pred = classifier.predict(X)
from sklearn import metrics
print("F-Score: ", metrics.f1_score(y, y_pred, average = 'weighted'))

# Do Prediction
X = [75000,92000,31000]
X = np.array(X)
X = X.reshape(-1,1)
X = sc.fit_transform(X)
predictedValues = classifier.predict(X)
print(predictedValues)

# Install Gradio Library
!pip install gradio -q
import gradio as gr
from sklearn.preprocessing import StandardScaler
import numpy as np
# Define function to predict salary
def predict_preference(prompt):
  model = 'Preference Prediction'     
  input_value = int(prompt)
  input_value = np.array([input_value]).reshape(-1, 1) 
  input_value = sc.transform(input_value)   
  predictedValue = classifier.predict(input_value)    
  preference = int(predictedValue[0])    
  return preference
# Create Gradio interface
with gr.Blocks() as PreferencePredictor:
	gr.Markdown("## Preference Predictor")
   	prompt_input = gr.Textbox(label="Enter Your Salary", placeholder = "e.g. 100000")
	style_input = gr.Dropdown(choices =["watercolor", "photorealistic"],label = "Choose a style" )
    output_value = gr.Textbox(label="Predicted Preference")
    generate_btn = gr.Button("Predict Preference")
    sc = StandardScaler()
    sc.fit(X)
    # Set the function to be called on button click
    generate_btn.click(fn = predict_preference,inputs = [prompt_input],  
                                                      outputs = output_value)
# Launch the Gradio interface
PreferencePredictor.launch()
