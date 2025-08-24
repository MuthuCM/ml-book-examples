# Example 2.3
# Simple Linear Regression - Example 3
# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Read Data from a CSV File
df = pd.read_csv('Salary_Data.csv')
X = df.iloc[:, 0].values
Y = df.iloc[:, 1].values
X = X.reshape(-1, 1)
# Fit Linear Regression Model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X, Y)
regression_coefficient = lr.coef_[0]
intercept = lr.intercept_
# Display Output
print(f"Regression Coefficient =  {regression_coefficient:5.2f} ")
print(f"Intercept = { intercept : 5.2f}" )
print()
print(f"Regression Equation is: Y = {regression_coefficient:5.2f} X  + { intercept : 5.2f}")
print()
# Calculate Accuracy Score
Y_pred = lr.predict(X)
from sklearn.metrics import r2_score
print(f"Accuracy is: {r2_score(Y, Y_pred):5.2f}")
print()
# Do Prediction
testInput = [7,9,3]
testInput = np.array (testInput)
testInput = testInput.reshape (-1,1)
predictedValues = regressor.predict(testInput)
print (predictedValues)
# Visualize the Regression Line
plt.scatter(X,Y,color = 'red')
plt.plot(X,lr.predict(X), color = 'blue')
plt.show()

# Install Gradio Library
! pip install gradio -q
import gradio as gr
# Define function to predict salary
def predict_salary(prompt):
  model = 'Salary Prediction'   # Replace with the desired model name
  input_value = int(prompt)
  input_value = np.array([input_value]).reshape(-1, 1)    
  predictedValue = regressor.predict(input_value)    
  salary = int(predictedValue[0])    
  return salary
# Create Gradio interface
with gr.Blocks() as SalaryPredictor:
  gr.Markdown("## Salary Predictor")
  prompt_input = gr.Textbox(label="Enter your Years of Experience", placeholder="e.g. 11")
  style_input = gr.Dropdown(choices=["watercolor", "photorealistic",
                                        "no style", "enhance", "anime"],
                                        label="Choose a style"
                                       )
  output_value = gr.Textbox(label="Predicted Salary")
  generate_btn = gr.Button("Predict Salary")
  # Set the function to be called on button click
  generate_btn.click(fn=predict_salary,inputs=[prompt_input], outputs=output_value)

  # Launch the Gradio interface
  SalaryPredictor.launch()
