# DSML-Final-Project
**Itua Eriaye**

Data Science and Machine Learning - Full Time Course | June 2024

Brief Overview of the Problem
The Supply Chain Industry is constantly experiencing high variability in shipping costs, Lead Times, pack prices,Inventory Levels, Supplier Reliability, Demand Variability etc., and these generally affects the overall cost efficiency.

Accurate predictions of one of these variables, shipping cost, in this case, based on various factors such as shipment mode, vendor, manufacturing site, origin, destination, and product category are essential for optimizing pricing strategies, reducing operational costs, and improving profit margins. This variability necessitates a predictive model to support strategic decision-making.

Main Objective
•	Specific Problem: I want to attempt optimize shipping costs by predicting the total shipment cost using various shipment features, with the goal of identifying cost-saving opportunities.

•	Goal and Benefit to Companies: To provide actionable insights that can help reduce shipping costs can significantly impact a company's bottom line, leading to increased profitability, reduced operational costs and optimise competitive pricing strategies.

•	Proposed Solution: Develop a predictive model to forecast shipment costs based on various factors such as shipment weight, distance, mode of transport, and more. 
The idea is to use the insights to identify high-cost routes and factors, and suggest cost-saving measures.

Below are the instructions on how to run your Shipping Cost Prediction code in Supply Chain Management and reproduce the results:

Prerequisites:
Python Environment: Ensure you have Python installed on your system. It is recommended to use Python 3.6 or above.
Libraries: Install the necessary libraries. Commonly used libraries for such tasks include Pandas, NumPy, Scikit-learn, and Matplotlib. You can install these using pip.
Step-by-Step Instructions:
1. Set Up Your Environment:
Create a Virtual Environment (Optional but recommended):
On Windows:
python -m venv myenv
myenv\Scripts\activate
On macOS and Linux:
python3 -m venv myenv
source myenv/bin/activate
Install Required Libraries:
pip install pandas numpy scikit-learn matplotlib seaborn

2. Prepare Your Data:
Load the Dataset:
Ensure your dataset is in a CSV file format. Place it in your project directory.
import pandas as pd
df = pd.read_csv('shipping_data.csv')
Explore and Preprocess the Data:
Check for missing values, data types, and perform any necessary preprocessing like encoding categorical variables, normalizing numerical data, etc.
df.info()
df.describe()
df.fillna(method='ffill', inplace=True)

3. Feature Engineering:
Select Features and Target Variable:
X = df[['feature1', 'feature2', 'feature3']]  # Replace with your feature columns
y = df['shipping_cost']  # Replace with your target column
Split the Data into Training and Testing Sets:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

4. Build and Train the Model:
Choose a Model:
For example, a linear regression model:
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
Evaluate the Model:
from sklearn.metrics import mean_squared_error, r2_score
y_pred = model.predict(X_test)
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('R-squared Score:', r2_score(y_test, y_pred))

5. Visualize Results:
Plot Actual vs Predicted Values:
import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Shipping Costs')
plt.ylabel('Predicted Shipping Costs')
plt.title('Actual vs Predicted Shipping Costs')
plt.show()

6. Save the Model:
Save the Trained Model for Future Use:
import joblib
joblib.dump(model, 'shipping_cost_model.joblob')
7. Load and Use the Model:
Load the Model:
model = joblib.load('shipping_cost_model.joblob')
Make Predictions with New Data:
new_data = pd.DataFrame({
    'feature1': [value1],
    'feature2': [value2],
    'feature3': [value3]
})
prediction = model.predict(new_data)
print('Predicted Shipping Cost:', prediction)


By following these instructions, you will be able to run the code for predicting shipping costs in supply chain management and reproduce the results. Ensure you adjust the code snippets to fit your specific dataset and feature names. If you encounter any specific issues or errors, please feel free to reach out to me so we can resolve them together.
























