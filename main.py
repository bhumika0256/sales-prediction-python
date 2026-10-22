import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# 1. Load Data
try:
    df = pd.read_csv('advertising.csv')
    print("✅ Dataset loaded successfully!")
except FileNotFoundError:
    print("❌ Error: advertising.csv not found. Please put the file in the same folder.")
    exit()

# 2. Exploratory Data Analysis (EDA) - Creating Charts for your Video
plt.figure(figsize=(15, 5))

# Chart 1: TV vs Sales
plt.subplot(1, 3, 1)
sns.regplot(x='TV', y='Sales', data=df, color='blue')
plt.title('TV Spend vs Sales')

# Chart 2: Radio vs Sales
plt.subplot(1, 3, 2)
sns.regplot(x='Radio', y='Sales', data=df, color='green')
plt.title('Radio Spend vs Sales')

# Chart 3: Newspaper vs Sales
plt.subplot(1, 3, 3)
sns.regplot(x='Newspaper', y='Sales', data=df, color='red')
plt.title('Newspaper Spend vs Sales')

plt.tight_layout()
plt.savefig('advertising_analysis.png') # Save for your report/video
print("📊 Analysis charts saved as 'advertising_analysis.png'")
plt.show()

# 3. Data Preprocessing
X = df[['TV', 'Radio', 'Newspaper']] # Features
y = df['Sales']                      # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Evaluation
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n--- Model Performance ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared Score (R2): {r2:.2f}")

# 6. Save the model for the Demo App
joblib.dump(model, 'sales_model.pkl')
print("\n💾 Model saved as 'sales_model.pkl'")

# 7. Final Prediction Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, color='purple', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.savefig('actual_vs_predicted.png')
plt.show()