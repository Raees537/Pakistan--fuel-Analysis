import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression

# 1. Data load aur tayaar karein
df = pd.read_csv('fuel_data.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Dates ko numbers mein badlein taake machine learning model samajh sake
df['Time_Index'] = np.arange(len(df))

X = df[['Time_Index']]  # Independent variable (Time)
y_petrol = df['Petrol_Price']  # Dependent variable (Petrol Price)

# 2. Model ko train karein (Machine Learning)
model_petrol = LinearRegression()
model_petrol.fit(X, y_petrol)

# 3. Aglay 3 mahino ki prediction karein
last_index = len(df) - 1
future_indices = np.array([[last_index + 1], [last_index + 2], [last_index + 3]])
predicted_prices = model_petrol.predict(future_indices)

# Prediction dates generate karein
last_date = df['Date'].iloc[-1]
future_dates = [last_date + pd.DateOffset(months=m) for m in range(1, 4)]

# 4. Results screen par print karein
print("\n--- 🔮 FUTURE FUEL PRICE FORECAST (NEXT 3 MONTHS) ---")
months_text = ["Next Month 1", "Next Month 2", "Next Month 3"]
for month, price, date in zip(months_text, predicted_prices, future_dates):
    print(f"{month} ({date.strftime('%Y-%m')}): {price:.2f} PKR")
print("---------------------------------------------------\n")

# 5. Graph show karein with Historical + Prediction Line
plt.figure(figsize=(10, 6))

# Historical Data Plot (Dark Blue)
plt.plot(df['Date'], df['Petrol_Price'], marker='o', color='#1f77b4', linewidth=2.5, label='Historical Petrol Price')

# Prediction Data Plot (Red, Dashed)
# Pehla point prediction line ka hum historical data ke last point se shuru karenge taake line continuous lage
plt.plot([df['Date'].iloc[-1]] + future_dates, [df['Petrol_Price'].iloc[-1]] + predicted_prices.tolist(), 
         color='red', linestyle='--', marker='s', markersize=8, linewidth=2.5, label='Model Prediction (Next 3 Months)')

# Graph Style and Labels
plt.title('Pakistan Petrol Price Forecast (2024-2026)', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price per Liter (PKR)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# X-axis dates format karna
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3)) # 3 mahine ke gap se date show karein
plt.xticks(rotation=45)

plt.legend(fontsize=11, loc='upper left')
plt.tight_layout()
plt.show()
