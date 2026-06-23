import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data load karein
df = pd.read_csv('fuel_data.csv')

# Visual style set karein
sns.set_theme(style="darkgrid")
plt.figure(figsize=(10, 6))

# Lines plot karein
sns.lineplot(data=df, x='Date', y='Petrol_Price', marker='o', label='Petrol Price (PKR)', linewidth=2.5)
sns.lineplot(data=df, x='Date', y='Diesel_Price', marker='s', label='Diesel Price (PKR)', linewidth=2.5)

# Labels aur Title
plt.title('Pakistan Fuel Price Trend (2024)', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price per Liter (PKR)', fontsize=12)
plt.xticks(rotation=45)
plt.legend(fontsize=11)

# Layout automatically adjust karein aur show karein
plt.tight_layout()
plt.show()
