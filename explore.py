import matplotlib.pyplot as plt
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", None)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")

df = pd.read_csv("used_car_sales.csv")

print(df.shape)
print(df.to_string(index=False))
print(df.describe())

plt.scatter(df["Mileage_K"], df["Sale_Price"])
plt.xlabel("Mileage_K")
plt.ylabel("Sale_Price")
plt.tight_layout()
plt.savefig("scatter.png")
