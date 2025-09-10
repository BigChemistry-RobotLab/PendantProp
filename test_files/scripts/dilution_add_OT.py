import pandas as pd

df = pd.read_csv("ML/Output/dilution_manager/dilution_volumes_CTAB_SDS.csv")
print(df.tail())
df_stocks_x = df["stock_x_used"].unique()
df_stocks_y = df["stock_y_used"].unique()

x: float = 0.
y: float = 0.
water: float = 0.

solution1 = "CTAB"
solution2 = "SDS"

for stocks in df_stocks_x:
	total = df[df["stock_x_used"] == stocks]["vol_stock_x"].sum()
	print(f"Total volume for {stocks} (x) :", round(total,2))
	x += total
	print(f"Total amount of {solution1} used: {round(x,2)}")

for stocks in df_stocks_y:
	total = df[df["stock_y_used"] == stocks]["vol_stock_y"].sum()
	print(f"Total volume for {stocks} (y):", round(total,2))
	y += total
	print(f"Total amount of {solution2} used: {round(y,2)}")

for stocks in df_stocks_x:
	total = df[df["stock_x_used"] == stocks]["vol_milliq"].sum()
	water += total
print(f"Total amount of MilliQ used: {round(water,2)}")
