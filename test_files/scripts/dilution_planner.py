import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DilutionPlanner:
    def min_stock_needed(C_target_min, C_target_max, C_stock, V_total=300, V_min=20):
        current_stock = C_stock
        stocks = [C_stock]
        if C_target_min > C_target_max:
            raise ValueError("C_target_min must be less than or equal to C_target_max.")
        if C_target_min <= 0 or C_target_max <= 0:
            raise ValueError("C_target_min and C_target_max must be greater than 0.")
        if C_target_max > C_stock / 2:
            raise ValueError("Stock should at least be twice as high as C_target_max.")
        while True:
            C_min_achievable = current_stock * V_min / V_total
            if C_min_achievable <= C_target_min:
                break
            current_stock /= 5
            stocks.append(current_stock)
        return sorted(stocks, reverse=True)

    def compute_dilution_volumes_2d(C_target_x, C_target_y, stocks_x, stocks_y, V_total=300, V_min=20):
        vol_x = None
        stock_used_x = None
        for C_stock_x in sorted(stocks_x, reverse=True):
            V_stock_x = C_target_x / C_stock_x * V_total
            if V_min <= V_stock_x <= V_total - V_min:
                vol_x = round(V_stock_x, 2)
                stock_used_x = C_stock_x
                break

        vol_y = None
        stock_used_y = None
        for C_stock_y in sorted(stocks_y, reverse=True):
            V_stock_y = C_target_y / C_stock_y * V_total
            if V_min <= V_stock_y <= V_total - V_min:
                vol_y = round(V_stock_y, 2)
                stock_used_y = C_stock_y
                break

        if vol_x is None or vol_y is None:
            print(vol_x, vol_y)
            return None

        vol_milliq = round(V_total - vol_x - vol_y, 2)
        if vol_milliq < 0:
            print("milliq error")
            return None

        return {
            "C_target_x": C_target_x,
            "C_target_y": C_target_y,
            "stock_x_used": stock_used_x,
            "vol_stock_x": vol_x,
            "stock_y_used": stock_used_y,
            "vol_stock_y": vol_y,
            "vol_milliq": vol_milliq
        }

    def compute_from_csv_targets(df, stocks_x, stocks_y, V_total=300, V_min=20):
        results = []
        for _, row in df.iterrows():
            cx = row['X']
            cy = row['Y']
            res = DilutionPlanner.compute_dilution_volumes_2d(cx, cy, stocks_x, stocks_y, V_total, V_min)
            print(res)
            if res:
                results.append(res)

        df_results = pd.DataFrame(results)
        return df_results

    def plot_results(df_results, solution_x_name, solution_y_name):
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(df_results["C_target_x"], df_results["C_target_y"],
                        c=df_results["vol_milliq"], cmap="viridis", s=50, edgecolor='k')
        plt.colorbar(sc, label="Volume MilliQ (µL)")
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(f'{solution_x_name} concentration (mM)')
        plt.ylabel(f'{solution_y_name} concentration (mM)')
        plt.title("2D Dilution Volumes: MilliQ Volume Distribution")
        plt.grid(True, which="both", ls="--", lw=0.5)
        plt.tight_layout()
        plt.show()