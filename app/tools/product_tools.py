from __future__ import annotations

import pandas as pd


def calculate_product_metrics(df: pd.DataFrame) -> dict:
    product_df = df.copy()
    required = {"product_id", "product_name", "category", "gender", "size", "color", "quantity", "unit_price", "discount_pct", "cost_per_unit", "order_status"}
    missing = required - set(product_df.columns)
    if missing:
        raise ValueError(f"Product analysis requires columns: {sorted(missing)}")

    product_df["quantity"] = pd.to_numeric(product_df["quantity"], errors="coerce")
    product_df["unit_price"] = pd.to_numeric(product_df["unit_price"], errors="coerce")
    product_df["discount_pct"] = pd.to_numeric(product_df["discount_pct"], errors="coerce")
    product_df["cost_per_unit"] = pd.to_numeric(product_df["cost_per_unit"], errors="coerce")

    product_df["gross_sales"] = product_df["quantity"] * product_df["unit_price"]
    product_df["discount_amount"] = product_df["gross_sales"] * product_df["discount_pct"] / 100
    product_df["net_sales"] = product_df["gross_sales"] - product_df["discount_amount"]
    product_df["total_cost"] = product_df["quantity"] * product_df["cost_per_unit"]
    product_df["profit"] = product_df["net_sales"] - product_df["total_cost"]

    completed = product_df[product_df["order_status"].astype(str).str.strip() == "Completed"].copy()

    revenue_by_product = completed.groupby(["product_id", "product_name"], dropna=False)["net_sales"].sum().sort_values(ascending=False)
    units_by_product = completed.groupby(["product_id", "product_name"], dropna=False)["quantity"].sum().sort_values(ascending=False)
    category_revenue = completed.groupby("category", dropna=False)["net_sales"].sum().sort_values(ascending=False)
    size_popularity = completed.groupby("size", dropna=False)["quantity"].sum().sort_values(ascending=False)
    color_popularity = completed.groupby("color", dropna=False)["quantity"].sum().sort_values(ascending=False)
    gender_revenue = completed.groupby("gender", dropna=False)["net_sales"].sum().sort_values(ascending=False)

    return {
        "top_products_by_revenue": [
            {"product_id": str(pid), "product_name": str(name), "net_sales": round(float(value), 2)}
            for (pid, name), value in revenue_by_product.head(5).items()
        ],
        "top_products_by_quantity": [
            {"product_id": str(pid), "product_name": str(name), "units_sold": int(value)}
            for (pid, name), value in units_by_product.head(5).items()
        ],
        "category_revenue": {str(key): round(float(value), 2) for key, value in category_revenue.items()},
        "size_popularity": {str(key): int(value) for key, value in size_popularity.items()},
        "color_popularity": {str(key): int(value) for key, value in color_popularity.items()},
        "gender_revenue": {str(key): round(float(value), 2) for key, value in gender_revenue.items()},
    }
