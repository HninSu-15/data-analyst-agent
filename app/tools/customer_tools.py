from __future__ import annotations

import pandas as pd


def calculate_customer_metrics(df: pd.DataFrame) -> dict:
    customer_df = df.copy()
    required = {"customer_id", "order_id", "quantity", "unit_price", "discount_pct", "cost_per_unit", "order_status"}
    missing = required - set(customer_df.columns)
    if missing:
        raise ValueError(f"Customer analysis requires columns: {sorted(missing)}")

    customer_df["quantity"] = pd.to_numeric(customer_df["quantity"], errors="coerce")
    customer_df["unit_price"] = pd.to_numeric(customer_df["unit_price"], errors="coerce")
    customer_df["discount_pct"] = pd.to_numeric(customer_df["discount_pct"], errors="coerce")
    customer_df["cost_per_unit"] = pd.to_numeric(customer_df["cost_per_unit"], errors="coerce")

    customer_df["gross_sales"] = customer_df["quantity"] * customer_df["unit_price"]
    customer_df["discount_amount"] = customer_df["gross_sales"] * customer_df["discount_pct"] / 100
    customer_df["net_sales"] = customer_df["gross_sales"] - customer_df["discount_amount"]

    completed = customer_df[customer_df["order_status"].astype(str).str.strip() == "Completed"].copy()
    customer_order_counts = completed.groupby("customer_id")["order_id"].nunique()
    customer_spend = completed.groupby("customer_id")["net_sales"].sum()

    returning_customers = int((customer_order_counts > 1).sum())
    total_customers = int(customer_order_counts.shape[0])
    unique_customers = int(completed["customer_id"].nunique())
    avg_customer_spend = float(customer_spend.mean()) if not customer_spend.empty else 0.0
    order_count_distribution = customer_order_counts.value_counts().sort_index()

    return {
        "unique_customers": unique_customers,
        "total_customers": total_customers,
        "returning_customers": returning_customers,
        "returning_customer_rate": round((returning_customers / total_customers * 100), 2) if total_customers else 0.0,
        "average_customer_spend": round(avg_customer_spend, 2),
        "order_count_distribution": {
            str(int(order_count)): int(customer_count)
            for order_count, customer_count in order_count_distribution.items()
        },
        "top_customer_spend_values": [
            round(float(value), 2)
            for value in customer_spend.sort_values(ascending=False).head(10).tolist()
        ],
    }
