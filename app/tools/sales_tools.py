from __future__ import annotations

import pandas as pd


def _prepare_sales_frame(df: pd.DataFrame) -> pd.DataFrame:
    sales_df = df.copy()
    required_columns = {"quantity", "unit_price", "discount_pct", "cost_per_unit", "order_status", "order_date"}
    missing = required_columns - set(sales_df.columns)
    if missing:
        raise ValueError(f"Sales analysis requires columns: {sorted(missing)}")

    sales_df["order_date"] = pd.to_datetime(sales_df["order_date"], errors="coerce")
    sales_df["quantity"] = pd.to_numeric(sales_df["quantity"], errors="coerce")
    sales_df["unit_price"] = pd.to_numeric(sales_df["unit_price"], errors="coerce")
    sales_df["discount_pct"] = pd.to_numeric(sales_df["discount_pct"], errors="coerce")
    sales_df["cost_per_unit"] = pd.to_numeric(sales_df["cost_per_unit"], errors="coerce")

    sales_df["gross_sales"] = sales_df["quantity"] * sales_df["unit_price"]
    sales_df["discount_amount"] = sales_df["gross_sales"] * sales_df["discount_pct"] / 100
    sales_df["net_sales"] = sales_df["gross_sales"] - sales_df["discount_amount"]
    sales_df["total_cost"] = sales_df["quantity"] * sales_df["cost_per_unit"]
    sales_df["profit"] = sales_df["net_sales"] - sales_df["total_cost"]
    sales_df["profit_margin"] = sales_df["profit"] / sales_df["net_sales"].replace(0, pd.NA)
    return sales_df


def calculate_sales_metrics(df: pd.DataFrame) -> dict:
    sales_df = _prepare_sales_frame(df)

    completed_orders = sales_df[sales_df["order_status"].astype(str).str.strip() == "Completed"].copy()
    if completed_orders.empty:
        return {
            "total_net_sales": 0.0,
            "total_units_sold": 0,
            "completed_orders": 0,
            "average_order_value": 0.0,
            "revenue_by_region": {},
            "monthly_sales_trend": {},
        }

    region_revenue = (
        completed_orders.groupby("region", dropna=False)["net_sales"].sum().sort_values(ascending=False)
    )
    monthly = completed_orders.assign(order_month=lambda frame: frame["order_date"].dt.to_period("M").astype(str))
    monthly_sales = monthly.groupby("order_month")["net_sales"].sum().sort_index()

    order_value = completed_orders.groupby("order_id")["net_sales"].sum()
    total_net_sales = float(completed_orders["net_sales"].sum())
    total_units = int(completed_orders["quantity"].sum())
    completed_order_count = int(completed_orders["order_id"].nunique())
    average_order_value = float(order_value.mean()) if not order_value.empty else 0.0

    result = {
        "total_net_sales": round(total_net_sales, 2),
        "total_units_sold": total_units,
        "completed_orders": completed_order_count,
        "average_order_value": round(average_order_value, 2),
        "revenue_by_region": {str(region): round(float(value), 2) for region, value in region_revenue.items()},
        "monthly_sales_trend": {str(month): round(float(value), 2) for month, value in monthly_sales.items()},
    }

    if len(monthly_sales) > 1:
        result["growth_vs_previous_period"] = _growth_rate(monthly_sales)

    return result


def _growth_rate(series: pd.Series) -> dict:
    series = series.sort_index()
    current = float(series.iloc[-1])
    previous = float(series.iloc[-2]) if len(series) > 1 else current
    if previous == 0:
        change = 0.0
    else:
        change = ((current - previous) / previous) * 100
    return {
        "current_period": str(series.index[-1]),
        "previous_period": str(series.index[-2]) if len(series) > 1 else str(series.index[-1]),
        "pct_change": round(change, 2),
    }
