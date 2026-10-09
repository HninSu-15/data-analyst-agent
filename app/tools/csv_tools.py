from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

REQUIRED_COLUMNS = [
    "order_id",
    "order_date",
    "customer_id",
    "product_id",
    "product_name",
    "category",
    "gender",
    "size",
    "color",
    "quantity",
    "unit_price",
    "discount_pct",
    "cost_per_unit",
    "region",
    "payment_method",
    "order_status",
]

ALLOWED_ORDER_STATUSES = {"Completed", "Cancelled", "Returned"}


def load_csv(file_path: str | Path) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Dataset is empty.")
    return df


def validate_schema(df: pd.DataFrame, required_columns: Iterable[str] | None = None) -> dict:
    required = list(required_columns) if required_columns is not None else REQUIRED_COLUMNS
    missing_columns = [column for column in required if column not in df.columns]

    if missing_columns:
        return {
            "valid": False,
            "missing_columns": missing_columns,
            "warnings": [],
            "errors": [f"Missing required column: {column}" for column in missing_columns],
        }

    warnings: list[str] = []
    errors: list[str] = []

    try:
        df = df.copy()
        df["order_date"] = pd.to_datetime(df["order_date"], errors="raise")
    except (TypeError, ValueError):
        errors.append("order_date could not be parsed as a date.")

    numeric_columns = ["quantity", "unit_price", "discount_pct", "cost_per_unit"]
    for column in numeric_columns:
        if column in df.columns:
            try:
                numeric_values = pd.to_numeric(df[column], errors="raise")
            except (TypeError, ValueError):
                errors.append(f"{column} must be numeric.")
                continue

            if column == "quantity" and (numeric_values < 0).any():
                warnings.append("Some rows have negative quantity values.")
            elif column == "unit_price" and (numeric_values < 0).any():
                warnings.append("Some rows have negative unit_price values.")
            elif column == "discount_pct" and ((numeric_values < 0) | (numeric_values > 100)).any():
                warnings.append("Some rows have discount_pct outside the 0-100 range.")

    if "order_status" in df.columns:
        valid_statuses = set(df["order_status"].dropna().astype(str).str.strip())
        unexpected = valid_statuses - ALLOWED_ORDER_STATUSES
        if unexpected:
            warnings.append(
                f"Unexpected order_status values found: {sorted(unexpected)}. "
                "Values outside the expected catalog will be treated as warnings."
            )

    return {
        "valid": not errors,
        "missing_columns": [],
        "warnings": warnings,
        "errors": errors,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
    }


def profile_dataset(df: pd.DataFrame) -> dict:
    profile = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": {
            column: int(df[column].isna().sum())
            for column in df.columns
            if df[column].isna().any()
        },
        "missing_percentage": {
            column: round(float(df[column].isna().mean() * 100), 2)
            for column in df.columns
            if df[column].isna().any()
        },
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
    }

    numeric_checks = {
        "quantity": (df["quantity"] if "quantity" in df.columns else pd.Series(dtype="float64")),
        "unit_price": (df["unit_price"] if "unit_price" in df.columns else pd.Series(dtype="float64")),
        "discount_pct": (df["discount_pct"] if "discount_pct" in df.columns else pd.Series(dtype="float64")),
    }

    profile["numeric_ranges"] = {
        column: {
            "min": float(series.min()) if not series.empty and series.notna().any() else None,
            "max": float(series.max()) if not series.empty and series.notna().any() else None,
        }
        for column, series in numeric_checks.items()
        if column in df.columns
    }

    if "order_status" in df.columns:
        profile["order_status_counts"] = df["order_status"].value_counts(dropna=False).to_dict()

    return profile
