"""ADK wrappers around deterministic CSV profiling and analysis functions."""

from __future__ import annotations

from typing import Any

from google.adk.tools import ToolContext

from app.tools import (
    calculate_customer_metrics,
    calculate_product_metrics,
    calculate_sales_metrics,
    load_csv,
    profile_dataset,
    validate_schema,
)


def _csv_path_from_context(tool_context: ToolContext) -> str:
    csv_path = tool_context.state.get("csv_path")
    if not isinstance(csv_path, str) or not csv_path.strip():
        raise ValueError("CSV path is missing from session state under 'csv_path'.")
    return csv_path


def profile_uploaded_dataset(tool_context: ToolContext) -> dict[str, Any]:
    """Validate the uploaded sales CSV and return its profile and data-quality warnings."""
    try:
        dataframe = load_csv(_csv_path_from_context(tool_context))
        validation = validate_schema(dataframe)
        profile = profile_dataset(dataframe)
    except (FileNotFoundError, OSError, ValueError) as exc:
        return {"available": False, "error": str(exc)}

    return {
        "available": True,
        "validation": validation,
        "profile": profile,
    }


def analyze_sales(tool_context: ToolContext) -> dict[str, Any]:
    """Calculate deterministic sales totals, order values, regional revenue, and monthly trends."""
    try:
        dataframe = load_csv(_csv_path_from_context(tool_context))
        metrics = calculate_sales_metrics(dataframe)
    except (FileNotFoundError, OSError, ValueError) as exc:
        return {"available": False, "error": str(exc)}

    return {
        "available": True,
        "metrics": metrics,
        "data_quality_warnings": [],
    }


def analyze_products(tool_context: ToolContext) -> dict[str, Any]:
    """Calculate deterministic product, category, size, color, and gender performance metrics."""
    try:
        dataframe = load_csv(_csv_path_from_context(tool_context))
        metrics = calculate_product_metrics(dataframe)
    except (FileNotFoundError, OSError, ValueError) as exc:
        return {"available": False, "error": str(exc)}

    return {
        "available": True,
        "metrics": metrics,
        "data_quality_warnings": [],
    }


def analyze_customers(tool_context: ToolContext) -> dict[str, Any]:
    """Calculate deterministic customer counts, repeat rates, order frequency, and spend."""
    try:
        dataframe = load_csv(_csv_path_from_context(tool_context))
        metrics = calculate_customer_metrics(dataframe)
    except (FileNotFoundError, OSError, ValueError) as exc:
        return {"available": False, "error": str(exc)}

    return {
        "available": True,
        "metrics": metrics,
        "data_quality_warnings": [],
    }
