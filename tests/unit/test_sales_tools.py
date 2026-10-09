from app.config import SAMPLE_DATA_PATH
from app.tools import calculate_sales_metrics, load_csv


def test_calculate_sales_metrics():
    df = load_csv(SAMPLE_DATA_PATH)
    result = calculate_sales_metrics(df)

    assert "total_net_sales" in result
    assert "total_units_sold" in result
    assert "completed_orders" in result
    assert "average_order_value" in result
    assert "revenue_by_region" in result
    assert "monthly_sales_trend" in result
    assert result["completed_orders"] > 0
    assert result["total_units_sold"] > 0
