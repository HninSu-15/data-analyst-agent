from app.config import SAMPLE_DATA_PATH
from app.tools import calculate_customer_metrics, load_csv


def test_calculate_customer_metrics():
    df = load_csv(SAMPLE_DATA_PATH)
    result = calculate_customer_metrics(df)

    assert "unique_customers" in result
    assert "returning_customers" in result
    assert "average_customer_spend" in result
    assert "top_customer_spend_values" in result
    assert "order_count_distribution" in result
    assert result["unique_customers"] > 0
