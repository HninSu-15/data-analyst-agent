from app.config import SAMPLE_DATA_PATH
from app.tools import calculate_product_metrics, load_csv


def test_calculate_product_metrics():
    df = load_csv(SAMPLE_DATA_PATH)
    result = calculate_product_metrics(df)

    assert "top_products_by_revenue" in result
    assert "top_products_by_quantity" in result
    assert "category_revenue" in result
    assert "size_popularity" in result
    assert "color_popularity" in result
    assert "gender_revenue" in result
    assert len(result["top_products_by_revenue"]) > 0
