from app.config import SAMPLE_DATA_PATH
from app.tools import load_csv, profile_dataset, validate_schema


def test_load_csv_and_validate_schema():
    df = load_csv(SAMPLE_DATA_PATH)
    assert len(df) > 0

    schema_result = validate_schema(df)
    assert schema_result["valid"] is True
    assert schema_result["missing_columns"] == []

    profile = profile_dataset(df)
    assert profile["row_count"] == len(df)
    assert profile["column_count"] == len(df.columns)


def test_profile_reports_missing_values_and_duplicates():
    df = load_csv(SAMPLE_DATA_PATH)
    profile = profile_dataset(df)

    assert "duplicate_rows" in profile
    assert "missing_values" in profile
    assert "numeric_ranges" in profile
