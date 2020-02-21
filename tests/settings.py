"""Settings for the tests package."""

# fixture locations

# CsvTask fixture
CSV_TASK_SPLIT_FIXTURES = {
    "train": "fixtures/csv-task/train.test-preprocessed.csv",
    "validation": "fixtures/csv-task/validation.test-preprocessed.csv",
}
"""A dictionary mapping split names to fixtures for testing CsvTask."""

CSV_TASK_SPLIT_NUM_EXAMPLES = {"train": 5, "validation": 5}
"""A dictionary mapping split names to their numbers of examples."""
