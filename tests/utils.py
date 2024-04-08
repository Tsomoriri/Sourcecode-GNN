import pytest
from utils import load_json_data
import json

# Test data
test_json_data = [
    {"func": "def test_func():"},
    {"func": "def another_func():"},
    {"func": "def third_func():"}
]
test_json_file = "test_data.json"

# Helper function to create a temporary JSON file
def create_test_json_file(data, file_path):
    with open(file_path, "w") as file:
        for item in data:
            json.dump(item, file)
            file.write("\n")

# Fixture to create and remove the temporary JSON file
@pytest.fixture
def temp_json_file(tmpdir):
    test_file_path = str(tmpdir / test_json_file)
    create_test_json_file(test_json_data, test_file_path)
    yield test_file_path
    tmpdir.remove()

def test_load_json_data(temp_json_file):
    data = load_json_data(temp_json_file)
    assert data == test_json_data

