import pytest
import sys
sys.path.append('..')
from src.eda_utils import load_data

def test_load_data():
    df = load_data('insurance_data.csv')
    assert df is not None
    assert 'TotalPremium' in df.columns
    assert 'TotalClaims' in df.columns
