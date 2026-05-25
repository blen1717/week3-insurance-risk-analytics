import pandas as pd

def load_data(filepath):
    """Load CSV with error handling and column validation."""
    try:
        df = pd.read_csv(filepath)
        required = ['TotalPremium', 'TotalClaims', 'Province', 'Gender']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def compute_loss_ratio(df, group_col):
    """Compute loss ratio by a grouping column."""
    return df.groupby(group_col).apply(lambda x: x['TotalClaims'].sum() / x['TotalPremium'].sum(), include_groups=False)
