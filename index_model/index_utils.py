# index_model/index_utils.py

import pandas as pd
import numpy as np

# ==================================================================

def calculate_eom_flag(df):
    """
    Flags the last trading day of each month in the DataFrame.
    """
    df = df.copy()
    df['EOM_Flag'] = 0

    for period, group in df.groupby(pd.Grouper(freq='M')):
        if len(group) > 0:
            last_date = group.index.max()
            df.loc[last_date, 'EOM_Flag'] = 1

    return df

# ==================================================================

def calculate_index_return(df, stocks):
    """
    Calculates index daily returns using weights from T-2 (two days prior).
    """
    df = df.copy()
    return_cols = [f'{stock}_Return' for stock in stocks]
    weight_cols = [f'{stock}_Weight' for stock in stocks]
    df['Index_Return'] = 0.0

    # Ensure return columns exist
    for col in return_cols:
        if col not in df.columns:
            raise KeyError(f"Column {col} not found in DataFrame. Calculate daily returns first.")

    # Apply T-2 weighted return logic
    for i in range(2, len(df)):
        current_date = df.index[i]
        t_minus_2_date = df.index[i - 2]

        weights = df.loc[t_minus_2_date, weight_cols].values
        current_returns = df.loc[current_date, return_cols].values
        df.loc[current_date, 'Index_Return'] = np.dot(weights, current_returns)

    # Set first two days to zero (no valid T-2 weights)
    df.loc[df.index[:2], 'Index_Return'] = 0.0
    return df

# ==================================================================

def calculate_index_level(df, start_date=None):
    """
    Calculates the index level time series, starting at 100.
    """
    df = df.copy()
    df['Index_Level'] = 100.0
    growth_factors = 1 + df['Index_Return']

    # Determine starting index for compounding
    if start_date is None:
        start_idx = 0
    else:
        start_idx = df.index.get_indexer([start_date], method='bfill')[0]
        if start_idx == -1:
            start_idx = len(df) - 1

    # Set base level
    df.iloc[start_idx, df.columns.get_loc('Index_Level')] = 100.0

    # Compound returns forward
    for i in range(start_idx + 1, len(df)):
        prev_level = df.iloc[i - 1]['Index_Level']
        growth = growth_factors.iloc[i]
        df.iloc[i, df.columns.get_loc('Index_Level')] = prev_level * growth

    return df

# ==================================================================

def calculate_daily_returns(df, stocks):
    """
    Calculates daily percentage returns for each stock in the list.
    """
    df = df.copy()

    for stock in stocks:
        return_col = f'{stock}_Return'
        df[return_col] = df[stock].pct_change()

    return_cols = [f'{stock}_Return' for stock in stocks]
    df[return_cols] = df[return_cols].fillna(0)

    return df
