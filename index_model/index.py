# index_model/index.py

import pandas as pd
import numpy as np
from .index_utils import (
    calculate_eom_flag,
    calculate_daily_returns,
    calculate_index_return,
    calculate_index_level
)


class IndexModel:
    def __init__(self):
        """
        Initializes the IndexModel by loading data, computing EOM flags,
        identifying stock columns, assigning weights, and calculating daily returns.
        """
        self.stock_prices_csv_path = 'data_sources/stock_prices.csv'
        self.df = pd.read_csv(self.stock_prices_csv_path)

        # Convert 'Date' column to datetime and set as index
        self.df['Date'] = pd.to_datetime(self.df['Date'], format='%d/%m/%Y')
        self.df = self.df.set_index('Date')

        # Flag end-of-month rows
        self.df = calculate_eom_flag(self.df)

        # Identify stock columns dynamically
        self.stocks = [col for col in self.df.columns if col.startswith('Stock_')]

        # Assign monthly weights based on EOM ranking
        self.df = self.calculate_weights(self.df, self.stocks)

        # Calculate daily returns for each stock
        self.df = calculate_daily_returns(self.df, self.stocks)

    def calculate_weights(self, df, stocks):
        """
        Calculates and assigns monthly weights based on the EOM ranking strategy.
        Uses a nested helper function to initialize weights.
        """
        df = df.copy()

        # --- Nested Helper Function ---
        def initialize_weights(df, stocks):
            """Initializes weight columns for all stocks to 0.0"""
            for stock in stocks:
                df[f'{stock}_Weight'] = 0.0
            return df
        # --- End of Helper Function ---

        # Initialize all weight columns
        df = initialize_weights(df, stocks)
        weight_cols = [f'{stock}_Weight' for stock in stocks]

        # Define rebalancing periods using EOM flags
        df['Period'] = df['EOM_Flag'].cumsum()
        rebalance_rows = df[df['EOM_Flag'] == 1]

        # Assign weights for each rebalancing period
        for idx, row in rebalance_rows.iterrows():
            period = row['Period']
            prices = row[stocks].astype(float)
            ranked = prices.sort_values(ascending=False)
            top_3 = ranked.index[:3]

            period_mask = df['Period'] == period
            df.loc[period_mask, weight_cols] = 0

            if len(top_3) > 0:
                df.loc[period_mask, f'{top_3[0]}_Weight'] = 0.5
            if len(top_3) > 1:
                df.loc[period_mask, f'{top_3[1]}_Weight'] = 0.25
            if len(top_3) > 2:
                df.loc[period_mask, f'{top_3[2]}_Weight'] = 0.25

        # Drop temporary 'Period' column
        df = df.drop('Period', axis=1)
        return df

    def calc_index_level(self, start_date, end_date):
        """
        Computes the index level between start_date and end_date.
        Applies T-2 weighted returns and compounds them from base level 100.
        """
        self.df = calculate_index_return(self.df, self.stocks)
        self.df = calculate_index_level(self.df, start_date=start_date)
        self.result_df = self.df.loc[start_date:end_date, ['Index_Level']]
        return self.result_df

    def export_values(self, output_filename):
        """
        Exports the calculated index levels to a CSV file.
        """
        self.result_df.to_csv(output_filename)
        print(f"Index values exported to {output_filename}")

            
            
            
            
            
            
            