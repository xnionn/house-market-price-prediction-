# src/preprocessing.py
import pandas as pd
import numpy as np

class DataPreprocessor:
    """
    Handles advanced cleanups, outlier detection, and safety filters 
    on the data stream before it passes to the ML regressors.
    """
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()

    def remove_outliers_iqr(self, columns: list, factor: float = 1.5) -> pd.DataFrame:
        """
        Filters extreme anomalies using the Interquartile Range (IQR) method.
        Prevents skewed metrics on volatile variables like total price.
        """
        initial_count = len(self.df)
        
        for col in columns:
            if col in self.df.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - (factor * IQR)
                upper_bound = Q3 + (factor * IQR)
                
                
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        
        final_count = len(self.df)
        print(f"[Preprocessing] Removed {initial_count - final_count} statistical outlier rows.")
        return self.df

    def process_boolean_features(self) -> pd.DataFrame:
        """Converts explicit boolean types to safe binary integers (0/1) for models."""
        bool_cols = self.df.select_dtypes(include=['bool']).columns
        for col in bool_cols:
            self.df[col] = self.df[col].astype(int)
        return self.df