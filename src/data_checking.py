

import pandas as pd

class DatasetChecker:
    """
    Handles initial structural verification, data integrity checks, 
    and diagnostic reporting for the raw housing dataset.
    """
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe

    def run_all_checks(self):
        """Executes a full profile check of the dataset."""
        print("\n" + "="*40)
        print(" TASK 1: DATASET CHECKING AUDIT REPORT ")
        print("="*40)
        self.check_structure()
        self.check_missing_data()
        self.check_basic_statistics()
        print("="*40 + "\n")

    def check_structure(self):
        print("\n[1] DATASET STRUCTURE & TYPES:")
        print(f"Total Records (Rows): {self.df.shape[0]}")
        print(f"Total Features (Columns): {self.df.shape[1]}")
        print("\nColumn Data Types:")
        print(self.df.dtypes)

    def check_missing_data(self):
        print("\n[2] MISSING VALUES DETECTED:")
        missing_counts = self.df.isnull().sum()
        missing_percentages = 100 * (missing_counts / len(self.df))
        
        missing_table = pd.DataFrame({
            'Missing Values': missing_counts,
            'Percentage (%)': missing_percentages
        })
        

        active_missing = missing_table[missing_table['Missing Values'] > 0]
        if active_missing.empty:
            print("Excellent! No missing values found in the dataset.")
        else:
            print(active_missing)

    def check_basic_statistics(self):
        print("\n[3] NUMERIC SUMMARY STATISTICS:")

        numeric_cols = self.df.select_dtypes(include=['number']).columns
        if not numeric_cols.empty:
            print(self.df[numeric_cols].describe().T[['count', 'mean', 'min', 'max']])