# src/eda_plots.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class EDAVisualizer:
    """
    Generates exploratory data visualization metrics and saves 
    distribution assets directly to your project workspace.
    """
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        
        sns.set_theme(style="whitegrid")

    def generate_all_plots(self):
        """Generates and updates project visualization visual assets."""
        print("[EDA] Generating Target Variable Distribution plot...")
        self.plot_target_distribution()
        
        print("[EDA] Generating Area vs Price Correlation plot...")
        self.plot_area_vs_price()
        print("[EDA] Visualization complete. Images updated successfully.")

    def plot_target_distribution(self):
        
        target_col = self.df.columns[0]
        
        plt.figure(figsize=(10, 5))
        sns.histplot(self.df[target_col], kde=True, color='teal', bins=40)
        plt.title('Distribution of Housing Prices (Target Metric)', fontsize=14, pad=15)
        plt.xlabel('Price (KZT)', fontsize=11)
        plt.ylabel('Frequency Density', fontsize=11)
        plt.ticklabel_format(style='plain', axis='x') 
        plt.tight_layout()
        plt.savefig('eda_price_distribution.png', dpi=150)
        plt.close()

    def plot_area_vs_price(self):
        target_col = self.df.columns[0]
        
        area_col = next((col for col in self.df.columns if 'площадь' in col), None)
        
        if area_col and area_col in self.df.columns:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=self.df, x=area_col, y=target_col, alpha=0.6, color='royalblue')
            sns.regplot(data=self.df, x=area_col, y=target_col, scatter=False, color='red', line_kws={"linewidth": 2})
            plt.title('Housing Price Analysis vs. Living Area (Square Meters)', fontsize=14, pad=15)
            plt.xlabel('Total Area (sq. m)', fontsize=11)
            plt.ylabel('Price (KZT)', fontsize=11)
            plt.ticklabel_format(style='plain', axis='y')
            plt.tight_layout()
            plt.savefig('eda_area_vs_price.png', dpi=150)
            plt.close()