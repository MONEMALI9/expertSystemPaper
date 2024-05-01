import matplotlib.pyplot as plt
import pandas as pd

laliga_df_processed = pd.read_csv('playerstatsProcessed.csv',delimiter=',')

def statisticsMeasures(measures):
    
    if measures == 'std':
        # Calculate standard deviation values for each column
        std_values = laliga_df_processed[['Age', 'MP', 'Starts', 'Min', '90s', 'Gls', 'Ast', 'CrdY', 'CrdR']].std()
        
        # Plotting
        plt.figure(figsize=(10, 6))
        std_values.plot(kind='bar', color='skyblue')
        plt.title('Std Values of Selected Columns')
        plt.xlabel('Columns')
        plt.ylabel('Std Value')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    elif measures == 'mean':
        # Calculate mean values for each column
        mean_values = laliga_df_processed[['Age', 'MP', 'Starts', 'Min', '90s', 'Gls', 'Ast', 'CrdY', 'CrdR']].mean()

        # Plotting
        plt.figure(figsize=(10, 6))
        mean_values.plot(kind='bar', color='skyblue')
        plt.title('Mean Values of Selected Columns')
        plt.xlabel('Columns')
        plt.ylabel('Mean Value')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    elif measures == 'median':
        # Calculate mean values for each column
        median_values = laliga_df_processed[['Age', 'MP', 'Starts', 'Min', '90s', 'Gls', 'Ast', 'CrdY', 'CrdR']].median()

        # Plotting
        plt.figure(figsize=(10, 6))
        median_values.plot(kind='bar', color='skyblue')
        plt.title('median Values of Selected Columns')
        plt.xlabel('Columns')
        plt.ylabel('median Value')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
