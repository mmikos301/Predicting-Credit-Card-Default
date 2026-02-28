import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(data_dir):
    df = pd.read_csv(data_dir)
    # Zmiana nazwy dla spójności 
    df.rename(columns={'PAY_0': 'PAY_1'}, inplace=True)
    return df

def check_data_integrity(df):
    print("\n=== RAPORT INTEGRALNOŚCI DANYCH ===")
    # Techniczne braki
    print(f"Brakujące wartości (NaN): {df.isnull().sum().sum()}")
    # Sprawdzenie kategorii w Education
    print("\nRozkład kategorii w EDUCATION (przed czyszczeniem):")
    print(df['EDUCATION'].value_counts().sort_index())
    # Sprawdzenie duplikatów
    print(f"\nLiczba zduplikowanych wierszy: {df.duplicated().sum()}")

def preprocess_data(df):
    # 1. Naprawa kategorii Education (0, 5, 6 -> 4 czyli 'Others')
    df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
    # 2. Naprawa kategorii Marriage (0 -> 3 czyli 'Others')
    df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})
    # 3. Inżynieria cech: Utilization Ratio
    df['utilization_ratio'] = df['BILL_AMT1'] / df['LIMIT_BAL']
    return df

def make_corr_matrix(df, ax, title):
    # Rysujemy heatmapę na konkretnym obszarze (ax)
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax, cbar=False)
    ax.set_title(title)

if __name__ == "__main__":
    # Ścieżka do danych
    DATA_DIR = os.path.join('data', 'UCI_Credit_Card.csv')
    
    if not os.path.exists(DATA_DIR):
        print(f"Błąd: Nie znaleziono pliku w {DATA_DIR}! Upewnij się, że go tam skopiowałeś.")
    else:
        # 1. Wczytanie i audyt
        data = load_data(DATA_DIR)
        check_data_integrity(data)
        
        # 2. Czyszczenie i Feature Engineering
        data = preprocess_data(data)
        
        # 3. Definicja grup do analizy korelacji
        feature_groups = {
            "Payment_Status": ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'default.payment.next.month'],
            "Exposure_and_LTV": ['LIMIT_BAL', 'BILL_AMT1', 'utilization_ratio', 'default.payment.next.month'],
            "Amounts": ['BILL_AMT1', 'PAY_AMT1', 'BILL_AMT2', 'PAY_AMT2', 'default.payment.next.month']
        }
        
        # 4. Wizualizacja (Subplots)
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        for i, (group_name, cols) in enumerate(feature_groups.items()):
            make_corr_matrix(data[cols], ax=axes[i], title=group_name)
            
        plt.tight_layout()
        
        # 5. Zapisywanie i wyświetlanie
        PLOTS_DIR = 'plots'
        if not os.path.exists(PLOTS_DIR):
            os.makedirs(PLOTS_DIR)
            
        plt.savefig(os.path.join(PLOTS_DIR, 'correlation_matrices.png'), dpi=300, bbox_inches='tight')
        print(f"\nSukces! Wykres zapisany w folderze {PLOTS_DIR}")
        plt.show()