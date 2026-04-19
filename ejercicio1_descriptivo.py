import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_and_clean_data(file_path):
    """
    Carga y limpia el dataset, preparando las variables categóricas
    basadas en la columna de tiempo y gestionando valores nulos.
    """
    df = pd.read_csv(file_path)
    
    # Convertir 'time' a datetime
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df = df.sort_values('time').reset_index(drop=True)
    
    # Crear variables categóricas basadas en la columna de tiempo
    # 1. Estación del año 
    months = df['time'].dt.month
    season_map = {12: 'Invierno', 1: 'Invierno', 2: 'Invierno',
                  3: 'Primavera', 4: 'Primavera', 5: 'Primavera',
                  6: 'Verano', 7: 'Verano', 8: 'Verano',
                  9: 'Otoño', 10: 'Otoño', 11: 'Otoño'}
    df['season'] = months.map(season_map)
    
    # 2. Tipo de Día: Laborable vs Fin de Semana (day_type)
    df['day_type'] = df['time'].dt.dayofweek.apply(
        lambda x: 'Fin de Semana' if x >= 5 else 'Laborable'
    )
    
    # Tratamiento de nulos: al ser Serie Temporal de intervalos constantes,
    df = df.ffill().bfill()
    
    # Eliminar columnas irrelevantes o constantes 
    cols_to_drop = [c for c in df.columns if df[c].nunique() <= 1]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        
    return df

def main():
    # Crear carpeta output si no existe
    os.makedirs('output', exist_ok=True)
    
    # 1. Cargar datos
    dataset_path = 'Data/energy_dataset.csv'
    df = load_and_clean_data(dataset_path)
    target = 'price actual'
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = ['season', 'day_type']
    
    print(f"Filas: {df.shape[0]}")
    print(f"Columnas Totales: {df.shape[1]}")
    print(f"Variables Numéricas ({len(num_cols)}):", num_cols[:5], "...")
    print(f"Variables Categóricas ({len(cat_cols)}):", cat_cols)
    
    # 2. Estadísticos Descriptivos
    stats = df[num_cols].describe().T
    stats['IQR'] = stats['75%'] - stats['25%']
    stats['skewness'] = df[num_cols].skew()
    stats['kurtosis'] = df[num_cols].kurtosis()
    stats.to_csv('output/ej1_descriptivo.csv')
    
    # Detección de Outliers IQR en Variable Target
    Q1 = df[target].quantile(0.25)
    Q3 = df[target].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[target] < lower_bound) | (df[target] > upper_bound)]
    perc_outliers = (len(outliers) / len(df)) * 100
    
    with open('output/ej1_outliers.txt', 'w', encoding='utf-8') as f:
        f.write(f"Detección de outliers (Target: {target})\n")
        f.write("="*40 + "\n")
        f.write(f"Método utilizado: Rango Intercuartílico (IQR)\n")
        f.write(f"Límite inferior: {lower_bound:.2f}\n")
        f.write(f"Límite superior: {upper_bound:.2f}\n")
        f.write(f"Nº de outliers detectados: {len(outliers)} ({perc_outliers:.2f}% del total)\n")
    
    # 3. Distribuciones
    
    # Histogramas para variables numéricas
    n_cols_plot = 5
    n_rows_plot = int(np.ceil(len(num_cols) / n_cols_plot))
    
    fig = plt.figure(figsize=(20, 4 * n_rows_plot))
    for i, col in enumerate(num_cols, 1):
        ax = fig.add_subplot(n_rows_plot, n_cols_plot, i)
        sns.histplot(df[col], bins=30, ax=ax, kde=False)
        ax.set_title(col, fontsize=10)
        ax.set_ylabel('')
        ax.set_xlabel('')
    plt.tight_layout()
    plt.savefig('output/ej1_histogramas.png', dpi=150)
    plt.close()
    
    # Boxplots del target segmentado por variables categóricas
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.boxplot(data=df, x='season', y=target, ax=axes[0], hue='season', legend=False)
    axes[0].set_title(f'Distribución de {target} por Estación')
    
    sns.boxplot(data=df, x='day_type', y=target, ax=axes[1], hue='day_type', legend=False)
    axes[1].set_title(f'Distribución de {target} por Tipo de Día')
    
    plt.tight_layout()
    plt.savefig('output/ej1_boxplots.png', dpi=150)
    plt.close()
    
    # 4. Variables categóricas
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Frecuencias relativas y gráficos
    sns.countplot(data=df, x='season', ax=axes[0], hue='season', order=['Primavera', 'Verano', 'Otoño', 'Invierno'], legend=False)
    axes[0].set_title('Frecuencia de las Estaciones')
    
    sns.countplot(data=df, x='day_type', ax=axes[1], hue='day_type', legend=False)
    axes[1].set_title('Frecuencia del Tipo de Día')
    
    plt.tight_layout()
    plt.savefig('output/ej1_categoricas.png', dpi=150)
    plt.close()
    
    # 5. Correlaciones
    corr_matrix = df[num_cols].corr()
    
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, fmt=".2f")
    plt.title('Mapa de Calor - Correlaciones Lineales Numéricas')
    plt.tight_layout()
    plt.savefig('output/ej1_heatmap_correlacion.png', dpi=150)
    plt.close()
    
    # Top 3 correlaciones absolutas con la variable target
    target_corr = corr_matrix[target].drop(target).abs().sort_values(ascending=False)
    for i, (var, val) in enumerate(target_corr.head(3).items(), 1):
        signo = corr_matrix.loc[target, var]
        print(f"{i}. {var}: {signo:.4f} (abs: {val:.4f})")
    
if __name__ == '__main__':
    main()
