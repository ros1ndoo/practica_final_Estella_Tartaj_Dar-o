import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

def main():
    os.makedirs('output', exist_ok=True)
    
    # 1. Carga y preprocesamiento de datos
    df = pd.read_csv('Data/energy_dataset.csv')
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df = df.sort_values('time').reset_index(drop=True)
    
    months = df['time'].dt.month
    season_map = {12: 'Invierno', 1: 'Invierno', 2: 'Invierno',
                  3: 'Primavera', 4: 'Primavera', 5: 'Primavera',
                  6: 'Verano', 7: 'Verano', 8: 'Verano',
                  9: 'Otoño', 10: 'Otoño', 11: 'Otoño'}
    df['season'] = months.map(season_map)
    df['day_type'] = df['time'].dt.dayofweek.apply(lambda x: 'Fin de Semana' if x >= 5 else 'Laborable')
    
    df = df.ffill().bfill()
    cols_to_drop = [c for c in df.columns if df[c].nunique() <= 1]
    df = df.drop(columns=cols_to_drop)
    
    target = 'price actual'
    X = df.drop(columns=[target, 'time'])
    y = df[target]
    
    cat_cols = ['season', 'day_type']
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nDefiniendo el pipeline (Codificación + Escalado + LinearRegression)...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first'), cat_cols)
        ])
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    # Métricas
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Resultados:\nMAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR2: {r2:.2f}")
    
    with open('output/ej2_metricas_regresion.txt', 'w', encoding='utf-8') as f:
        f.write("Métricas de la Regresión Lineal (Test Set)\n")
        f.write("="*40 + "\n")
        f.write(f"MAE:  {mae:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"R2:   {r2:.4f}\n")
        
    residuos = y_test - y_pred
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuos, alpha=0.3, color='blue', edgecolor='w', linewidths=0.5)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Valores Predichos')
    plt.ylabel('Residuos')
    plt.title('Gráfico de Residuos - Regresión Lineal')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig('output/ej2_residuos.png', dpi=150)
    plt.close()
    
if __name__ == '__main__':
    main()
