import pandas as pd
import joblib
import numpy as np

# 📂 Cargar modelo entrenado
modelo = joblib.load('modelo_entrenado.joblib')
print("✅ Modelo cargado exitosamente.")

# 📈 Cargar datos de partidos futuros
df_futuro = pd.read_csv('partidos_futuros.csv')
print(f"📊 Dataset de predicción cargado con {df_futuro.shape[0]} registros.")

# 🧠 Crear columnas necesarias si no existen
df_futuro['GF_Gral_x'] = df_futuro.get('GF_Gral_x', df_futuro.get('Home_Goals', 0))
df_futuro['Away_GF_Gral'] = df_futuro.get('Away_GF_Gral', df_futuro.get('Away_Goals', 0))
df_futuro['GC_Gral_x'] = df_futuro.get('GC_Gral_x', 0)
df_futuro['PTS_Gral_x'] = df_futuro.get('PTS_Gral_x', 0)
df_futuro['Pos_x'] = df_futuro.get('Pos_x', 0)
df_futuro['Away_GC_Gral'] = df_futuro.get('Away_GC_Gral', 0)
df_futuro['Away_PTS_Gral'] = df_futuro.get('Away_PTS_Gral', 0)
df_futuro['Away_Pos'] = df_futuro.get('Away_Pos', 0)
df_futuro['GF_ult3'] = df_futuro.get('GF_ult3', df_futuro['GF_Gral_x'])
df_futuro['racha_victorias'] = df_futuro.get('racha_victorias', 0)

# 📋 Features del modelo
features = [
    'GF_Gral_x', 'GC_Gral_x', 'PTS_Gral_x', 'Pos_x',
    'Away_GF_Gral', 'Away_GC_Gral', 'Away_PTS_Gral', 'Away_Pos',
    'GF_ult3', 'racha_victorias'
]

X_nuevo = df_futuro[features].fillna(0)

# 🔮 Predicción y probabilidad
probabilidades = modelo.predict_proba(X_nuevo)[:, 1]
df_futuro['Probabilidad_Local_Gana'] = np.round(probabilidades, 2)
df_futuro['Predicción'] = (df_futuro['Probabilidad_Local_Gana'] > 0.5)
df_futuro['Resultado_Estimado'] = df_futuro['Predicción'].apply(lambda x: 'Gana Local' if x else 'No Gana Local')

# 📊 Mostrar resultados
print("\n🔮 Predicciones de partidos futuros con probabilidades:")
print(df_futuro[['Home_Team', 'Away_Team', 'Probabilidad_Local_Gana', 'Resultado_Estimado']])