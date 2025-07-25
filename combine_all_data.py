import pandas as pd

# 📥 Cargar el archivo base
fixtures = pd.read_csv('api_football_fixtures.csv')

# 🛠 Función para cargar y preparar un archivo por columnas de equipo
def cargar_datos_equipo(ruta, columna_equipo_original):
    df = pd.read_csv(ruta)
    df.rename(columns={columna_equipo_original: 'Equipo'}, inplace=True)
    return df

# 📦 Cargar datasets adicionales
stats_as_com = cargar_datos_equipo('as_com_stats_combined.csv', 'Club')
ligamx_net = cargar_datos_equipo('ligamx_net_current_data.csv', 'Club')
ligamx_stats = cargar_datos_equipo('ligamx_stats.csv', 'Club')
liga_mx_full = cargar_datos_equipo('liga_mx_full_data.csv', 'Club_Nombre')

# 🧠 Función para unir stats de un equipo al fixture
def unir_stats(base_df, stats_df, columna_equipo):
    # Eliminar columna 'Equipo' si ya existe para evitar colisiones
    if 'Equipo' in stats_df.columns:
        stats_df = stats_df.drop(columns=['Equipo'])
    return base_df.merge(stats_df, left_on=columna_equipo, right_index=True, how='left')

# 💡 Reindexear los DataFrames para usar 'Equipo' como índice en el merge
for df in [stats_as_com, ligamx_net, ligamx_stats, liga_mx_full]:
    df.set_index('Equipo', inplace=True)

# 🔗 Unir estadísticas al equipo local
fixtures = unir_stats(fixtures, stats_as_com, 'Home_Team')
fixtures = unir_stats(fixtures, ligamx_net, 'Home_Team')
fixtures = unir_stats(fixtures, ligamx_stats, 'Home_Team')
fixtures = unir_stats(fixtures, liga_mx_full, 'Home_Team')

# 🔗 Unir estadísticas al equipo visitante
for df in [stats_as_com, ligamx_net, ligamx_stats, liga_mx_full]:
    df_visit = df.copy()
    # Renombrar columnas para distinguir equipo visitante
    df_visit.columns = ['Away_' + col for col in df_visit.columns]
    fixtures = fixtures.merge(df_visit, left_on='Away_Team', right_index=True, how='left')

# 🧼 Limpieza final
fixtures.fillna(0, inplace=True)
fixtures.drop_duplicates(inplace=True)

# 💾 Guardar archivo combinado
fixtures.to_csv('datos_entrenamiento.csv', index=False)
print("✅ Archivo combinado guardado como: datos_entrenamiento.csv")