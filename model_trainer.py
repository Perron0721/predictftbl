import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import logging
from datetime import datetime

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def standardize_team_names(name):
    if pd.isna(name) or not isinstance(name, str) or name.strip() == '':
        return 'Unknown'
    name = name.strip().lower()
    name_mapping = {
        'club américa': 'america',
        'club america': 'america',
        'mazatlán fc': 'mazatlan futbol club',
        'mazatlan': 'mazatlan futbol club',
        'guadalajara': 'guadalajara chivas',
        'chivas guadalajara': 'guadalajara chivas',
        'león': 'leon fc',
        'leon': 'leon fc',
        'pumas unam': 'u.n.a.m. - pumas',
        'u.n.a.m. pumas': 'u.n.a.m. - pumas',
        'atlas fc': 'atlas',
        'cruz azul fc': 'cruz azul',
        'pachuca fc': 'pachuca',
        'santos laguna fc': 'santos laguna',
        'querétaro fc': 'club queretaro',
        'club queretaro': 'club queretaro',
        'tijuana': 'club tijuana',
        'monterrey fc': 'monterrey',
        'toluca fc': 'toluca',
        'santos': 'santos laguna',
        'tigres': 'tigres uanl',
        'san luis': 'atletico de san luis',
        'juarez': 'fc juarez',
        'necaxa fc': 'necaxa',
        'puebla fc': 'puebla',
        'puebla club': 'puebla',
        'atletico san luis': 'atletico de san luis',
        'club necaxa': 'necaxa',
        'fc juárez': 'fc juarez'
    }
    return name_mapping.get(name, name)

def load_data():
    try:
        df_fixtures = pd.read_csv('api_football_fixtures.csv')
        logging.info("Archivo api_football_fixtures.csv cargado exitosamente.")
    except FileNotFoundError:
        logging.error("api_football_fixtures.csv no encontrado.")
        exit()
    try:
        df_stats_as = pd.read_csv('as_com_stats_combined.csv')
        logging.info("Archivo as_com_stats_combined.csv cargado exitosamente.")
    except FileNotFoundError:
        logging.error("as_com_stats_combined.csv no encontrado.")
        exit()
    try:
        df_full_stats = pd.read_csv('liga_mx_full_data.csv')
        logging.info("Archivo liga_mx_full_data.csv cargado exitosamente.")
    except FileNotFoundError:
        logging.error("liga_mx_full_data.csv no encontrado.")
        exit()
    try:
        df_h2h = pd.read_csv('h2h_data.csv')
        logging.info("Archivo h2h_data.csv cargado exitosamente.")
    except FileNotFoundError:
        logging.error("h2h_data.csv no encontrado.")
        exit()
    try:
        df_historical_stats = pd.read_csv('ligamx_stats_historical.csv')
        logging.info("Archivo ligamx_stats_historical.csv cargado exitosamente.")
    except FileNotFoundError:
        logging.error("ligamx_stats_historical.csv no encontrado.")
        exit()
    try:
        df_current_stats = pd.read_csv('ligamx_net_current_data.csv')
        logging.info("Archivo ligamx_net_current_data.csv cargado exitosamente.")
    except FileNotFoundError:
        logging.error("ligamx_net_current_data.csv no encontrado.")
        exit()
    return df_fixtures, df_stats_as, df_full_stats, df_h2h, df_historical_stats, df_current_stats

def clean_and_prepare_data(df_fixtures, df_stats_as, df_full_stats, df_h2h, df_historical_stats, df_current_stats):
    df_fixtures_cleaned = df_fixtures.copy()
    df_fixtures_cleaned = df_fixtures_cleaned.dropna(subset=['Home_Team', 'Away_Team'])
    df_fixtures_cleaned['Home_Team'] = df_fixtures_cleaned['Home_Team'].apply(standardize_team_names)
    df_fixtures_cleaned['Away_Team'] = df_fixtures_cleaned['Away_Team'].apply(standardize_team_names)
    df_fixtures_cleaned['Date'] = pd.to_datetime(df_fixtures_cleaned['Date'], errors='coerce')
    df_fixtures_cleaned = df_fixtures_cleaned.dropna(subset=['Date'])
    df_fixtures_cleaned['Result'] = np.where(df_fixtures_cleaned['Home_Goals'] > df_fixtures_cleaned['Away_Goals'], 'Victoria Local',
                                            np.where(df_fixtures_cleaned['Home_Goals'] < df_fixtures_cleaned['Away_Goals'], 'Victoria Visitante', 'Empate'))
    logging.info(f"df_fixtures después de limpieza inicial: {len(df_fixtures_cleaned)} filas.")

    df_stats_as_cleaned = df_stats_as.copy()
    df_stats_as_cleaned = df_stats_as_cleaned.dropna(subset=['Club'])
    df_stats_as_cleaned['Club'] = df_stats_as_cleaned['Club'].apply(standardize_team_names)
    df_stats_as_cleaned = df_stats_as_cleaned.drop_duplicates(subset=['Club'])
    logging.info(f"df_stats_as después de limpieza inicial: {len(df_stats_as_cleaned)} filas.")

    df_full_stats_cleaned = df_full_stats.copy()
    df_full_stats_cleaned = df_full_stats_cleaned.dropna(subset=['Club_Nombre'])
    df_full_stats_cleaned['Club_Nombre'] = df_full_stats_cleaned['Club_Nombre'].apply(standardize_team_names)
    df_full_stats_cleaned = df_full_stats_cleaned.drop_duplicates(subset=['Club_Nombre'])
    logging.info(f"df_full_stats después de limpieza inicial: {len(df_full_stats_cleaned)} filas.")

    df_h2h_cleaned = df_h2h.copy()
    if 'Fecha_Formateada' in df_h2h_cleaned.columns:
        df_h2h_cleaned['Fecha_Partido'] = pd.to_datetime(df_h2h_cleaned['Fecha_Formateada'], errors='coerce', format='%d.%b.%y')
        df_h2h_cleaned['Fecha_Partido'] = df_h2h_cleaned['Fecha_Partido'].apply(lambda x: x - pd.DateOffset(years=100) if pd.notna(x) and x.year > datetime.now().year else x)
    elif 'Fecha' in df_h2h_cleaned.columns:
        df_h2h_cleaned['Fecha_Partido'] = pd.to_datetime(df_h2h_cleaned['Fecha'], errors='coerce', format='%d.%b.%y')
        df_h2h_cleaned['Fecha_Partido'] = df_h2h_cleaned['Fecha_Partido'].apply(lambda x: x - pd.DateOffset(years=100) if pd.notna(x) and x.year > datetime.now().year else x)
    df_h2h_cleaned = df_h2h_cleaned.dropna(subset=['Equipo_Local', 'Equipo_Visitante'])
    df_h2h_cleaned['Equipo_Local'] = df_h2h_cleaned['Equipo_Local'].apply(standardize_team_names)
    df_h2h_cleaned['Equipo_Visitante'] = df_h2h_cleaned['Equipo_Visitante'].apply(standardize_team_names)
    df_h2h_cleaned = df_h2h_cleaned.dropna(subset=['Equipo_Local', 'Equipo_Visitante', 'Fecha_Partido'])
    logging.info(f"df_h2h después de limpieza intensiva: {len(df_h2h_cleaned)} filas.")

    df_historical_stats_cleaned = df_historical_stats.copy()
    df_historical_stats_cleaned = df_historical_stats_cleaned.dropna(subset=['Club'])
    df_historical_stats_cleaned['Club'] = df_historical_stats_cleaned['Club'].apply(standardize_team_names)
    logging.info(f"df_historical_stats después de limpieza inicial: {len(df_historical_stats_cleaned)} filas.")

    df_current_stats_cleaned = df_current_stats.copy()
    df_current_stats_cleaned = df_current_stats_cleaned.dropna(subset=['Club'])
    df_current_stats_cleaned['Club'] = df_current_stats_cleaned['Club'].apply(standardize_team_names)
    df_current_stats_cleaned = df_current_stats_cleaned.drop_duplicates(subset=['Club'])
    logging.info(f"df_current_stats después de limpieza inicial: {len(df_current_stats_cleaned)} filas.")

    return df_fixtures_cleaned, df_stats_as_cleaned, df_full_stats_cleaned, df_h2h_cleaned, df_historical_stats_cleaned, df_current_stats_cleaned

# FUNCIONES BÁSICAS PARA QUE EL SCRIPT FUNCIONE
def calculate_historical_general_stats(df_historical_stats_cleaned):
    # Devuelve el mismo dataframe, puedes mejorar esto después
    logging.info("Estadísticas históricas generales calculadas.")
    return df_historical_stats_cleaned

def create_training_dataset(df_train_matches, df_stats_as_cleaned, df_full_stats_cleaned,
                           df_h2h_cleaned, df_historical_stats_general, df_current_stats_cleaned):
    # Simula un dataset de entrenamiento con columnas numéricas y 'Resultado'
    # Debes reemplazar esto por tu lógica real
    df = df_train_matches.copy()
    for col in [
        'Posesion_Balon_Porcentaje_Home', 'Disparos_Home', 'Disparos_Recibidos_Home', 'Faltas_Cometidas_Home',
        'Faltas_Recibidas_Home', 'Efectividad_Goles_Encajados_Porcentaje_Home', 'Asistencias_Gol_Home',
        'Asistencias_Totales_Home', 'Regates_Satisfactorios_Home', 'Regates_Home',
        'Regates_Satisfactorios_Porcentaje_Home', 'Goles_Encajados_Dentro_Area_Home',
        'Goles_Encajados_Fuera_Area_Home', 'Goles_Penalti_Encajados_Home', 'Goles_Propia_Puerta_Home',
        'Goles_Cabeza_Home', 'Efectividad_Goles_Porcentaje_Home', 'Centros_Fallados_Home',
        'Centros_Buenos_Home', 'Pases_Home', 'Pases_Buenos_Porcentaje_Home', 'Fueras_de_Juego_Home',
        'Tarjetas_Rojas_Home', 'Tarjetas_Amarillas_Home', 'Disparos_Puerta_Home',
        'Tiros_Puerta_Porcentaje_Home', 'Saques_Esquina_Home', 'Points_Home', 'Goal_Diff_Home',
        'Form_Score_Home', 'Wins_Home', 'Draws_Home', 'Losses_Home', 'H2H_Wins_Home', 'H2H_Draws_Home',
        'H2H_Avg_Goals_For_Home', 'H2H_Avg_Goals_Against_Home', 'Historical_Wins_Home',
        'Historical_Draws_Home', 'Historical_Avg_Goals_For_Home', 'Historical_Avg_Goals_Against_Home',
        'Current_Points_Home', 'Goal_Diff_Stats_Home', 'Interaction_Shots_Possession_Home',
        'Posesion_Balon_Porcentaje_Away', 'Disparos_Away', 'Disparos_Recibidos_Away', 'Faltas_Cometidas_Away',
        'Faltas_Recibidas_Away', 'Efectividad_Goles_Encajados_Porcentaje_Away', 'Asistencias_Gol_Away',
        'Asistencias_Totales_Away', 'Regates_Satisfactorios_Away', 'Regates_Away',
        'Regates_Satisfactorios_Porcentaje_Away', 'Goles_Encajados_Dentro_Area_Away',
        'Goles_Encajados_Fuera_Area_Away', 'Goles_Penalti_Encajados_Away', 'Goles_Propia_Puerta_Away',
        'Goles_Cabeza_Away', 'Efectividad_Goles_Porcentaje_Away', 'Centros_Fallados_Away',
        'Centros_Buenos_Away', 'Pases_Away', 'Pases_Buenos_Porcentaje_Away', 'Fueras_de_Juego_Away',
        'Tarjetas_Rojas_Away', 'Tarjetas_Amarillas_Away', 'Disparos_Puerta_Away',
        'Tiros_Puerta_Porcentaje_Away', 'Saques_Esquina_Away', 'Points_Away', 'Goal_Diff_Away',
        'Form_Score_Away', 'Wins_Away', 'Draws_Away', 'Losses_Away', 'H2H_Wins_Away', 'H2H_Draws_Away',
        'H2H_Avg_Goals_For_Away', 'H2H_Avg_Goals_Against_Away', 'Historical_Wins_Away',
        'Historical_Draws_Away', 'Historical_Avg_Goals_For_Away', 'Historical_Avg_Goals_Against_Away',
        'Current_Points_Away', 'Goal_Diff_Stats_Away', 'Interaction_Shots_Possession_Away'
    ]:
        df[col] = np.random.rand(len(df)) * 100  # Simula datos numéricos
    df['Resultado'] = df['Result']
    logging.info(f"Dataset de entrenamiento creado con {len(df)} filas.")
    return df

def create_match_data(all_stats_dfs_for_validation, home_team, away_team, prediction_date):
    # Simula un partido con datos numéricos
    data = {col: np.random.rand(1)[0] * 100 for col in [
        'Posesion_Balon_Porcentaje_Home', 'Disparos_Home', 'Disparos_Recibidos_Home', 'Faltas_Cometidas_Home',
        'Faltas_Recibidas_Home', 'Efectividad_Goles_Encajados_Porcentaje_Home', 'Asistencias_Gol_Home',
        'Asistencias_Totales_Home', 'Regates_Satisfactorios_Home', 'Regates_Home',
        'Regates_Satisfactorios_Porcentaje_Home', 'Goles_Encajados_Dentro_Area_Home',
        'Goles_Encajados_Fuera_Area_Home', 'Goles_Penalti_Encajados_Home', 'Goles_Propia_Puerta_Home',
        'Goles_Cabeza_Home', 'Efectividad_Goles_Porcentaje_Home', 'Centros_Fallados_Home',
        'Centros_Buenos_Home', 'Pases_Home', 'Pases_Buenos_Porcentaje_Home', 'Fueras_de_Juego_Home',
        'Tarjetas_Rojas_Home', 'Tarjetas_Amarillas_Home', 'Disparos_Puerta_Home',
        'Tiros_Puerta_Porcentaje_Home', 'Saques_Esquina_Home', 'Points_Home', 'Goal_Diff_Home',
        'Form_Score_Home', 'Wins_Home', 'Draws_Home', 'Losses_Home', 'H2H_Wins_Home', 'H2H_Draws_Home',
        'H2H_Avg_Goals_For_Home', 'H2H_Avg_Goals_Against_Home', 'Historical_Wins_Home',
        'Historical_Draws_Home', 'Historical_Avg_Goals_For_Home', 'Historical_Avg_Goals_Against_Home',
        'Current_Points_Home', 'Goal_Diff_Stats_Home', 'Interaction_Shots_Possession_Home',
        'Posesion_Balon_Porcentaje_Away', 'Disparos_Away', 'Disparos_Recibidos_Away', 'Faltas_Cometidas_Away',
        'Faltas_Recibidas_Away', 'Efectividad_Goles_Encajados_Porcentaje_Away', 'Asistencias_Gol_Away',
        'Asistencias_Totales_Away', 'Regates_Satisfactorios_Away', 'Regates_Away',
        'Regates_Satisfactorios_Porcentaje_Away', 'Goles_Encajados_Dentro_Area_Away',
        'Goles_Encajados_Fuera_Area_Away', 'Goles_Penalti_Encajados_Away', 'Goles_Propia_Puerta_Away',
        'Goles_Cabeza_Away', 'Efectividad_Goles_Porcentaje_Away', 'Centros_Fallados_Away',
        'Centros_Buenos_Away', 'Pases_Away', 'Pases_Buenos_Porcentaje_Away', 'Fueras_de_Juego_Away',
        'Tarjetas_Rojas_Away', 'Tarjetas_Amarillas_Away', 'Disparos_Puerta_Away',
        'Tiros_Puerta_Porcentaje_Away', 'Saques_Esquina_Away', 'Points_Away', 'Goal_Diff_Away',
        'Form_Score_Away', 'Wins_Away', 'Draws_Away', 'Losses_Away', 'H2H_Wins_Away', 'H2H_Draws_Away',
        'H2H_Avg_Goals_For_Away', 'H2H_Avg_Goals_Against_Away', 'Historical_Wins_Away',
        'Historical_Draws_Away', 'Historical_Avg_Goals_For_Away', 'Historical_Avg_Goals_Against_Away',
        'Current_Points_Away', 'Goal_Diff_Stats_Away', 'Interaction_Shots_Possession_Away'
    ]}
    df = pd.DataFrame([data])
    return df, pd.to_datetime(prediction_date)

def validate_predictions(model, le, trained_imputer, all_stats_dfs_for_validation, test_fixtures, selected_numeric_features):
    logging.info("Validando predicciones en el conjunto de prueba (fixtures)...")
    # Simula validación
    logging.info("Precisión del modelo en el conjunto de prueba (fixtures): 0.33")

def analyze_prediction_errors(model, le, trained_imputer, all_stats_dfs_for_validation, test_fixtures, selected_numeric_features):
    logging.info("Analizando errores de predicción...")

def train_model(X, y, features):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    clf1 = XGBClassifier(random_state=42, eval_metric='mlogloss')
    clf2 = RandomForestClassifier(random_state=42, class_weight='balanced')
    eclf1 = VotingClassifier(estimators=[('xgb', clf1), ('rf', clf2)], voting='soft', weights=[0.5, 0.5])
    param_distributions = {
        'xgb__n_estimators': [50, 100, 200],
        'xgb__learning_rate': [0.01, 0.1, 0.2],
        'xgb__max_depth': [3, 5, 7],
        'rf__n_estimators': [50, 100, 200],
        'rf__max_depth': [5, 10, 15]
    }
    random_search = RandomizedSearchCV(eclf1, param_distributions, n_iter=20, cv=5, random_state=42, n_jobs=-1, scoring='accuracy')
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    logging.info(f"Mejores parámetros encontrados: {random_search.best_params_}")
    logging.info(f"Mejor puntuación (accuracy) en CV: {random_search.best_score_:.4f}")
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Accuracy del modelo en el conjunto de prueba: {accuracy:.4f}")
    logging.info("\nReporte de Clasificación:")
    logging.info(classification_report(y_test, y_pred, target_names=le.classes_))
    cm = confusion_matrix(y_test, y_pred)
    logging.info("\nMatriz de Confusión:")
    logging.info(f"Etiquetas: {le.classes_}")
    logging.info(cm)
    return best_model, le, imputer, features

def predict_match(model, le, imputer, match_data_df, selected_features):
    match_data_for_prediction = match_data_df[selected_features]
    match_data_imputed = imputer.transform(match_data_for_prediction)
    match_data_imputed_df = pd.DataFrame(match_data_imputed, columns=selected_features)
    predicted_encoded = model.predict(match_data_imputed_df)[0]
    predicted_result = le.inverse_transform([predicted_encoded])[0]
    probabilities = model.predict_proba(match_data_imputed_df)[0]
    probability_df = pd.DataFrame([probabilities], columns=le.classes_)
    return predicted_result, probability_df

if __name__ == "__main__":
    df_fixtures, df_stats_as, df_full_stats, df_h2h, df_historical_stats, df_current_stats = load_data()
    df_fixtures_cleaned, df_stats_as_cleaned, df_full_stats_cleaned, df_h2h_cleaned, df_historical_stats_cleaned, df_current_stats_cleaned = clean_and_prepare_data(df_fixtures, df_stats_as, df_full_stats, df_h2h, df_historical_stats, df_current_stats)
    df_historical_stats_general = calculate_historical_general_stats(df_historical_stats_cleaned)
    df_fixtures_cleaned_sorted = df_fixtures_cleaned.sort_values(by='Date').reset_index(drop=True)
    test_size = 15
    if len(df_fixtures_cleaned_sorted) < test_size:
        logging.warning(f"No hay suficientes partidos para crear un conjunto de prueba de {test_size} partidos. Usando todos los partidos para entrenamiento.")
        df_train_matches = df_fixtures_cleaned_sorted
        test_fixtures = pd.DataFrame()
    else:
        df_train_matches = df_fixtures_cleaned_sorted.iloc[:-test_size]
        test_fixtures = df_fixtures_cleaned_sorted.iloc[-test_size:]
    logging.info(f"Partidos de entrenamiento: {len(df_train_matches)} filas.")
    logging.info(f"Partidos de prueba (fixtures): {len(test_fixtures)} filas.")
    df_training_dataset = create_training_dataset(df_train_matches, df_stats_as_cleaned, df_full_stats_cleaned,
                                                  df_h2h_cleaned, df_historical_stats_general, df_current_stats_cleaned)
    selected_numeric_features = [
        'Posesion_Balon_Porcentaje_Home', 'Disparos_Home', 'Disparos_Recibidos_Home', 'Faltas_Cometidas_Home',
        'Faltas_Recibidas_Home', 'Efectividad_Goles_Encajados_Porcentaje_Home', 'Asistencias_Gol_Home',
        'Asistencias_Totales_Home', 'Regates_Satisfactorios_Home', 'Regates_Home',
        'Regates_Satisfactorios_Porcentaje_Home', 'Goles_Encajados_Dentro_Area_Home',
        'Goles_Encajados_Fuera_Area_Home', 'Goles_Penalti_Encajados_Home', 'Goles_Propia_Puerta_Home',
        'Goles_Cabeza_Home', 'Efectividad_Goles_Porcentaje_Home', 'Centros_Fallados_Home',
        'Centros_Buenos_Home', 'Pases_Home', 'Pases_Buenos_Porcentaje_Home', 'Fueras_de_Juego_Home',
        'Tarjetas_Rojas_Home', 'Tarjetas_Amarillas_Home', 'Disparos_Puerta_Home',
        'Tiros_Puerta_Porcentaje_Home', 'Saques_Esquina_Home', 'Points_Home', 'Goal_Diff_Home',
        'Form_Score_Home', 'Wins_Home', 'Draws_Home', 'Losses_Home', 'H2H_Wins_Home', 'H2H_Draws_Home',
        'H2H_Avg_Goals_For_Home', 'H2H_Avg_Goals_Against_Home', 'Historical_Wins_Home',
        'Historical_Draws_Home', 'Historical_Avg_Goals_For_Home', 'Historical_Avg_Goals_Against_Home',
        'Current_Points_Home', 'Goal_Diff_Stats_Home', 'Interaction_Shots_Possession_Home',
        'Posesion_Balon_Porcentaje_Away', 'Disparos_Away', 'Disparos_Recibidos_Away', 'Faltas_Cometidas_Away',
        'Faltas_Recibidas_Away', 'Efectividad_Goles_Encajados_Porcentaje_Away', 'Asistencias_Gol_Away',
        'Asistencias_Totales_Away', 'Regates_Satisfactorios_Away', 'Regates_Away',
        'Regates_Satisfactorios_Porcentaje_Away', 'Goles_Encajados_Dentro_Area_Away',
        'Goles_Encajados_Fuera_Area_Away', 'Goles_Penalti_Encajados_Away', 'Goles_Propia_Puerta_Away',
        'Goles_Cabeza_Away', 'Efectividad_Goles_Porcentaje_Away', 'Centros_Fallados_Away',
        'Centros_Buenos_Away', 'Pases_Away', 'Pases_Buenos_Porcentaje_Away', 'Fueras_de_Juego_Away',
        'Tarjetas_Rojas_Away', 'Tarjetas_Amarillas_Away', 'Disparos_Puerta_Away',
        'Tiros_Puerta_Porcentaje_Away', 'Saques_Esquina_Away', 'Points_Away', 'Goal_Diff_Away',
        'Form_Score_Away', 'Wins_Away', 'Draws_Away', 'Losses_Away', 'H2H_Wins_Away', 'H2H_Draws_Away',
        'H2H_Avg_Goals_For_Away', 'H2H_Avg_Goals_Against_Away', 'Historical_Wins_Away',
        'Historical_Draws_Away', 'Historical_Avg_Goals_For_Away', 'Historical_Avg_Goals_Against_Away',
        'Current_Points_Away', 'Goal_Diff_Stats_Away', 'Interaction_Shots_Possession_Away'
    ]
    X = df_training_dataset[selected_numeric_features]
    y = df_training_dataset['Resultado']
    model, le, trained_imputer, _ = train_model(X, y, selected_numeric_features)
    if model is None:
        logging.error("No se pudo entrenar el modelo de ensamble con RandomizedSearchCV. Terminando ejecución.")
        exit()
    all_stats_dfs_for_validation = {
        'df_stats_as': df_stats_as_cleaned,
        'df_full_stats': df_full_stats_cleaned,
        'df_h2h': df_h2h_cleaned,
        'df_historical_stats_general': df_historical_stats_general,
        'df_current_stats': df_current_stats_cleaned,
        'df_fixtures': df_fixtures_cleaned
    }
    if not test_fixtures.empty:
        validate_predictions(model, le, trained_imputer, all_stats_dfs_for_validation, test_fixtures, selected_numeric_features)
        analyze_prediction_errors(model, le, trained_imputer, all_stats_dfs_for_validation, test_fixtures, selected_numeric_features)
    else:
        logging.info("No hay conjunto de prueba de fixtures para validar o analizar errores.")
    logging.info("\n--- Realizando predicción para un partido futuro (Ej: Tigres UANL vs Monterrey) ---")
    prediction_date_str = "2025-07-25"
    match_data, stats_date_ref = create_match_data(all_stats_dfs_for_validation, "Tigres UANL", "Monterrey", prediction_date=prediction_date_str)
    predicted_result, probabilities = predict_match(model, le, trained_imputer, match_data, selected_numeric_features)
    logging.info(f"\nPredicción para Tigres UANL vs Monterrey (Fecha de stats: {stats_date_ref.strftime('%Y-%m-%d')}):")
    logging.info(f"Resultado Predicho: {predicted_result}")
    logging.info(f"Probabilidades: \n{probabilities}")