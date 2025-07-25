import pandas as pd
import requests
from bs4 import BeautifulSoup
import traceback
import json
import datetime
import re
from io import StringIO
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

class DataScraper:
    def __init__(self):
        # URLs para las diferentes fuentes de datos
        self.ligamx_url = "https://ligamx.net/cancha/estadisticahistorica"
        self.threesixtyfive_url = "https://www.365scores.com/es-mx/football/league/liga-mx-141/stats"
        
        # URLs para tablas históricas de Liga MX
        self.ligamx_historical_urls = {
            'Clausura 2024-2025': 'https://ligamx.net/cancha/estadisticahistorica/1/eyJpZERpdmlzaW9uIjoiMSIsImlkVGVtcG9yYWRhIjoiNzUiLCAiaWRUb3JuZW8iOiIyIn0=',
            'Apertura 2024-2025': 'https://ligamx.net/cancha/estadisticahistorica/1/eyJpZERpdmlzaW9uIjoiMSIsImlkVGVtcG9yYWRhIjoiNzUiLCAiaWRUb3JuZW8iOiIxIn0=',
            'Clausura 2023-2024': 'https://ligamx.net/cancha/estadisticahistorica/1/eyJpZERpdmlzaW9uIjoiMSIsImlkVGVtcG9yYWRhIjoiNzQiLCAiaWRUb3JuZW8iOiIyIn0=',
            'Apertura 2023-2024': 'https://ligamx.net/cancha/estadisticahistorica/1/eyJpZERpdmlzaW9uIjoiMSIsImlkVGVtcG9yYWRhIjoiNzQiLCAiaWRUb3JuZW8iOiIxIn0='
        }
        
        # DICCIONARIO DE TODAS LAS ESTADÍSTICAS DE AS.COM
        self.as_stats_urls = {
            "Posesion_Balon_Porcentaje": "https://mexico.as.com/resultados/futbol/mexico_apertura/2025/ranking/equipos/porcentaje-de-posesion/",
            "Disparos": "https://mexico.as.com/resultados/futbol/mexico_apertura/2025/ranking/equipos/disparos/",
            "Disparos_Recibidos": "https://mexico.as.com/resultados/futbol/mexico_apertura/2025/ranking/equipos/disparos-recibidos/",
            "Faltas_Cometidas": "https://mexico.as.com/resultados/futbol/mexico_apertura/2025/ranking/equipos/faltas-cometidas/",
            "Faltas_Recibidas": "https://mexico.as.com/resultados/futbol/mexico_apertura/2025/ranking/equipos/faltas-recibidas/",
            "Efectividad_Goles_Encajados_Porcentaje": "https://mexico.as.com/resultados/futbol/mexico_apertura/2025/ranking/equipos/porcentaje-de-efectividad-de-goles-encajados/",
            "Asistencias_Gol": "https://mexico.as.com/resultados/futbol/mexico_apertura/2025/ranking/equipos/asistencias-de-gol/",
            "Asistencias_Totales": "https://mexico.as.com/resultados/futbol/mexico_apertura/2025/ranking/equipos/asistencias-totales/",
            "Regates_Satisfactorios": "https://mexico.as.com/resultados/futbol/mexico_apertura/2025/ranking/equipos/regates-satisfactorios/",
            "Regates": "https://mexico.as.com/resultados/futbol/mexico_apertura/2025/ranking/equipos/regates/",
            "Regates_Satisfactorios_Porcentaje": "https://mexico.as.com/resultados/futbol/mexico_apertura/2025/ranking/equipos/porcentaje-de-regates-satisfactorios/",
            "Goles_Encajados_Dentro_Area": "https://mexico.as.com/resultados/futbol/mexico_apertura/2025/ranking/equipos/goles-encajados-desde-dentro-del-area/",
            "Goles_Encajados_Fuera_Area": "https://mexico.as.com/resultados/futbol/mexico_apertura/2025/ranking/equipos/goles-encajados-desde-fuera-del-area/",
            "Goles_Penalti_Encajados": "https://mexico.as.com/resultados/futbol/mexico_apertura/2025/ranking/equipos/goles-encajados-de-penalti/",
            "Goles_Propia_Puerta": "https://mexico.as.com/resultados/futbol/mexico_apertura/2025/ranking/equipos/goles-en-propia-puerta/",
            "Goles_Cabeza": "https://mexico.as.com/resultados/futbol/mexico_apertura/2025/ranking/equipos/goles-con-la-cabeza/",
            "Efectividad_Goles_Porcentaje": "https://mexico.as.com/resultados/futbol/mexico_apertura/2025/ranking/equipos/porcentaje-de-efectividad-de-goles/",
            "Centros_Fallados": "https://mexico.as.com/resultados/futbol/mexico_apertura/2025/ranking/equipos/centros-fallados/",
            "Centros_Buenos": "https://mexico.as.com/resultados/futbol/mexico_apertura/2025/ranking/equipos/centros-buenos/",
            "Pases": "https://mexico.as.com/resultados/futbol/mexico_apertura/2025/ranking/equipos/pases/",
            "Pases_Buenos_Porcentaje": "https://mexico.as.com/resultados/futbol/mexico_apertura/2025/ranking/equipos/porcentaje-de-pases-buenos/",
            "Fueras_de_Juego": "https://mexico.as.com/resultados/futbol/mexico_apertura/2025/ranking/equipos/fueras-de-juego/",
            "Tarjetas_Rojas": "https://mexico.as.com/resultados/futbol/mexico_apertura/2025/ranking/equipos/tarjetas-rojas/",
            "Tarjetas_Amarillas": "https://mexico.as.com/resultados/futbol/mexico_apertura/2025/ranking/equipos/tarjetas-amarillas/",
            "Disparos_Puerta": "https://mexico.as.com/resultados/futbol/mexico_apertura/2025/ranking/equipos/disparos-a-puerta/",
            "Tiros_Puerta_Porcentaje": "https://mexico.as.com/resultados/futbol/mexico_apertura/2025/ranking/equipos/porcentaje-de-tiros-a-puerta/",
            "Saques_Esquina": "https://mexico.as.com/resultados/futbol/mexico_apertura/2025/ranking/equipos/saques-de-esquina/"
        }
        
        # Configuración para la API-Football
        self.api_football_base_url = "https://v3.football.api-sports.io/"
        self.api_football_key = "9bfc46b4d1cb7a3993701c054b993e81"
        self.headers = {
            "x-rapidapi-host": "v3.football.api-sports.io",
            "x-rapidapi-key": self.api_football_key
        }

        # Mapeo de nombres de equipos de AS.com y LigaMX.net a API-Football
        self.team_name_mapping_as_to_api = {
            "América": "Club America",
            "Cruz Azul": "Cruz Azul",
            "Atlas": "Atlas",
            "Atlético San Luis": "Atletico San Luis",
            "Bravos": "FC Juarez",
            "Chivas": "Guadalajara Chivas",
            "León FC": "Leon",
            "Mazatlán Fútbol Club": "Mazatlan",  # Sin tilde
            "Necaxa": "Necaxa",
            "Pachuca": "Pachuca",
            "Puebla": "Puebla",
            "Pumas": "U.N.A.M. - Pumas",
            "Rayados": "Monterrey",
            "Santos": "Santos Laguna",
            "Tigres": "Tigres UANL",
            "Toluca": "Toluca",
            "Xolos": "Club Tijuana",
            "Gallos Blancos": "Club Queretaro"
        }
        self.team_name_mapping_ligamx_to_api = {
            "América": "Club America",
            "Cruz Azul": "Cruz Azul",
            "Atlas": "Atlas",
            "Atlético de San Luis": "Atletico San Luis",
            "Bravos": "FC Juarez",
            "Chivas": "Guadalajara Chivas",
            "Guadalajara": "Guadalajara Chivas",  # Corrección para LigaMX.net
            "León": "Leon",
            "Mazatlán FC": "Mazatlan",  # Sin tilde
            "Necaxa": "Necaxa",
            "Pachuca": "Pachuca",
            "Puebla": "Puebla",
            "Pumas": "U.N.A.M. - Pumas",
            "Rayados": "Monterrey",
            "Santos": "Santos Laguna",
            "Tigres": "Tigres UANL",
            "Toluca": "Toluca",
            "Tijuana": "Club Tijuana",
            "Querétaro": "Club Queretaro"
        }

    def get_ligamx_historical_data(self):
        """
        Extrae los datos históricos de la tabla general de ligamx.net.
        """
        print(f"Intentando obtener datos de: {self.ligamx_url}")
        try:
            response = requests.get(self.ligamx_url, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', class_=['default', 'tbl_grals'])
            
            if not table:
                print("Tabla no encontrada con requests. Probando con Selenium...")
                options = Options()
                options.headless = True
                driver = webdriver.Chrome(options=options)
                driver.get(self.ligamx_url)
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                driver.quit()
                table = soup.find('table', class_=['default', 'tbl_grals'])
                if not table:
                    print("Error: Tabla no encontrada incluso con Selenium.")
                    return pd.DataFrame()

            table_body = table.find('tbody')
            if not table_body:
                print("Error: No se encontró el tbody dentro de la tabla principal.")
                return pd.DataFrame()
            
            all_cells = table_body.find_all('td')
            extracted_data_flat = [cell.get_text(strip=True) for cell in all_cells]
            print(f"Primeras 50 celdas extraídas: {extracted_data_flat[:50]}")
            
            expected_cols_per_team = 26
            data = []
            for i in range(0, len(extracted_data_flat), expected_cols_per_team):
                team_data = extracted_data_flat[i:i + expected_cols_per_team]
                if len(team_data) == expected_cols_per_team:
                    data.append(team_data)
                else:
                    print(f"Advertencia: Fila parcial con {len(team_data)} columnas: {team_data}")

            columns = [
                'Pos', 'Club',
                'JJ_Gral', 'JG_Gral', 'JE_Gral', 'JP_Gral', 'GF_Gral', 'GC_Gral', 'Dif_Gral', 'PTS_Gral',
                'JJ_Loc', 'JG_Loc', 'JE_Loc', 'JP_Loc', 'GF_Loc', 'GC_Loc', 'Dif_Loc', 'PTS_Loc',
                'JJ_Vis', 'JG_Vis', 'JE_Vis', 'JP_Vis', 'GF_Vis', 'GC_Vis', 'Dif_Vis', 'PTS_Vis'
            ]
            
            df = pd.DataFrame(data, columns=columns)
            numeric_cols = [col for col in columns if col != 'Club']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            print("Datos de ligamx.net obtenidos y procesados.")
            return df

        except requests.exceptions.RequestException as e:
            print(f"Error de conexión al intentar obtener datos de ligamx.net: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Ocurrió un error inesperado al procesar ligamx.net: {e}")
            traceback.print_exc()
            return pd.DataFrame()

    def get_ligamx_historical_tables(self):
        """
        Extrae y combina las tablas históricas de Liga MX para Clausura 2024-2025, Apertura 2024-2025, Clausura 2023-2024 y Apertura 2023-2024.
        """
        print("--- Scrapeando tablas históricas de Liga MX ---")
        dfs = []
        for season, url in self.ligamx_historical_urls.items():
            print(f"Scrapeando tabla de {season} desde {url}")
            try:
                response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                table = soup.find('table', class_=['default', 'tbl_grals'])
                
                if not table:
                    print(f"Tabla no encontrada para {season} con requests. Probando con Selenium...")
                    options = Options()
                    options.headless = True
                    driver = webdriver.Chrome(options=options)
                    driver.get(url)
                    soup = BeautifulSoup(driver.page_source, 'html.parser')
                    driver.quit()
                    table = soup.find('table', class_=['default', 'tbl_grals'])
                    if not table:
                        print(f"Error: Tabla no encontrada incluso con Selenium para {season}.")
                        continue

                table_body = table.find('tbody')
                if not table_body:
                    print(f"Error: No se encontró el tbody dentro de la tabla principal para {season}.")
                    continue
                
                all_cells = table_body.find_all('td')
                extracted_data_flat = [cell.get_text(strip=True) for cell in all_cells]
                print(f"Primeras 50 celdas extraídas para {season}: {extracted_data_flat[:50]}")
                
                expected_cols_per_team = 26
                data = []
                for i in range(0, len(extracted_data_flat), expected_cols_per_team):
                    team_data = extracted_data_flat[i:i + expected_cols_per_team]
                    if len(team_data) == expected_cols_per_team:
                        team_data.append(season)  # Añadir columna de temporada
                        data.append(team_data)
                    else:
                        print(f"Advertencia: Fila parcial con {len(team_data)} columnas en {season}: {team_data}")

                columns = [
                    'Pos', 'Club',
                    'JJ_Gral', 'JG_Gral', 'JE_Gral', 'JP_Gral', 'GF_Gral', 'GC_Gral', 'Dif_Gral', 'PTS_Gral',
                    'JJ_Loc', 'JG_Loc', 'JE_Loc', 'JP_Loc', 'GF_Loc', 'GC_Loc', 'Dif_Loc', 'PTS_Loc',
                    'JJ_Vis', 'JG_Vis', 'JE_Vis', 'JP_Vis', 'GF_Vis', 'GC_Vis', 'Dif_Vis', 'PTS_Vis',
                    'Season'
                ]
                
                df = pd.DataFrame(data, columns=columns)
                numeric_cols = [col for col in columns if col not in ['Club', 'Season']]
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df = self.standardize_team_names(df, column_name='Club', source_type='ligamx')
                dfs.append(df)
                print(f"Tabla de {season} scrapeada: {len(df)} filas")
                
            except requests.exceptions.RequestException as e:
                print(f"Error de conexión al intentar obtener datos de {season}: {e}")
                continue
            except Exception as e:
                print(f"Ocurrió un error inesperado al procesar {season}: {e}")
                traceback.print_exc()
                continue
        
        if not dfs:
            print("Error: No se scrapeó ninguna tabla histórica.")
            return pd.DataFrame()
        
        # Combinar tablas
        df_combined = pd.concat(dfs, ignore_index=True)
        print(f"Tablas históricas combinadas: {len(df_combined)} filas")
        
        # Guardar en CSV
        df_combined.to_csv('ligamx_stats_historical.csv', index=False, encoding='utf-8')
        print("Datos históricos guardados en: ligamx_stats_historical.csv")
        return df_combined

    def get_threesixtyfive_possession_data(self):
        """
        Extrae los datos de la tabla "Posesión del balón" de 365scores.com.
        """
        print(f"Intentando obtener datos de posesión de balón de: {self.threesixtyfive_url}")
        try:
            response = requests.get(self.threesixtyfive_url, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            possession_title_div = soup.find('h2', class_='card-title_title__M4Qdq', string='Posesión del balón')
            
            if not possession_title_div:
                print("Error: No se encontró el título 'Posesión del balón' en 365scores.com.")
                return pd.DataFrame()

            main_widget_div = possession_title_div.find_parent('div').find_parent('div')
            if not main_widget_div:
                print("Error: No se pudo encontrar el contenedor principal del widget de posesión.")
                return pd.DataFrame()

            team_rows = main_widget_div.find_all('a', class_='entity-stats-widget_row__sCcPB')
            if not team_rows:
                print("Advertencia: No se encontraron filas de equipos para la posesión del balón.")
                return pd.DataFrame()

            data = []
            for row in team_rows:
                club_name_span = row.find('span', class_='entity-stats-widget_player_name__WvQPB')
                matches_played_div = row.find('div', class_='entity-stats-widget_entity_competitor_name__pXZcQ')
                possession_value_div = row.find('div', class_='entity-stats-widget_stats_value__des13')

                club_name = club_name_span.get_text(strip=True) if club_name_span else None
                matches_played = None
                if matches_played_div:
                    matches_text = matches_played_div.get_text(strip=True)
                    if 'Partidos jugados:' in matches_text:
                        try:
                            matches_played = int(matches_text.split(':')[1].strip())
                        except ValueError:
                            matches_played = None

                possession_percent = possession_value_div.get_text(strip=True).replace('%', '') if possession_value_div else None
                if possession_percent:
                    try:
                        possession_percent = float(possession_percent)
                    except ValueError:
                        possession_percent = None

                data.append({
                    'Club': club_name,
                    'Partidos_Jugados': matches_played,
                    'Posesion_Balon_Porcentaje': possession_percent
                })

            df = pd.DataFrame(data)
            print("Datos de posesión de balón de 365scores.com obtenidos y procesados.")
            return df

        except requests.exceptions.RequestException as e:
            print(f"Error de conexión al intentar obtener datos de 365scores.com: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Ocurrió un error inesperado al procesar 365scores.com (posesión): {e}")
            traceback.print_exc()
            return pd.DataFrame()

    def get_api_football_data(self, endpoint, params=None):
        """
        Función genérica para obtener datos de la API-Football.
        """
        url = f"{self.api_football_base_url}{endpoint}"
        print(f"Intentando obtener datos de API-Football desde: {url}")
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            if data and 'response' in data:
                if not data['response']:
                    print(f"Advertencia: La respuesta de API-Football para '{endpoint}' no contiene datos.")
                    print(f"Respuesta completa: {json.dumps(data, indent=2)}")
                else:
                    print(f"Datos de API-Football ({endpoint}) obtenidos con éxito.")
                return data['response']
            else:
                print(f"Advertencia: La clave 'response' no se encontró para API-Football ({endpoint}).")
                print(f"Respuesta completa: {json.dumps(data, indent=2)}")
                return []
        except requests.exceptions.RequestException as e:
            print(f"Error de conexión al intentar obtener datos de API-Football ({endpoint}): {e}")
            if response is not None and response.status_code == 429:
                print("Posiblemente excediste los límites de tu plan. Espera y reintenta.")
            return []
        except json.JSONDecodeError:
            print(f"Error al decodificar la respuesta JSON de API-Football ({endpoint}).")
            return []
        except Exception as e:
            print(f"Ocurrió un error inesperado al procesar datos de API-Football ({endpoint}): {e}")
            traceback.print_exc()
            return []

    def get_ligamx_standings(self, season):
        """
        Obtiene las clasificaciones de la Liga MX (ID 262).
        """
        print(f"Obteniendo clasificaciones de Liga MX para la temporada {season}...")
        params = {"league": 262, "season": season}
        standings_data = self.get_api_football_data("standings", params)
        if not standings_data:
            print(f"No se encontraron datos de clasificación para la temporada {season}.")
            return pd.DataFrame()
        processed_data = []
        try:
            if standings_data and 'league' in standings_data[0] and 'standings' in standings_data[0]['league']:
                for standing in standings_data[0]['league']['standings'][0]:
                    processed_data.append({
                        'Rank': standing.get('rank'),
                        'Team': standing.get('team', {}).get('name'),
                        'Points': standing.get('points'),
                        'Played': standing.get('all', {}).get('played'),
                        'Wins': standing.get('all', {}).get('win'),
                        'Draws': standing.get('all', {}).get('draw'),
                        'Losses': standing.get('all', {}).get('lose'),
                        'Goals_For': standing.get('all', {}).get('goals', {}).get('for'),
                        'Goals_Against': standing.get('all', {}).get('goals', {}).get('against'),
                        'Goal_Diff': standing.get('goalsDiff'),
                        'Form': standing.get('form')
                    })
        except Exception as e:
            print(f"Error al procesar los datos de clasificación para la temporada {season}: {e}")
            traceback.print_exc()
            return pd.DataFrame()
        df = pd.DataFrame(processed_data)
        return df

    def get_ligamx_teams(self, season):
        """
        Obtiene la lista de equipos de la Liga MX.
        """
        print(f"Obteniendo equipos de Liga MX para la temporada {season}...")
        params = {"league": 262, "season": season}
        teams_data = self.get_api_football_data("teams", params)
        if not teams_data:
            print(f"No se encontraron datos de equipos para la temporada {season}.")
            return pd.DataFrame()
        processed_data = []
        try:
            for team in teams_data:
                processed_data.append({
                    'Team_ID': team.get('team', {}).get('id'),
                    'Team_Name': team.get('team', {}).get('name'),
                    'Team_Code': team.get('team', {}).get('code'),
                    'Country': team.get('team', {}).get('country'),
                    'Founded': team.get('team', {}).get('founded'),
                    'Logo_URL': team.get('team', {}).get('logo')
                })
        except Exception as e:
            print(f"Error al procesar los datos de equipos para la temporada {season}: {e}")
            traceback.print_exc()
            return pd.DataFrame()
        df = pd.DataFrame(processed_data)
        return df

    def get_ligamx_fixtures(self, season, date=None, team_id=None, status=None):
        """
        Obtiene los partidos de la Liga MX.
        """
        print(f"Obteniendo partidos de Liga MX para la temporada {season}...")
        params = {"league": 262, "season": season}
        if date:
            params["date"] = date
        if team_id:
            params["team"] = team_id
        if status:
            params["status"] = status
        fixtures_data = self.get_api_football_data("fixtures", params)
        if not fixtures_data:
            print(f"No se encontraron partidos para la temporada {season}.")
            return pd.DataFrame()
        processed_data = []
        try:
            for fixture in fixtures_data:
                processed_data.append({
                    'Fixture_ID': fixture.get('fixture', {}).get('id'),
                    'Date': fixture.get('fixture', {}).get('date'),
                    'Time': fixture.get('fixture', {}).get('date').split('T')[1].split('+')[0] if fixture.get('fixture', {}).get('date') else None,
                    'Status_Long': fixture.get('fixture', {}).get('status', {}).get('long'),
                    'Status_Short': fixture.get('fixture', {}).get('status', {}).get('short'),
                    'Elapsed': fixture.get('fixture', {}).get('status', {}).get('elapsed'),
                    'Home_Team': fixture.get('teams', {}).get('home', {}).get('name'),
                    'Away_Team': fixture.get('teams', {}).get('away', {}).get('name'),
                    'Home_Goals': fixture.get('goals', {}).get('home'),
                    'Away_Goals': fixture.get('goals', {}).get('away'),
                    'Venue': fixture.get('fixture', {}).get('venue', {}).get('name'),
                    'Referee': fixture.get('fixture', {}).get('referee')
                })
        except Exception as e:
            print(f"Error al procesar los datos de partidos para la temporada {season}: {e}")
            traceback.print_exc()
            return pd.DataFrame()
        df = pd.DataFrame(processed_data)
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        return df

    def get_fixture_statistics(self, fixture_id):
        """
        Obtiene las estadísticas detalladas de un partido.
        """
        print(f"Obteniendo estadísticas para el partido (Fixture ID: {fixture_id})...")
        params = {"fixture": fixture_id}
        stats_data = self.get_api_football_data("fixtures/statistics", params)
        if not stats_data:
            print(f"No se encontraron estadísticas para el partido (Fixture ID: {fixture_id}).")
            return pd.DataFrame()
        processed_data = []
        try:
            for team_stats_entry in stats_data:
                team_name = team_stats_entry.get('team', {}).get('name')
                team_id = team_stats_entry.get('team', {}).get('id')
                stats_dict = {'Fixture_ID': fixture_id, 'Team_ID': team_id, 'Team_Name': team_name}
                for stat in team_stats_entry.get('statistics', []):
                    type = stat.get('type')
                    value = stat.get('value')
                    if isinstance(value, str) and '%' in value:
                        try:
                            value = float(value.replace('%', ''))
                        except ValueError:
                            pass
                    stats_dict[type.replace(' ', '_')] = value
                processed_data.append(stats_dict)
        except Exception as e:
            print(f"Error al procesar las estadísticas del partido (Fixture ID: {fixture_id}): {e}")
            traceback.print_exc()
            return pd.DataFrame()
        df = pd.DataFrame(processed_data)
        return df

    def _get_single_as_stat_table(self, url, metric_column_name):
        """
        Extrae una tabla de estadísticas de AS.com.
        """
        print(f"Intentando obtener datos de '{metric_column_name}' de: {url}")
        try:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            dfs = pd.read_html(StringIO(response.text))
            if not dfs:
                print(f"Error: No se pudo leer ninguna tabla HTML para {metric_column_name}.")
                return pd.DataFrame()
            stat_df = pd.DataFrame()
            expected_cols_in_table = ['Equipo', 'Total']
            for df_found in dfs:
                if all(col in df_found.columns for col in expected_cols_in_table) and \
                   ('Pos.' in df_found.columns or 'Pos' in df_found.columns):
                    stat_df = df_found
                    print(f"Tabla de '{metric_column_name}' identificada por columnas esperadas.")
                    break
            if stat_df.empty:
                print(f"Error: No se encontraron tablas con columnas esperadas para {metric_column_name}.")
                return pd.DataFrame()
            column_mapping = {'Equipo': 'Club', 'Total': metric_column_name}
            stat_df.rename(columns={k: v for k, v in column_mapping.items() if k in stat_df.columns}, inplace=True)
            if metric_column_name in stat_df.columns:
                stat_df[metric_column_name] = stat_df[metric_column_name].astype(str).str.replace(',', '.')
                stat_df[metric_column_name] = stat_df[metric_column_name].str.extract(r'(\d+\.?\d*)', expand=False)
                stat_df[metric_column_name] = pd.to_numeric(stat_df[metric_column_name], errors='coerce')
            print(f"Datos de '{metric_column_name}' de AS.com obtenidos y procesados.")
            required_cols = ['Club', metric_column_name]
            return stat_df[[col for col in required_cols if col in stat_df.columns]]
        except requests.exceptions.RequestException as e:
            print(f"Error de conexión al intentar obtener datos de AS.com para {metric_column_name}: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Ocurrió un error inesperado al procesar AS.com para {metric_column_name}: {e}")
            traceback.print_exc()
            return pd.DataFrame()

    def get_all_as_statistics(self):
        """
        Obtiene todas las estadísticas de AS.com y las combina.
        """
        print("\n--- Iniciando la extracción de todas las estadísticas de AS.com ---")
        all_stats_dfs = []
        for stat_name, stat_url in self.as_stats_urls.items():
            df = self._get_single_as_stat_table(stat_url, stat_name)
            if not df.empty:
                all_stats_dfs.append(df)
        if not all_stats_dfs:
            print("No se pudo obtener ninguna estadística de AS.com.")
            return pd.DataFrame()
        final_df = all_stats_dfs[0]
        for i in range(1, len(all_stats_dfs)):
            final_df = pd.merge(final_df, all_stats_dfs[i], on='Club', how='outer')
        print("\n--- Todas las estadísticas de AS.com obtenidas y combinadas exitosamente. ---")
        return final_df

    def standardize_team_names(self, df, column_name='Club', source_type='as_com'):
        """
        Estandariza los nombres de los equipos, eliminando tildes y aplicando mapeo.
        """
        if df.empty:
            return df
        # Normalizar tildes y caracteres especiales
        df[column_name] = df[column_name].astype(str).str.strip()
        df[column_name] = df[column_name].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('ascii')
        if source_type == 'as_com':
            print(f"Estandarizando nombres de equipos en columna '{column_name}' (Origen: AS.com)...")
            df[column_name] = df[column_name].str.replace(r'\s*Mexico$', '', regex=True)
            df[column_name] = df[column_name].map(self.team_name_mapping_as_to_api).fillna(df[column_name])
            print(f"Nombres estandarizados (AS.com): {df[column_name].unique()}")
        elif source_type == 'ligamx':
            print(f"Estandarizando nombres de equipos en columna '{column_name}' (Origen: LigaMX.net)...")
            df[column_name] = df[column_name].map(self.team_name_mapping_ligamx_to_api).fillna(df[column_name])
            print(f"Nombres estandarizados (LigaMX.net): {df[column_name].unique()}")
        elif source_type == 'api_football':
            print(f"Nombres de equipos en columna '{column_name}' (Origen: API-Football) no requieren mapeo.")
        else:
            print(f"Tipo de fuente '{source_type}' no reconocido.")
        return df

if __name__ == "__main__":
    scraper = DataScraper()
    
    # --- API-Football (2023) ---
    print("======================================================================")
    print("--- Intentando obtener datos de API-Football (Temporadas 2021-2023) ---")
    api_football_test_season = 2023

    ligamx_standings_api = scraper.get_ligamx_standings(season=api_football_test_season)
    ligamx_standings_api = scraper.standardize_team_names(ligamx_standings_api, column_name='Team', source_type='api_football')
    if not ligamx_standings_api.empty:
        print(f"\n--- Clasificaciones de Liga MX (API-Football, {api_football_test_season} - primeras 5 filas) ---")
        print(ligamx_standings_api.head())
    else:
        print(f"\n--- No se pudieron obtener las clasificaciones de Liga MX (API-Football, {api_football_test_season}). ---")

    print("\n" + "-"*30 + "\n")
    ligamx_teams_api = scraper.get_ligamx_teams(season=api_football_test_season)
    ligamx_teams_api = scraper.standardize_team_names(ligamx_teams_api, column_name='Team_Name', source_type='api_football')
    if not ligamx_teams_api.empty:
        print(f"\n--- Equipos de Liga MX (API-Football, {api_football_test_season} - primeras 5 filas) ---")
        print(ligamx_teams_api.head())
    else:
        print(f"\n--- No se pudieron obtener los equipos de Liga MX (API-Football, {api_football_test_season}). ---")

    print("\n" + "-"*30 + "\n")
    fixtures_api = scraper.get_ligamx_fixtures(season=api_football_test_season, status="FT")
    fixtures_api = scraper.standardize_team_names(fixtures_api, column_name='Home_Team', source_type='api_football')
    fixtures_api = scraper.standardize_team_names(fixtures_api, column_name='Away_Team', source_type='api_football')
    if not fixtures_api.empty:
        print(f"\n--- Partidos (Fixtures) de Liga MX (API-Football, {api_football_test_season} - primeras 5 filas) ---")
        print(fixtures_api.head())
        if not fixtures_api['Fixture_ID'].empty:
            sample_fixture_id = fixtures_api['Fixture_ID'].iloc[0]
            print(f"\nIntentando obtener estadísticas para el primer partido (ID: {sample_fixture_id})...")
            fixture_stats_df = scraper.get_fixture_statistics(sample_fixture_id)
            fixture_stats_df = scraper.standardize_team_names(fixture_stats_df, column_name='Team_Name', source_type='api_football')
            if not fixture_stats_df.empty:
                print(f"\n--- Estadísticas del Partido (API-Football, Fixture ID: {sample_fixture_id}) ---")
                print(fixture_stats_df.head())
            else:
                print(f"\n--- No se pudieron obtener las estadísticas para el partido (Fixture ID: {sample_fixture_id}). ---")
    else:
        print(f"\n--- No se pudieron obtener los partidos de Liga MX para la temporada {api_football_test_season}. ---")

    print("\n" + "="*70 + "\n")

    # --- LigaMX.net (Historical Tables) ---
    print("--- Intentando obtener datos históricos de LigaMX.net ---")
    ligamx_historical_df = scraper.get_ligamx_historical_tables()
    if not ligamx_historical_df.empty:
        print("\n--- DataFrame de Tablas Históricas de LigaMX.net (Primeras 5 filas) ---")
        print(ligamx_historical_df.head())
        print("\n--- Nombres de equipos únicos en Tablas Históricas de LigaMX.net ---")
        print(ligamx_historical_df['Club'].unique())
    else:
        print("\n--- No se pudieron obtener datos históricos de LigaMX.net ---")

    print("\n" + "="*70 + "\n")

    # --- LigaMX.net (2025) ---
    print("--- Intentando obtener datos de LigaMX.net (Apertura 2025) ---")
    ligamx_stats_df = scraper.get_ligamx_historical_data()
    ligamx_stats_df = scraper.standardize_team_names(ligamx_stats_df, column_name='Club', source_type='ligamx')
    if not ligamx_stats_df.empty:
        print("\n--- DataFrame de LigaMX.net (Primeras 5 filas) ---")
        print(ligamx_stats_df.head())
        print("\n--- Nombres de equipos únicos en LigaMX.net ---")
        print(ligamx_stats_df['Club'].unique())
    else:
        print("\n--- No se pudieron obtener datos de LigaMX.net ---")

    print("\n" + "="*70 + "\n")

    # --- AS.com (2025) ---
    print(f"\n--- Intentando obtener TODAS las estadísticas de AS.com para el Apertura 2025 ---")
    all_as_stats_df = scraper.get_all_as_statistics()
    all_as_stats_df = scraper.standardize_team_names(all_as_stats_df, column_name='Club', source_type='as_com')
    if not all_as_stats_df.empty:
        print("\n--- DataFrame Combinado de TODAS las Estadísticas (AS.com) - Primeras 5 filas ---")
        print(all_as_stats_df.head())
        print("\n--- Información del DataFrame Combinado (AS.com) ---")
        print(all_as_stats_df.info())
        print("\n--- Nombres de equipos únicos en el DataFrame de AS.com ---")
        print(all_as_stats_df['Club'].unique())
    else:
        print("\n--- No se pudieron obtener datos de AS.com ---")

    print("\n" + "="*70 + "\n")

    # --- Combinar DataFrames ---
    print("\n--- Combinando DataFrames de API-Football, AS.com y LigaMX.net ---")
    if not ligamx_standings_api.empty and not ligamx_teams_api.empty:
        ligamx_standings_api_renamed = ligamx_standings_api.rename(columns={'Team': 'Club_Nombre'})
        combined_api_df = pd.merge(
            ligamx_standings_api_renamed,
            ligamx_teams_api.drop(columns=['Team_Code', 'Country', 'Founded', 'Logo_URL'], errors='ignore'),
            left_on='Club_Nombre',
            right_on='Team_Name',
            how='left',
            suffixes=('_Standings', '_Teams')
        )
        if 'Team_Name' in combined_api_df.columns:
            combined_api_df.drop(columns=['Team_Name'], inplace=True)
        print("\n--- DataFrame Combinado de API-Football (Clasificaciones y Equipos) - Primeras 5 filas ---")
        print(combined_api_df.head())
        print("\n--- Información del DataFrame Combinado de API-Football ---")
        print(combined_api_df.info())

        final_combined_df = pd.merge(
            combined_api_df,
            all_as_stats_df,
            left_on='Club_Nombre',
            right_on='Club',
            how='left',
            suffixes=('_API', '_AS')
        )
        if 'Club' in final_combined_df.columns:
            final_combined_df.drop(columns=['Club'], inplace=True)

        if not ligamx_stats_df.empty:
            final_combined_df = pd.merge(
                final_combined_df,
                ligamx_stats_df,
                left_on='Club_Nombre',
                right_on='Club',
                how='left',
                suffixes=('_API_AS', '_LigaMX')
            )
            if 'Club' in final_combined_df.columns:
                final_combined_df.drop(columns=['Club'], inplace=True)

        # Eliminar filas con Club_Nombre NaN
        final_combined_df = final_combined_df.dropna(subset=['Club_Nombre'])

        # Rellenar valores nulos en columnas numéricas con la media
        numeric_cols = final_combined_df.select_dtypes(include=['float64', 'int64']).columns
        final_combined_df[numeric_cols] = final_combined_df[numeric_cols].fillna(final_combined_df[numeric_cols].mean())

        print("\n--- DataFrame Final Combinado (API-Football + AS.com + LigaMX.net) - Primeras 5 filas ---")
        print(final_combined_df.head())
        print("\n--- Información del DataFrame Final Combinado ---")
        print(final_combined_df.info())
        print("\n--- Nombres de equipos únicos en el DataFrame Final Combinado ---")
        print(final_combined_df['Club_Nombre'].unique())
        print("\n--- Equipos en API-Football pero no en AS.com ---")
        print(final_combined_df[final_combined_df['Posesion_Balon_Porcentaje'].isna()]['Club_Nombre'].unique())
        print("\n--- Equipos en AS.com pero no en API-Football ---")
        print(final_combined_df[final_combined_df['Rank'].isna()]['Club_Nombre'].unique())
        print("\n--- Equipos en API-Football pero no en LigaMX.net ---")
        print(final_combined_df[final_combined_df['Pos'].isna()]['Club_Nombre'].unique())
    else:
        print("\n--- No se pudieron combinar los DataFrames. Verifique que los DataFrames individuales no estén vacíos. ---")

    # --- Guardar DataFrames como CSV ---
    print("\n--- Guardando DataFrames procesados como archivos CSV ---")
    try:
        if not fixtures_api.empty:
            fixtures_api.to_csv('api_football_fixtures.csv', index=False)
            print("DataFrame 'fixtures_api' guardado como 'api_football_fixtures.csv'")
        if not ligamx_stats_df.empty:
            ligamx_stats_df.to_csv('ligamx_stats.csv', index=False)
            print("DataFrame 'ligamx_stats_df' guardado como 'ligamx_stats.csv'")
        if not all_as_stats_df.empty:
            all_as_stats_df.to_csv('as_com_stats_combined.csv', index=False)
            print("DataFrame 'all_as_stats_df' guardado como 'as_com_stats_combined.csv'")
        if 'final_combined_df' in locals() and not final_combined_df.empty:
            final_combined_df.to_csv('liga_mx_full_data.csv', index=False)
            print("DataFrame 'final_combined_df' guardado como 'liga_mx_full_data.csv'")
        print("Proceso de guardado de CSVs finalizado.")
    except Exception as e:
        print(f"Error al guardar archivos CSV: {e}")