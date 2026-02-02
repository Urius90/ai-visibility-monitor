#!/usr/bin/env python3
"""
AI Visibility Monitor - Grupo Vaughan
Monitorea la presencia de marca en respuestas de múltiples modelos de IA
"""

import anthropic
import google.generativeai as genai
from openai import OpenAI
import requests
import pandas as pd
from datetime import datetime
import json
import time
import os
import sys
import gspread
from google.oauth2.service_account import Credentials

# ===============================================
# CONFIGURACIÓN
# ===============================================

# API Keys desde environment variables
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')
BING_API_KEY = os.getenv('BING_API_KEY')

# Google Sheets
GOOGLE_SHEETS_CREDENTIALS = "credentials.json"
SPREADSHEET_NAME = "AI Visibility Monitor - Grupo Vaughan"
SPREADSHEET_ID = "1Zj47IExAqH0wP6yKO3VBIDyxRsaTCCZ8404jOCoAWMY"

# Validar API keys obligatorias
REQUIRED_KEYS = {
    'ANTHROPIC_API_KEY': ANTHROPIC_API_KEY,
    'GOOGLE_API_KEY': GOOGLE_API_KEY,
    'OPENAI_API_KEY': OPENAI_API_KEY,
}

missing_keys = [k for k, v in REQUIRED_KEYS.items() if not v]
if missing_keys:
    print(f"❌ ERROR: Faltan las siguientes API keys: {', '.join(missing_keys)}")
    sys.exit(1)

# Queries a monitorear
QUERIES = [
    # Transaccionales
    "mejores academias de inglés en España",
    "cursos de inglés online certificados",
    "clases de inglés para empresas",
    "academia de inglés Madrid",
    "cursos intensivos de inglés",
    
    # Informacionales
    "cómo aprender inglés rápido",
    "diferencia entre B1 y B2 inglés",
    "método para aprender inglés adultos",
    "cuánto tiempo se tarda en aprender inglés",
    
    # Branded/Defensivas
    "método Vaughan funciona",
    "opiniones Grupo Vaughan",
]

BRAND_KEYWORDS = [
    "vaughan",
    "grupo vaughan",
    "richard vaughan",
    "vaughantown",
]

# ===============================================
# FUNCIONES DE CONSULTA A LLMs
# ===============================================

def check_chatgpt(query: str) -> dict:
    """Consulta ChatGPT (GPT-4)"""
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": query}],
            max_tokens=1500,
            temperature=0.7
        )
        text = response.choices[0].message.content
        return {
            'model': 'ChatGPT',
            'mentioned': check_mentions(text),
            'position': find_mention_position(text),
            'context': extract_mention_context(text),
            'full_response': text[:500],
            'error': None
        }
    except Exception as e:
        return {'model': 'ChatGPT', 'mentioned': False, 'position': None, 'error': str(e)}

def check_claude(query: str) -> dict:
    """Consulta Claude (Sonnet 4)"""
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        message = client.messages.create(
            model="claude-3-5-sonnet-latest", # Alias estable
            max_tokens=1000,
            messages=[{"role": "user", "content": query}]
        )
        text = message.content[0].text
        return {
            'model': 'Claude',
            'mentioned': check_mentions(text),
            'position': find_mention_position(text),
            'context': extract_mention_context(text),
            'full_response': text[:500],
            'error': None
        }
    except anthropic.APIStatusError as e:
        error_msg = f"{e.status_code} - {e.message}"
        if hasattr(e, 'body'):
            error_msg += f" | Body: {e.body}"
        return {'model': 'Claude', 'mentioned': False, 'position': None, 'error': error_msg}
    except Exception as e:
        return {'model': 'Claude', 'mentioned': False, 'position': None, 'error': str(e)}

def check_gemini(query: str) -> dict:
    """Consulta Gemini"""
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        genai.configure(api_key=GOOGLE_API_KEY)
        # Probamos Gemini 2.0 Flash Lite que sabíamos que existía (dio 429 antes)
        model = genai.GenerativeModel('gemini-2.0-flash-lite-preview-02-05') 
        # Fallback a nombre generico si hace falta, pero intentemos ser específicos o usar el que funcionaba
        # ERROR 404 indicated 1.5-flash-001 was gone. 
        # ERROR 429 earlier indicated 2.0-flash-lite existed.
        # Vamos a probar con 'gemini-2.0-flash' genérico que suele ser el alias estable.
        model = genai.GenerativeModel('gemini-2.0-flash') 
        response = model.generate_content(query)
        text = response.text
        return {
            'model': 'Gemini',
            'mentioned': check_mentions(text),
            'position': find_mention_position(text),
            'context': extract_mention_context(text),
            'full_response': text[:500],
            'error': None
        }
    except Exception as e:
        return {'model': 'Gemini', 'mentioned': False, 'position': None, 'error': str(e)}

def check_perplexity(query: str) -> dict:
    """Consulta Perplexity"""
    if not PERPLEXITY_API_KEY:
        return {'model': 'Perplexity', 'mentioned': False, 'position': None, 'error': 'API key no configurada'}
    
    try:
        url = "https://api.perplexity.ai/chat/completions"
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "sonar",
            "messages": [{"role": "user", "content": query}],
            "max_tokens": 1500
        }
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        text = response.json()['choices'][0]['message']['content']
        return {
            'model': 'Perplexity',
            'mentioned': check_mentions(text),
            'position': find_mention_position(text),
            'context': extract_mention_context(text),
            'full_response': text[:500],
            'error': None
        }
    except Exception as e:
        return {'model': 'Perplexity', 'mentioned': False, 'position': None, 'error': str(e)}

def check_copilot(query: str) -> dict:
    """Consulta Bing/Copilot (vía Bing Search API)"""
    if not BING_API_KEY:
        return {'model': 'Bing/Copilot', 'mentioned': False, 'position': None, 'error': 'API key no configurada'}
    
    try:
        search_url = "https://api.bing.microsoft.com/v7.0/search"
        headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}
        params = {"q": query, "count": 10}
        
        response = requests.get(search_url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        search_results = response.json()
        
        mentioned = False
        position = None
        
        for i, result in enumerate(search_results.get('webPages', {}).get('value', []), 1):
            result_text = f"{result.get('name', '')} {result.get('snippet', '')} {result.get('url', '')}".lower()
            if any(kw in result_text for kw in BRAND_KEYWORDS):
                mentioned = True
                position = i
                break
        
        return {
            'model': 'Bing/Copilot',
            'mentioned': mentioned,
            'position': position,
            'context': 'Search results analysis',
            'error': None
        }
    except Exception as e:
        return {'model': 'Bing/Copilot', 'mentioned': False, 'position': None, 'error': str(e)}

# ===============================================
# FUNCIONES AUXILIARES
# ===============================================

def check_mentions(text: str) -> bool:
    """Verifica si alguna keyword está en el texto"""
    text_lower = text.lower()
    return any(kw in text_lower for kw in BRAND_KEYWORDS)

def find_mention_position(text: str) -> int:
    """Encuentra la posición de la primera mención"""
    text_lower = text.lower()
    
    for kw in BRAND_KEYWORDS:
        if kw in text_lower:
            index = text_lower.index(kw)
            text_before = text[:index]
            lines_before = len([line for line in text_before.split('\n') if line.strip()])
            return 1 if lines_before <= 2 else lines_before
    
    return None

def extract_mention_context(text: str, window: int = 120) -> str:
    """Extrae contexto alrededor de la mención"""
    text_lower = text.lower()
    
    for kw in BRAND_KEYWORDS:
        if kw in text_lower:
            index = text_lower.index(kw)
            start = max(0, index - window)
            end = min(len(text), index + len(kw) + window)
            context = text[start:end].strip()
            return context
    
    return None

def categorize_query(query: str) -> str:
    """Categoriza el tipo de query"""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['mejores', 'opiniones', 'mejor', 'top']):
        return 'Comparativa'
    elif any(word in query_lower for word in ['vaughan', 'richard']):
        return 'Branded'
    elif any(word in query_lower for word in ['cómo', 'qué', 'cuánto', 'diferencia']):
        return 'Informacional'
    else:
        return 'Transaccional'

# ===============================================
# GOOGLE SHEETS INTEGRATION
# ===============================================

def init_google_sheets():
    """Inicializa conexión con Google Sheets"""
    try:
        if not os.path.exists(GOOGLE_SHEETS_CREDENTIALS):
            print(f"❌ No se encuentra el archivo {GOOGLE_SHEETS_CREDENTIALS}")
            return None
        
        scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive'
        ]
        
        creds = Credentials.from_service_account_file(
            GOOGLE_SHEETS_CREDENTIALS, 
            scopes=scope
        )
        client = gspread.authorize(creds)
        
        try:
            if SPREADSHEET_ID:
                spreadsheet = client.open_by_key(SPREADSHEET_ID)
                print(f"✅ Spreadsheet encontrado por ID: {SPREADSHEET_ID}")
            else:
                spreadsheet = client.open(SPREADSHEET_NAME)
                print(f"✅ Spreadsheet encontrado: {SPREADSHEET_NAME}")
        except gspread.SpreadsheetNotFound:
            print(f"❌ No se encontró el Spreadsheet con ID {SPREADSHEET_ID} o nombre '{SPREADSHEET_NAME}'")
            return None
        
        return spreadsheet
    except Exception as e:
        print(f"❌ Error conectando con Google Sheets: {e}")
        return None

def setup_sheets(spreadsheet):
    """Configura las hojas necesarias"""
    try:
        sheet_names = [sheet.title for sheet in spreadsheet.worksheets()]
        
        # Hoja 1: Raw Data
        if "Raw Data" not in sheet_names:
            spreadsheet.add_worksheet(title="Raw Data", rows=10000, cols=15)
            sheet = spreadsheet.worksheet("Raw Data")
            
            headers = [
                'Timestamp', 'Fecha', 'Hora', 'Query', 'Categoría',
                'ChatGPT_Mención', 'ChatGPT_Posición',
                'Claude_Mención', 'Claude_Posición',
                'Gemini_Mención', 'Gemini_Posición',
                'Perplexity_Mención', 'Perplexity_Posición',
                'Bing_Mención', 'Bing_Posición'
            ]
            sheet.update(range_name='A1:O1', values=[headers])
            sheet.format('A1:O1', {
                "backgroundColor": {"red": 0.2, "green": 0.2, "blue": 0.8},
                "textFormat": {"bold": True, "foregroundColor": {"red": 1, "green": 1, "blue": 1}}
            })
        
        # Hoja 2: Dashboard
        if "Dashboard" not in sheet_names:
            spreadsheet.add_worksheet(title="Dashboard", rows=50, cols=10)
            dashboard = spreadsheet.worksheet("Dashboard")
            
            dashboard.update(range_name='A1', values=[['AI VISIBILITY MONITOR - GRUPO VAUGHAN']])
            dashboard.format('A1', {"textFormat": {"bold": True, "fontSize": 16}})
            dashboard.merge_cells('A1:F1')
            
            dashboard.update(range_name='A3:F3', values=[['Modelo', 'Menciones', 'Tasa %', 'Pos. Promedio', 'Última Act.', 'Tendencia']])
            dashboard.format('A3:F3', {
                "backgroundColor": {"red": 0.95, "green": 0.95, "blue": 0.95},
                "textFormat": {"bold": True}
            })
        
        # Hoja 3: Timeline (historico por fecha)
        if "Timeline" not in sheet_names:
            spreadsheet.add_worksheet(title="Timeline", rows=1000, cols=8)
            timeline = spreadsheet.worksheet("Timeline")
            timeline_headers = ['Fecha', 'ChatGPT', 'Claude', 'Gemini', 'Perplexity', 'Bing', 'Total', 'Tasa %']
            timeline.update(range_name='A1:H1', values=[timeline_headers])
            timeline.format('A1:H1', {
                "backgroundColor": {"red": 0.2, "green": 0.2, "blue": 0.8},
                "textFormat": {"bold": True, "foregroundColor": {"red": 1, "green": 1, "blue": 1}}
            })

        # Hoja 4: Por Query (analisis por query individual)
        if "Por Query" not in sheet_names:
            spreadsheet.add_worksheet(title="Por Query", rows=100, cols=9)
            por_query = spreadsheet.worksheet("Por Query")
            pq_headers = ['Query', 'Categoría', 'ChatGPT', 'Claude', 'Gemini', 'Perplexity', 'Bing', 'Total Menciones', 'Tasa %']
            por_query.update(range_name='A1:I1', values=[pq_headers])
            por_query.format('A1:I1', {
                "backgroundColor": {"red": 0.2, "green": 0.2, "blue": 0.8},
                "textFormat": {"bold": True, "foregroundColor": {"red": 1, "green": 1, "blue": 1}}
            })

        print("✅ Hojas configuradas correctamente")
    except Exception as e:
        print(f"⚠️  Error configurando hojas: {e}")

def upload_to_sheets(spreadsheet, results: list):
    """Sube resultados al spreadsheet"""
    try:
        sheet = spreadsheet.worksheet("Raw Data")
        
        timestamp = datetime.now()
        fecha = timestamp.strftime('%Y-%m-%d')
        hora = timestamp.strftime('%H:%M:%S')
        
        rows_to_add = []
        
        for result in results:
            query = result['query']
            categoria = categorize_query(query)
            models = result['models']
            
            row = [
                timestamp.isoformat(),
                fecha,
                hora,
                query,
                categoria,
                'Sí' if models.get('ChatGPT', {}).get('mentioned') else 'No',
                models.get('ChatGPT', {}).get('position', ''),
                'Sí' if models.get('Claude', {}).get('mentioned') else 'No',
                models.get('Claude', {}).get('position', ''),
                'Sí' if models.get('Gemini', {}).get('mentioned') else 'No',
                models.get('Gemini', {}).get('position', ''),
                'Sí' if models.get('Perplexity', {}).get('mentioned') else 'No',
                models.get('Perplexity', {}).get('position', ''),
                'Sí' if models.get('Bing/Copilot', {}).get('mentioned') else 'No',
                models.get('Bing/Copilot', {}).get('position', ''),
            ]
            
            rows_to_add.append(row)
        
        sheet.append_rows(rows_to_add)
        print(f"✅ {len(rows_to_add)} filas añadidas a Google Sheets")

        # --- Actualizar Dashboard ---
        dashboard = spreadsheet.worksheet("Dashboard")
        dashboard.update(range_name='B1', values=[[datetime.now().strftime("%Y-%m-%d %H:%M")]])
        
        # Calcular KPIs
        kpi_rows = []
        for model in ['ChatGPT', 'Claude', 'Gemini', 'Perplexity', 'Bing/Copilot']:
            model_data = [r for r in results if r['models'].get(model)]
            mentions = sum(1 for r in model_data if r['models'][model].get('mentioned'))
            total = len(results)
            rate = f"{(mentions/total*100):.1f}%" if total > 0 else "0.0%"
            
            positions = [r['models'][model].get('position') for r in model_data if r['models'][model].get('position') and isinstance(r['models'][model]['position'], int)]
            avg_pos = f"{sum(positions)/len(positions):.1f}" if positions else "-"
            
            kpi_rows.append([model, mentions, rate, avg_pos, datetime.now().strftime("%H:%M"), "-"])
        
        dashboard.update(range_name='A4:F8', values=kpi_rows)
        print("✅ Dashboard actualizado")
        
        # --- Actualizar Timeline ---
        timeline = spreadsheet.worksheet("Timeline")
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Calcular totales del día
        day_mentions = {m: 0 for m in ['ChatGPT', 'Claude', 'Gemini', 'Perplexity', 'Bing']}
        
        for res in results:
            for model_key, model_name in [('ChatGPT', 'ChatGPT'), ('Claude', 'Claude'), ('Gemini', 'Gemini'), ('Perplexity', 'Perplexity'), ('Bing/Copilot', 'Bing')]:
                 if res['models'].get(model_key, {}).get('mentioned'):
                     day_mentions[model_name] += 1
        
        total_mentions_day = sum(day_mentions.values())
        # Esto es un snapshot de "ahora", idealmente habría que acumular si se ejecuta múltiples veces, 
        # pero para simplicidad añadiremos una nueva fila por ejecución
        
        timeline_row = [
            current_date,
            day_mentions['ChatGPT'],
            day_mentions['Claude'],
            day_mentions['Gemini'],
            day_mentions['Perplexity'],
            day_mentions['Bing'],
            total_mentions_day,
            f"{(total_mentions_day / (len(results)*5) * 100):.1f}%" # Tasa global aprox
        ]
        
        timeline.append_row(timeline_row)
        print("✅ Timeline actualizado")

        # --- Actualizar Por Query ---
        por_query = spreadsheet.worksheet("Por Query")
        pq_rows = []
        for res in results:
            row = [
                res['query'],
                categorize_query(res['query']), # Usar la función de categorización
                1 if res['models'].get('ChatGPT', {}).get('mentioned') else 0,
                1 if res['models'].get('Claude', {}).get('mentioned') else 0,
                1 if res['models'].get('Gemini', {}).get('mentioned') else 0,
                1 if res['models'].get('Perplexity', {}).get('mentioned') else 0,
                1 if res['models'].get('Bing/Copilot', {}).get('mentioned') else 0,
            ]
            row.append(sum(row[2:])) # Total
            row.append(f"{(sum(row[2:])/5*100):.0f}%") # Tasa
            pq_rows.append(row)
        
        por_query.append_rows(pq_rows)
        print("✅ Hoja 'Por Query' actualizada")

        return True
    except Exception as e:
        print(f"❌ Error subiendo a Sheets: {e}")
        return False

def _get_raw_dataframe(spreadsheet):
    """Lee Raw Data y devuelve un DataFrame. Reutilizado por varias funciones."""
    raw_data = spreadsheet.worksheet("Raw Data")
    all_data = raw_data.get_all_values()[1:]
    if not all_data:
        return None
    return pd.DataFrame(all_data, columns=[
        'Timestamp', 'Fecha', 'Hora', 'Query', 'Categoría',
        'ChatGPT_M', 'ChatGPT_P', 'Claude_M', 'Claude_P',
        'Gemini_M', 'Gemini_P', 'Perplexity_M', 'Perplexity_P',
        'Bing_M', 'Bing_P'
    ])


def update_dashboard(spreadsheet):
    """Actualiza KPIs en Dashboard con tendencia real vs dia anterior"""
    try:
        dashboard = spreadsheet.worksheet("Dashboard")
        df = _get_raw_dataframe(spreadsheet)
        if df is None:
            return

        models = ['ChatGPT', 'Claude', 'Gemini', 'Perplexity', 'Bing']
        dates = sorted(df['Fecha'].unique())
        kpi_rows = []

        for model in models:
            mention_col = f'{model}_M'
            position_col = f'{model}_P'

            total_mentions = (df[mention_col] == 'Sí').sum()
            total_queries = len(df)
            mention_rate = (total_mentions / total_queries * 100) if total_queries > 0 else 0

            positions = df[df[mention_col] == 'Sí'][position_col]
            positions_numeric = pd.to_numeric(positions, errors='coerce').dropna()
            avg_position = positions_numeric.mean() if len(positions_numeric) > 0 else 'N/A'

            last_update = dates[-1] if dates else 'N/A'

            # Tendencia real: comparar ultimo dia vs dia anterior
            trend = '→'
            if len(dates) >= 2:
                today_df = df[df['Fecha'] == dates[-1]]
                yesterday_df = df[df['Fecha'] == dates[-2]]
                today_rate = (today_df[mention_col] == 'Sí').sum() / len(today_df) * 100 if len(today_df) > 0 else 0
                yesterday_rate = (yesterday_df[mention_col] == 'Sí').sum() / len(yesterday_df) * 100 if len(yesterday_df) > 0 else 0
                if today_rate > yesterday_rate:
                    trend = '↑'
                elif today_rate < yesterday_rate:
                    trend = '↓'

            kpi_rows.append([
                model,
                int(total_mentions),
                f'{mention_rate:.1f}%',
                f'{avg_position:.1f}' if avg_position != 'N/A' else 'N/A',
                last_update,
                trend
            ])

        dashboard.update('A4:F8', kpi_rows)
        print("✅ Dashboard actualizado")

    except Exception as e:
        print(f"⚠️  Error actualizando dashboard: {e}")


def update_timeline(spreadsheet):
    """Actualiza hoja Timeline: menciones agrupadas por fecha para graficos"""
    try:
        df = _get_raw_dataframe(spreadsheet)
        if df is None:
            return

        timeline = spreadsheet.worksheet("Timeline")
        models = ['ChatGPT', 'Claude', 'Gemini', 'Perplexity', 'Bing']
        dates = sorted(df['Fecha'].unique())

        rows = []
        for date in dates:
            day_df = df[df['Fecha'] == date]
            day_total = len(day_df)
            row = [date]
            total_mentions = 0
            for model in models:
                mention_col = f'{model}_M'
                mentions = int((day_df[mention_col] == 'Sí').sum())
                row.append(mentions)
                total_mentions += mentions
            row.append(total_mentions)
            rate = f'{(total_mentions / (day_total * len(models)) * 100):.1f}%' if day_total > 0 else '0%'
            row.append(rate)
            rows.append(row)

        # Sobreescribir datos (mantener headers)
        if rows:
            timeline.update(f'A2:H{len(rows) + 1}', rows)
            # Limpiar filas sobrantes si habia mas datos antes
            try:
                timeline.batch_clear([f'A{len(rows) + 2}:H1000'])
            except Exception:
                pass
        print("✅ Timeline actualizado")

    except Exception as e:
        print(f"⚠️  Error actualizando timeline: {e}")


def update_por_query(spreadsheet):
    """Actualiza hoja Por Query: analisis de visibilidad por query individual"""
    try:
        df = _get_raw_dataframe(spreadsheet)
        if df is None:
            return

        por_query = spreadsheet.worksheet("Por Query")
        models = ['ChatGPT', 'Claude', 'Gemini', 'Perplexity', 'Bing']
        queries = df['Query'].unique()

        rows = []
        for query in queries:
            query_df = df[df['Query'] == query]
            categoria = query_df['Categoría'].iloc[0] if len(query_df) > 0 else ''
            row = [query, categoria]
            total_mentions = 0
            for model in models:
                mention_col = f'{model}_M'
                mentions = int((query_df[mention_col] == 'Sí').sum())
                row.append(mentions)
                total_mentions += mentions
            row.append(total_mentions)
            total_checks = len(query_df) * len(models)
            rate = f'{(total_mentions / total_checks * 100):.1f}%' if total_checks > 0 else '0%'
            row.append(rate)
            rows.append(row)

        if rows:
            por_query.update(f'A2:I{len(rows) + 1}', rows)
            try:
                por_query.batch_clear([f'A{len(rows) + 2}:I100'])
            except Exception:
                pass
        print("✅ Por Query actualizado")

    except Exception as e:
        print(f"⚠️  Error actualizando por query: {e}")

# ===============================================
# EJECUCIÓN PRINCIPAL
# ===============================================

def run_monitoring():
    """Ejecuta el monitoreo completo"""
    
    print("\n" + "="*60)
    print("🚀 AI VISIBILITY MONITOR - GRUPO VAUGHAN")
    print("="*60)
    print(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔍 Queries: {len(QUERIES)}")
    print("="*60 + "\n")
    
    # Conectar con Google Sheets
    print("📊 Conectando con Google Sheets...")
    spreadsheet = init_google_sheets()
    
    if spreadsheet:
        setup_sheets(spreadsheet)
        print(f"🔗 URL: https://docs.google.com/spreadsheets/d/{spreadsheet.id}\n")
    else:
        print("⚠️  Continuando sin Google Sheets\n")
    
    all_results = []
    
    models_functions = [
        ('ChatGPT', check_chatgpt),
        ('Claude', check_claude),
        ('Gemini', check_gemini),
        ('Perplexity', check_perplexity),
        ('Bing/Copilot', check_copilot),
    ]
    
    for i, query in enumerate(QUERIES, 1):
        print(f"\n[{i}/{len(QUERIES)}] 🔍 {query}")
        print("-" * 60)
        
        query_results = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'models': {}
        }
        
        for model_name, check_function in models_functions:
            print(f"  {model_name:12} ... ", end='', flush=True)
            
            try:
                result = check_function(query)
                
                if result.get('error'):
                    print(f"❌ {result['error'][:200]}") # Mostrar mas detalle del error
                else:
                    mentioned = result['mentioned']
                    position = result.get('position', 'N/A')
                    
                    if mentioned:
                        print(f"✅ MENCIÓN (Pos: {position})")
                    else:
                        print("⚠️  No mencionado")
                
                query_results['models'][model_name] = result
                time.sleep(10)
                
            except Exception as e:
                print(f"❌ Error: {str(e)[:200]}")
                query_results['models'][model_name] = {'model': model_name, 'mentioned': False, 'error': str(e)}
        
        all_results.append(query_results)
    
    # Subir a Google Sheets
    if spreadsheet:
        print("\n" + "="*60)
        print("📤 Subiendo resultados a Google Sheets...")
        upload_to_sheets(spreadsheet, all_results)
        print(f"✅ Ver resultados: https://docs.google.com/spreadsheets/d/{spreadsheet.id}")
    
    # Generar resumen
    print("\n" + "="*60)
    print("📊 RESUMEN")
    print("="*60)
    
    for model_name in ['ChatGPT', 'Claude', 'Gemini', 'Perplexity', 'Bing/Copilot']:
        mentions = sum(1 for r in all_results if r['models'].get(model_name, {}).get('mentioned'))
        total = len(all_results)
        rate = (mentions / total * 100) if total > 0 else 0
        print(f"{model_name:12} {mentions:2}/{total:2} menciones ({rate:5.1f}%)")
    
    print("="*60)
    print("✅ Monitoreo completado\n")
    
    return all_results

if __name__ == "__main__":
    try:
        results = run_monitoring()
    except KeyboardInterrupt:
        print("\n\n⚠️  Monitoreo interrumpido por el usuario")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error fatal: {e}")
        sys.exit(1)