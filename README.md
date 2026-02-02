# AI Visibility Monitor - Grupo Vaughan

Monitoreo automatizado de visibilidad de marca en respuestas de modelos de IA principales.

## Que hace

Consulta diariamente 5 modelos de IA con queries relacionadas con academias de ingles y detecta si mencionan la marca Vaughan. Los resultados se almacenan en Google Sheets con dashboards automaticos.

## Modelos monitoreados

| Modelo | API | Modelo especifico |
|--------|-----|-------------------|
| ChatGPT | OpenAI | GPT-4o |
| Claude | Anthropic | Sonnet 4 |
| Gemini | Google AI | 2.0 Flash |
| Perplexity | Perplexity | Sonar |
| Bing/Copilot | Bing Search | v7.0 |

## Queries monitoreadas

**Transaccionales:** mejores academias, cursos online, clases para empresas, academia Madrid, cursos intensivos

**Informacionales:** como aprender rapido, diferencia B1/B2, metodo adultos, cuanto tiempo

**Branded:** metodo Vaughan funciona, opiniones Grupo Vaughan

## Keywords de marca detectadas

`vaughan`, `grupo vaughan`, `richard vaughan`, `vaughantown`

## Setup

### 1. APIs necesarias

Obtener API keys de:
- **OpenAI** (obligatoria): https://platform.openai.com/api-keys
- **Anthropic** (obligatoria): https://console.anthropic.com/
- **Google AI** (obligatoria): https://aistudio.google.com/app/apikey
- **Perplexity** (opcional): https://docs.perplexity.ai/
- **Bing Search** (opcional): https://www.microsoft.com/en-us/bing/apis/bing-web-search-api

### 2. Google Sheets Service Account

1. Ir a [Google Cloud Console](https://console.cloud.google.com/)
2. Crear proyecto o seleccionar existente
3. Habilitar Google Sheets API y Google Drive API
4. Crear Service Account (IAM > Service Accounts)
5. Generar clave JSON y descargar
6. Compartir el Google Sheet con el email del Service Account

### 3. Configurar secrets en GitHub

Ir a **Settings > Secrets and variables > Actions** del repositorio y crear:

| Secret | Descripcion |
|--------|-------------|
| `OPENAI_API_KEY` | API key de OpenAI |
| `ANTHROPIC_API_KEY` | API key de Anthropic |
| `GOOGLE_API_KEY` | API key de Google AI Studio |
| `PERPLEXITY_API_KEY` | API key de Perplexity (opcional) |
| `BING_API_KEY` | API key de Bing Search (opcional) |
| `GOOGLE_CREDENTIALS` | Contenido completo del JSON de Service Account |

### 4. Crear el Google Sheet

Crear un spreadsheet llamado **"AI Visibility Monitor - Grupo Vaughan"** y compartirlo con el email del Service Account. El script creara automaticamente las hojas necesarias en la primera ejecucion.

## Ejecucion

### Automatica

El workflow de GitHub Actions se ejecuta diariamente a las **8:00 AM UTC** (10:00 AM Madrid).

### Manual

Ir a **Actions > AI Visibility Monitor > Run workflow** en GitHub.

### Local (desarrollo)

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="AI..."
python src/monitor.py
```

## Estructura del Google Sheet

- **Raw Data**: datos brutos de cada ejecucion (timestamp, query, mencion por modelo)
- **Dashboard**: KPIs agregados (tasa de mencion, posicion promedio, tendencia)

## Estructura del proyecto

```
ai-visibility-monitor/
├── .github/
│   └── workflows/
│       └── monitor.yml
├── src/
│   └── monitor.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Autor

**Arturo** - Digital Marketing Specialist @ Grupo Vaughan
