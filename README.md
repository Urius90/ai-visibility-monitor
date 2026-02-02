# 🚀 AI Visibility Monitor - Grupo Vaughan

Este proyecto automatiza el monitoreo de la visibilidad de marca de **Grupo Vaughan** en los principales modelos de Inteligencia Artificial Generativa.

## 📊 ¿Qué hace este monitor?

Ejecuta una serie de queries estratégicas (transaccionales, informacionales y comparativas) relacionadas con la enseñanza de inglés y analiza las respuestas de los siguientes modelos para detectar si mencionan a "Vaughan":

| Modelo | Versión Configurada | Estrategia |
| :--- | :--- | :--- |
| **Gemini** | `gemini-2.5-flash-lite` | **User-Centric**: Simula la experiencia de usuarios móviles/gratuitos (rápido y conciso). |
| **Claude** | `claude-3-haiku-20240307` | **Cost-Efficiency**: Versión optimizada y económica. |
| **ChatGPT** | `gpt-3.5-turbo` | Estándar de mercado. |
| **Perplexity** | (API) | *Pendiente de API Key* |
| **Bing/Copilot** | (API) | *Pendiente de API Key* |

Los resultados se guardan automáticamente en un dashboard de **Google Sheets**.

---

## 🛠️ Configuración en GitHub

Este proyecto está diseñado para ejecutarse automáticamente mediante **GitHub Actions**.

### Secretos Requeridos
Para que funcione, debes configurar los siguientes secretos en el repositorio (`Settings` > `Secrets and variables` > `Actions`):

- `OPENAI_API_KEY`: Tu clave de OpenAI.
- `ANTHROPIC_API_KEY`: Tu clave de Anthropic.
- `GOOGLE_API_KEY`: Tu clave de Google AI Studio.
- `GSPREAD_CREDENTIALS`: El contenido completo de tu JSON de cuenta de servicio de Google (para Sheets).

---

## 🚀 Cómo Ejecutar

1. Ve a la pestaña **[Actions](https://github.com/Urius90/ai-visibility-monitor/actions)** en este repositorio.
2. Selecciona el workflow **"Run AI Visibility Monitor"**.
3. Pulsa el botón verde **"Run workflow"**.

El proceso tardará unos minutos y al finalizar verás los resultados actualizados en el Google Sheet vinculado.

---

## 📂 Estructura del Proyecto

- `src/monitor.py`: Código principal. Aquí se definen las `QUERIES` y la lógica de cada modelo.
- `.github/workflows/monitor.yml`: Configuración del automatismo (cron o manual).
- `requirements.txt`: Dependencias (incluye `google-genai` para soporte de Gemini 2.5).

## 📝 Personalización

### Añadir nuevas preguntas
Edita `src/monitor.py` y añade tu query a la lista `QUERIES`:

```python
QUERIES = [
    "mejores academias de inglés en España",
    "tu nueva pregunta aquí...",
]
```

---
*Desarrollado para el equipo de Marketing Digital de Grupo Vaughan.*
