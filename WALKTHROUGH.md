# 🚀 AI Visibility Monitor - Documentación Final

## Estado del Sistema
El sistema de monitoreo está **OPERATIVO** y configurado para ejecutarse en **GitHub Actions**.

### 🤖 Modelos Configurados (Estrategia "User-Centric")
Se han seleccionado los modelos que mejor representan al usuario promedio y optimizan costes:

| Modelo | Versión Técnica | Motivo |
| :--- | :--- | :--- |
| **Gemini** | `gemini-2.5-flash-lite` | Modelo 2025, muy rápido y usado en móviles. Simula entorno real. |
| **Claude** | `claude-3-haiku-20240307` | Versión estable y económica (3x más barato que 3.5). |
| **ChatGPT** | (Vía OpenAI API) | Estándar de mercado. |

## 🛠️ Cómo Ejecutar
1. Ir a la pestaña **Actions** en GitHub.
2. Seleccionar el workflow **Run AI Visibility Monitor**.
3. Pulsar **Run workflow** (botón verde).

## 🔮 Futuras Modificaciones
Si en el futuro deseas ampliar queries o cambiar modelos, aquí tienes la guía rápida:

### 1. Añadir más Queries
Editar el archivo `src/monitor.py` y buscar la lista `QUERIES`:
```python
QUERIES = [
    "mejores academias de inglés en España",
    "tu nueva query aqui",  # <--- Añadir aquí
    ...
]
```

### 2. Cambiar Modelos
Editar las funciones `check_claude` o `check_gemini` en `src/monitor.py` y cambiar el `model="..."`.

### 3. Google Sheets
Los resultados se vuelcan automáticamente en:
[Ver Spreadsheet](https://docs.google.com/spreadsheets/d/1Zj47IExAqH0wP6yKO3VBIDyxRsaTCCZ8404jOCoAWMY)
