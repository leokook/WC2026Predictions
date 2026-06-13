# WC 2026 Goal Prediction Model

Este proyecto implementa un modelo de Poisson Dixon-Coles para predecir resultados y simular la fase de grupos de la Copa del Mundo FIFA 2026.

## Características
- **Carga de datos**: Obtiene resultados históricos internacionales automáticamente.
- **Modelo Dixon-Coles**: Ajusta parámetros de ataque y defensa por equipo, considerando la ventaja de localía y el decaimiento temporal (forma reciente).
- **Simulación Monte Carlo**: Simula la fase de grupos miles de veces para estimar probabilidades de clasificación.
- **Criterios de Desempate**: Implementa la lógica oficial de la FIFA (Puntos, Diferencia de Goles, Goles a Favor).

## Instalación

1. Clona el repositorio.
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Uso

Ejecuta la simulación principal:
```bash
python main.py
```

Opciones principales:
- `--sims 50000`: Aumenta las iteraciones de la simulación.
- `--no-download`: Usa los datos en caché en lugar de descargar de nuevo.
- `--compare`: Analiza la precisión del modelo contra partidos ya jugados del Mundial 2026.