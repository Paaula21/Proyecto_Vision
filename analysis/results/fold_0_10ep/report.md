# Evaluación del detector — fold 0

El fold 0 valida el árbol/especie «apple», que no aparece en el entrenamiento. Por tanto, su AP por especie no mide solo detección: la clasificación de esa especie está fuera de distribución. AP_agnostic es la métrica principal para valorar si se encuentran frutas.

## Métricas

| Métrica | Entrenamiento | Validación |
|---|---:|---:|
| `AP` | 0.1422 | 0.0000 |
| `AP50` | 0.3230 | 0.0000 |
| `AP75` | 0.1101 | 0.0000 |
| `AP_agnostic` | 0.1607 | 0.2455 |
| `accuracy` | 0.5339 | 0.0000 |
| `AP_small` | 0.0991 | 0.0000 |
| `AP_medium` | 0.5475 | 0.0000 |
| `AP_large` | 0.8539 | 0.0000 |
| `AP_orange` | 0.2009 | nan |
| `AP_apple` | nan | 0.0000 |
| `AP_pear` | 0.1074 | nan |
| `AP_lemon` | 0.1183 | nan |

Imágenes: 50 de validación y 150 de entrenamiento.
Detecciones puntuadas: 7391 en validación y 25550 en entrenamiento.

## Errores diagnósticos en validación

- Coincidencias: 1750
- Falsos positivos: 3241
- Falsos negativos: 1710
- Errores de especie: 1750
- Errores de solapamiento (diagnóstico por caja): 596
- Falsos positivos con confianza ≥ 0,75: 212
- Falsos negativos pequeños: 1696
- Falsos negativos con indicios de oclusión: 1355

## Mejoras priorizadas

1. Revisar los falsos negativos pequeños y las máscaras de bajo AP75: el dataset está dominado por frutas pequeñas y parcialmente ocultas.
2. Si hay muchos falsos positivos de alta confianza, revisar resolución, augmentations y calibración de puntuaciones.
3. Tratar por separado la clasificación de la especie excluida del fold: su AP no es comparable con las especies vistas durante el entrenamiento.

La comparación entrenamiento/validación es diagnóstica y no modifica los pesos. Una diferencia grande entre ambas puede indicar sobreajuste.
