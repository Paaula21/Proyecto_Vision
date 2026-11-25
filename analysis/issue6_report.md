# Issue #6 — análisis de experimentos y errores

**Estado: provisional.** Este resultado usa una muestra limitada y no debe compararse como resultado LOTO final.

La selección usa la media de `AP_agnostic` en los folds disponibles. `AP75` y la proporción de máscaras imprecisas sirven como desempate.

Folds evaluados: 0. Imágenes por fold: todas las de validación.

## Configuración recomendada: `0.05`

| Umbral | AP agnóstico medio | AP75 medio | AP pequeño medio | Folds |
|---:|---:|---:|---:|---:|
| 0.05 | 0.0000 | 0.0000 | 0.0000 | 1 |
| 0.25 | 0.0000 | 0.0000 | 0.0000 | 1 |
| 0.5 | 0.0000 | 0.0000 | 0.0000 | 1 |

## Errores diagnósticos agregados

| Umbral | FP | FN | especie | máscaras imprecisas | pequeños FN |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 4999 | 3459 | 1 | 1 | 2951 |
| 0.25 | 2397 | 3459 | 1 | 1 | 2951 |
| 0.5 | 70 | 3460 | 0 | 0 | 2951 |

La especie excluida se conserva como condición de interpretación de cada fold: manzana, pera, naranja y limón, respectivamente. Las métricas por especie no se usan para seleccionar el umbral.
