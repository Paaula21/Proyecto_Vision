# Split leave-one-tree-out (LOTO)

Este directorio contiene el split oficial de validación de la Parte 1. **La
persona 1 lo consume tal cual: no se edita a mano.** Si hay que corregirlo, se
genera una versión nueva y se avisa.

## Qué contiene

| Archivo | Contenido |
|---|---|
| `make_split.py` | Construye y verifica el split a partir de `annotations.json` |
| `folds.json` | Los 4 folds con sus listas de ids y metadatos |
| `fold_k/train_ids.txt` | Ids de imagen de entrenamiento del fold `k`, uno por línea |
| `fold_k/val_ids.txt` | Ids de imagen de validación del fold `k`, uno por línea |
| `checks.json` | Resultado de las comprobaciones y del estudio de robustez |

Los ids son los `id` de `images.json`, no los nombres de archivo.

## Protocolo

**Leave-one-tree-out (LOTO), con el árbol como unidad.** En cada fold se valida
con un árbol completo (50 imágenes) y se entrena con los otros tres (150). Hay
cuatro folds y cada imagen es validación exactamente una vez.

No hay semilla aleatoria: el split es determinista. Los componentes se ordenan
por su id de imagen mínimo y se asignan en ese orden a los folds 0–3.

| Fold | Árbol de validación | Especie | Imágenes train | Imágenes val |
|---:|---:|---|---:|---:|
| 0 | 0 | manzana | 150 | 50 |
| 1 | 1 | pera | 150 | 50 |
| 2 | 2 | naranja | 150 | 50 |
| 3 | 3 | limón | 150 | 50 |

## Por qué por árbol y no por imagen

La misma fruta física aparece en varias imágenes: 839 `fruit_id` distintos, todos
vistos en al menos dos imágenes y hasta en 37. Un split aleatorio por imagen
metería la misma fruta, el mismo fondo y la misma iluminación en train y en val,
e inflaría la métrica.

El grafo de imágenes que comparten `fruit_id` da **4 componentes de 50 imágenes**,
cada uno de una sola especie. Es la evidencia de que hay 4 árboles, uno por
especie. La agrupación es robusta: usando solo frutas vistas en al menos 15
imágenes siguen saliendo 4 componentes de 50. El detalle está en
`checks.json → robustness_min_views`.

## Garantías

Hay que distinguir dos niveles:

- **Sin fuga de fruta:** ninguna `fruit_id` aparece en train y val del mismo fold.
  Es lo que impide que el modelo vea la misma fruta en los dos lados.
- **Sin fuga de árbol:** ningún árbol aparece en train y val del mismo fold. Es lo
  que impide que el modelo reconozca el mismo árbol y su fondo.

En este dataset se cumplen las dos, porque los componentes son árboles completos.
`checks.json` lo verifica fold a fold.

## Cómo regenerar y verificar

Desde la raíz del repositorio:

```bash
python3 splits/make_split.py
```

El script termina con código 0 solo si todas las comprobaciones pasan. Si algo
falla, escribe `passed: false` y los errores en `checks.json`, y termina con
código distinto de 0. Comprueba, entre otras cosas:

- 4 componentes de 50 imágenes;
- cada componente con una sola especie;
- `train ∩ val = ∅` y `train ∪ val = 200` en cada fold;
- 150 imágenes en train y 50 en val;
- ninguna `fruit_id` en ambos lados;
- cada `fruit_id` en un solo componente;
- sha256 de `annotations.json`.

## Casos de fallback

El script no decide solo. Si el grafo no da exactamente 4 componentes de 50 con
una especie cada uno, aborta y hay que revisar:

- **Más de 4 componentes:** no asignar componentes sueltos a folds, porque un
  mismo árbol podría quedar a ambos lados. Fusionar solo con evidencia visual
  (continuidad de fondo, iluminación, geometría del follaje, correspondencia de
  ramas). Si no hay evidencia, fusión conservadora por especie y documentarla.
- **Menos de 4 componentes:** posible colisión de `fruit_id` entre árboles;
  abortar y revisar.
- **Un componente con más de una especie:** error de datos; abortar.

Si se usa la fusión conservadora por especie, el split sigue siendo seguro a
nivel de fruta y de árbol, pero deja de separar árboles individuales y hay que
decirlo en la documentación.

## Cómo usar los folds

- **Detección y segmentación:** usar LOTO con `AP_agnostic` como métrica
  principal, más `AP50`, `AP75` y AP por tamaño. Es una estimación conservadora
  de la generalización a árboles nuevos.
- **Especie:** **no** se puede validar con LOTO, porque la especie del árbol de
  validación no aparece en entrenamiento y su AP será 0 por construcción. Para
  la especie hace falta un split distinto, dentro de cada árbol y agrupado por
  `fruit_id`, que es optimista porque comparte árbol y fondo. Debe presentarse
  como cota superior, no como resultado real.
- **Modelo final:** reentrenar con las **200 imágenes**. El split es un
  instrumento de validación, no el conjunto de entrenamiento de la entrega. Un
  modelo entrenado solo con 3 árboles se quedaría sin una especie.

## Limitaciones

- Solo hay 4 árboles, uno por especie. No se puede estimar la generalización de
  la especie a árboles nuevos.
- El test es externo y oculto. Se supone que viene del mismo pipeline sintético,
  pero no está verificado; si cambia la apariencia, el resultado de LOTO estará
  sobreestimado.
- La identificación de árboles se infiere del grafo de `fruit_id`; no hay EXIF ni
  metadatos de árbol.
- El dataset es sintético y tiene texturas y fondos homogéneos; el riesgo de
  sobreajuste al árbol es real.

## Integridad

Hash SHA-256 de `annotations.json` con el que se generó este split:

```
1708cfdd5c9e1f59bfec4df780e3660fc7f4913b6e9ab51a2e66d1e99320e67a
```

Estado actual: `checks.json → passed = true`, sin errores ni avisos.
