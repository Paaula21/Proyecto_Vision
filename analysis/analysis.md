# Análisis del train set de la Parte 1

Este documento analiza las 200 imágenes de `fruits_part1/train` y las anotaciones
COCO que las acompañan. No evalúa ningún modelo ni implementa ningún detector:
su objetivo es caracterizar los datos y justificar el protocolo de validación.

Todo lo que aparece aquí sale de `analysis/make_analysis.py` (cifras y figuras) o
está marcado explícitamente como observación visual. Los valores completos están
en `analysis/metrics.json`; las figuras, en `analysis/figures/`.

## Cómo reproducirlo

Desde la raíz del repositorio:

```bash
python3 analysis/make_analysis.py
```

Dependencias: `numpy`, `pillow`, `scipy` y `matplotlib` (solo para las figuras).
El script no descarga nada, no modifica el dataset y no necesita `pycocotools`:
incluye un decodificador propio del RLE de COCO. Si `matplotlib` no está
instalado:

```bash
python3 -m pip install matplotlib
```

---

## 1. Resumen ejecutivo

- **200 imágenes** de 1280×960 y **16.002 anotaciones**: 11.076 frutas visibles y
  4.926 fragmentos ignorados (`iscrowd = 1`, 30,78%).
- **839 frutas físicas** distintas (`fruit_id`), todas vistas en al menos dos
  imágenes.
- El grafo de imágenes que comparten `fruit_id` produce **4 componentes de 50
  imágenes**, cada uno de **una sola especie**. Es muy probable que sean 4
  árboles, uno por especie, y es la base del split.
- El **87,51% de las frutas visibles son "small"** para COCO (<32×32 px); el bbox
  mediano mide 23×23 px y el área visible mediana es de 171,5 px.
- El **66,8% de las máscaras visibles tiene 2 o más piezas** (hasta 18 con
  conectividad 8): las hojas y ramas ocultan partes de la fruta.
- El **57,23% de las frutas visibles contiene alguna pieza de menos de 20 px**,
  lo que indica que la regla de 20 px se aplica al área visible total de la
  fruta, no a cada pieza.
- Las imágenes son **renders 3D sintéticos**, no fotografías. Las máscaras
  parecen exactas y siguen la oclusión a nivel de píxel.
- La principal limitación es estructural: **solo hay 4 árboles, uno por especie**.
  Eso impide separar "generalización a árbol nuevo" de "generalización a especie
  nueva".

---

## 2. Origen de los datos: renders sintéticos

La inspección visual de imágenes de los cuatro grupos (por ejemplo
`img_000001.jpg`, `img_000002.jpg`, `img_000004.jpg` y `img_000009.jpg`) muestra
un árbol 3D: hojas con la misma forma repetida, troncos y ramas cilíndricos,
sombras coherentes con un motor de render y fondos que parecen imágenes o
geometría de escenario. No son fotografías reales.

Consecuencias:

- Es probable que las máscaras provengan de un pase de visibilidad del
  renderizador y no de anotación manual. La comprobación de bordes (sección 6)
  lo respalda.
- Las texturas, la iluminación y los fondos son mucho más homogéneos que en un
  dataset real. Un modelo puede aprender el aspecto de estos cuatro árboles en
  vez de aprender "fruta" en general.
- **No está verificado** que el test se genere con el mismo pipeline. Es la mayor
  incertidumbre del proyecto y se documenta como limitación.

---

## 3. Especies

| Especie | `category_id` | Imágenes | Anotaciones | Visibles | Ignoradas | `fruit_id` |
|---|---:|---:|---:|---:|---:|---:|
| naranja | 1 | 50 | 3.363 | 2.332 | 1.031 | 269 |
| manzana | 2 | 50 | 4.618 | 3.460 | 1.158 | 165 |
| pera | 3 | 50 | 4.214 | 2.863 | 1.351 | 175 |
| limón | 4 | 50 | 3.807 | 2.421 | 1.386 | 230 |

Cada especie aparece en 50 imágenes y cada imagen es monoespecífica. El número
de anotaciones no refleja el número de frutas físicas: las naranjas tienen más
`fruit_id` (269) que las manzanas (165), pero menos frutas visibles (2.332 frente
a 3.460), porque muchas naranjas están más ocultas o son más pequeñas.

**Confusión importante:** en el train set, árbol y especie están perfectamente
correlacionados (un árbol por especie). No se puede distinguir si un modelo
generaliza a un árbol nuevo o a una especie nueva. Esta limitación condiciona
todo el protocolo de validación (ver `splits/README.md`).

Figura: `figures/fig_species.png`.

---

## 4. Los cuatro árboles

A partir de las imágenes que comparten al menos una fruta física se obtienen
cuatro componentes disjuntos:

| Árbol | Especie | Imágenes | `fruit_id` | Visibles | Ignoradas |
|---:|---|---:|---:|---:|---:|
| 0 | manzana | 50 | 165 | 3.460 | 1.158 |
| 1 | pera | 50 | 175 | 2.863 | 1.351 |
| 2 | naranja | 50 | 269 | 2.332 | 1.031 |
| 3 | limón | 50 | 230 | 2.421 | 1.386 |

La agrupación es robusta: si se reconstruye el grafo usando solo frutas vistas en
al menos 15 imágenes, siguen apareciendo exactamente 4 componentes de 50. El
detalle y las comprobaciones están en `splits/`.

Figura: `figures/fig_trees.png`.

---

## 5. Tamaños

| Estadístico | Área visible (px) | Ancho bbox (px) | Alto bbox (px) |
|---|---:|---:|---:|
| mínimo | 20 | 3 | 3 |
| percentil 10 | 32 | 10 | 11 |
| mediana | 171,5 | 23 | 23 |
| media | 533,1 | 28,5 | 28,8 |
| percentil 90 | 1.308,5 | 54 | 54 |
| máximo | 16.592 | 179 | 192 |

Reparto oficial de COCO por área:

| Rango | Frutas | Porcentaje |
|---|---:|---:|
| small (<32² = 1.024 px) | 9.693 | 87,51% |
| medium (32²–96²) | 1.350 | 12,19% |
| large (>96²) | 33 | 0,30% |

Consecuencia: el problema es de **objetos pequeños**. Una fruta mediana ocupa
23×23 px y el área visible típica es de ~170 px. Cualquier reducción de resolución
destruye información; una máscara con un error de un píxel cambia mucho el IoU.
El AP estará dominado por `AP_small` y por la precisión del contorno.

Figuras: `figures/fig_area_hist.png`, `figures/fig_bbox_hexbin.png`.

---

## 6. Máscaras y oclusión

- **Máscaras multicomponente:** 7.399 de 11.076 frutas visibles (66,8%) tienen 2
  o más piezas con conectividad 8. La mediana es 2 piezas y el máximo 18. Con
  conectividad 4 el porcentaje sube y el máximo llega a 27, porque algunos píxeles
  solo se tocan en diagonal; el análisis usa 8, que es la conectividad natural
  para una máscara.
- **Relleno del bbox:** mediana 0,388 y media 0,402. Es decir, la máscara ocupa
  menos de la mitad del rectángulo que la contiene: las frutas son irregulares y
  están ocultas por hojas y ramas.
- **Frutas cortadas por el borde:** 410 (3,7% de las visibles). Hay que
  detectarlas y segmentarlas igual que las demás.
- **Formato:** las máscaras son RLE de COCO. El área calculada al decodificar
  coincide con el campo `area` en una muestra aleatoria de 300 anotaciones
  (`checks.rle_area_matches_annotation = true`).
- **Bordes:** al ampliar una fruta con 17 piezas se observa que el contorno sigue
  el borde de hojas y ramas con precisión de píxel. Es coherente con un pase de
  visibilidad del renderizador. **No se ha verificado** el caso de hojas
  semitransparentes: en la muestra revisada no aparecía ninguna.

Consecuencia: la máscara de una fruta **no es un blob conexo**. Cualquier método
que asuma una sola región, o que convierta la máscara a un polígono, puede perder
piezas y penalizar el IoU.

Figuras: `figures/fig_pieces.png`, `figures/fig_bbox_fill.png`,
`figures/fig_examples_hard.png`, `figures/fig_examples_species.png`.

---

## 7. Anotaciones ignoradas (`iscrowd`)

- Hay **4.926** anotaciones con `iscrowd = 1`: el **30,78%** del total.
- Su área va de 1 a 19 px, con mediana 5 px y media 6,7 px. Nunca llegan a 20 px,
  lo que confirma el umbral descrito en el enunciado.
- **No solapan con ninguna máscara visible**: el solape medio es 0,00 y el 0% de
  ellas alcanza un IoU de 0,5 con una fruta visible. Solo el 12,26% queda
  adyacente tras dilatar 1 px la unión de las visibles.
- El **57,23% de las frutas visibles tiene alguna pieza de menos de 20 px** y aun
  así es `iscrowd = 0`. Por tanto, el umbral de 20 px se aplica al **área visible
  total de la fruta**, no a cada pieza por separado.

Interpretación: las anotaciones `iscrowd` son frutas completas con menos de 20 px
visibles, no trozos desgajados de frutas ya anotadas. En evaluación se ignoran:
detectarlas no da crédito y no detectarlas no penaliza. En entrenamiento, la
decisión de usarlas como positivas o no es un experimento aparte; el argumento a
favor es no enseñar al modelo que una fruta real es fondo, y el argumento en
contra es que son objetos extremos de 1–19 px.

---

## 8. Densidad y variedad de escenas

| Métrica | Mínimo | Mediana | Media | Máximo |
|---|---:|---:|---:|---:|
| Anotaciones por imagen | 11 | 79,5 | 80,0 | 140 |
| Frutas visibles por imagen | 10 | 57 | 55,4 | 110 |

El máximo de 110 frutas visibles deja mucho margen frente al límite de 300
detecciones por imagen de la evaluación. Las escenas varían entre vistas de árbol
completo y primeros planos a ~1 m, con fondos, luces y distancias diferentes. Aun
así, al ser sintéticas, la variedad es menor que en un dataset real.

Figura: `figures/fig_density.png`.

---

## 9. Implicaciones para el método y la evaluación

Sin elegir técnica, el análisis fija varias condiciones:

1. **Resolución:** no conviene reducir a 640 px; las frutas de 23 px se
   convertirían en ~11 px. Hay que trabajar cerca de la resolución nativa o
   aumentarla, y vigilar el coste de memoria.
2. **Máscaras:** el método debe poder producir y evaluar máscaras con varias
   piezas. Convertir a polígonos o imponer una sola región pierde información.
3. **Objetos pequeños:** la métrica principal estará dominada por `AP_small`;
   hay que reportar AP por tamaño y no solo el AP global.
4. **Especie:** al estar confundida con el árbol, la clasificación de especie
   necesita un protocolo de validación distinto del LOTO (ver `splits/README.md`).
5. **Riesgo de atajo:** al ser renders, un modelo puede aprender textura, fondo o
   iluminación de estos cuatro árboles. La validación por árbol es la que mejor
   detecta ese atajo.
6. **Umbral de detección:** en AP, descartar detecciones de baja confianza solo
   reduce el recall; el umbral debe ser bajo y calibrarse en validación.
7. **Tope de detecciones:** 300 por imagen es holgado con un máximo de 110 frutas
   visibles, así que no debería ser una restricción activa.

---

## 10. Limitaciones y cosas no verificadas

- **Solo 4 árboles, uno por especie.** No se puede estimar la generalización de
  la especie a árboles nuevos. El LOTO mide "árbol nuevo", no "especie nueva".
- **Origen sintético.** No está verificado que el test use el mismo pipeline ni
  los mismos assets. Si cambia la apariencia, el rendimiento en test será menor
  que en validación.
- **Identificación de árboles.** Se infiere del grafo de `fruit_id`, no de
  metadatos (no hay EXIF). Depende de que `fruit_id` signifique realmente "fruta
  física"; la evidencia es muy fuerte, pero es una suposición.
- **Máscaras exactas.** La comprobación visual y el área coincidente lo respaldan,
  pero no se ha revisado cada una de las 16.002 anotaciones ni el caso de hojas
  semitransparentes.
- **Interpretación de `iscrowd`.** La conclusión de que son frutas completas de
  <20 px es la más consistente con los datos, pero no está confirmada por el
  enunciado más allá de la regla de los 20 px.
- **Este análisis no mide ningún modelo.** Los números de rendimiento llegarán
  con la fase de detección.

---

## 11. Comprobaciones de integridad

El script deja en `metrics.json` estas comprobaciones, todas verdaderas:

- el área decodificada coincide con `area` en la muestra;
- cada `fruit_id` tiene una sola categoría;
- las 200 imágenes están cubiertas por un componente;
- los componentes son disjuntos;
- hay exactamente 4 componentes de 50 imágenes;
- cada componente tiene una sola especie.

El hash SHA-256 de `annotations.json` usado en este análisis es:

```
1708cfdd5c9e1f59bfec4df780e3660fc7f4913b6e9ab51a2e66d1e99320e67a
```
