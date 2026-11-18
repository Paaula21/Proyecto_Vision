# Preparación del entorno

Instala Python 3.12 y ejecuta los comandos correspondientes desde la raíz del proyecto.
El repositorio incluye las 200 imágenes de entrenamiento en `fruits_part1/train/images/`.

## Windows (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Si PowerShell no permite activar el entorno, instala las dependencias directamente con:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Si la creación del entorno falla porque falta `venv`, instala el paquete correspondiente a Python 3.12 para tu distribución.

## Análisis de los datos

Desde la raíz del proyecto y con el entorno preparado, ejecuta:

```bash
python analysis/make_analysis.py
```

El comando regenera las métricas en `analysis/metrics.json` y las figuras en `analysis/figures/`.

## Entrenamiento del modelo

Desde la raíz del proyecto y con el entorno preparado, genera primero las divisiones de los datos:

```bash
python splits/make_split.py
```

Después, inicia una prueba de entrenamiento con el fold 0 y una época en CPU:

```bash
python train.py --data fruits_part1/train --splits splits --fold 0 --output .scratch/entrenamiento_fold_0 --device cpu --epochs 1
```
Se usa 1 época ya que el proceso es muy largo, pero se deben usar más.
La carpeta indicada en `--output` debe ser nueva. Para otra prueba, cambia su nombre; si dispones de una GPU CUDA, puedes cambiar `--device cpu` por `--device cuda`.

Para entrenar el modelo final con las 200 imágenes y las cuatro especies durante 10 épocas en CUDA, sin separar un fold de validación:

```bash
python train.py --data fruits_part1/train --all-data --output .artifacts/training/final_10ep --device cuda --epochs 10
python main.py detect --input fruits_part1/train --output .artifacts/training/final_10ep/predictions.json --weights .artifacts/training/final_10ep/detector.pth --device cuda
```

La primera ejecución del entrenamiento puede descargar los pesos iniciales de Mask R-CNN. Las predicciones de este ejemplo corresponden a las imágenes de entrenamiento.

# Información relevante
## Época
Una época significa que el entrenamiento ha recorrido una vez todas las imágenes seleccionadas.

## Evaluación del detector

Después de tener los pesos locales, el detector se puede evaluar con las
imágenes de entrenamiento y validación del fold. El script ejecuta inferencia
una sola vez, aplica el evaluador oficial a ambos conjuntos, clasifica errores
y genera visualizaciones de predicciones y anotaciones reales:

```bash
python analysis/evaluate_detector.py \
  --fold 0 \
  --data fruits_part1/train \
  --splits splits \
  --weights .artifacts/inference/detector.pth \
  --output .artifacts/evaluation/fold_0 \
  --device cpu
```

El resultado queda en `.artifacts/evaluation/fold_0/`, que está excluido de
Git. `report.md` compara entrenamiento y validación; la métrica principal para
valorar si se encuentran frutas es `AP_agnostic`, porque en cada fold la
especie del árbol de validación queda fuera del entrenamiento. La evaluación es
de solo lectura y no modifica los pesos.
