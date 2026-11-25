# Preparación del entorno

Instala Python 3.12 y ejecuta los comandos correspondientes desde la raíz del proyecto.

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

# Información relevante
## Época
Una época significa que el entrenamiento ha recorrido una vez todas las imágenes de entrenamiento del fold.

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

## Análisis de errores del issue #6

Para entrenar los cuatro folds LOTO, generar predicciones, comparar los
umbrales `0.05`, `0.25` y `0.50`, y producir el informe agregado:

```bash
python -m analysis.run_issue6_analysis --train --device cpu
```

El resultado queda en `.artifacts/issue6/`. El informe reproducible se escribe
en `.artifacts/issue6/issue6_report.md` y `analysis/issue6_report.md`; las
métricas agregadas y la recomendación se guardan en `aggregate_metrics.json` y
`recommendation.json`.

Si ya existen checkpoints en `.artifacts/issue6/fold_k/weights/`, se puede
reutilizar el análisis sin volver a entrenar:

```bash
python -m analysis.run_issue6_analysis \
  --weights-root .artifacts/issue6 \
  --skip-visualizations
```

El checkpoint disponible actualmente puede reutilizarse para una comprobación
provisional del fold 0. `--image-limit` limita la inferencia a imágenes de
validación y el informe queda marcado como provisional:

```powershell
.\.venv\Scripts\python.exe -m analysis.run_issue6_analysis `
  --weights .artifacts/inference/detector.pth --folds 0 --image-limit 2 `
  --device cpu --skip-visualizations
```

Para completar el análisis oficial de los cuatro folds, sin limitar imágenes:

```powershell
.\.venv\Scripts\python.exe -m analysis.run_issue6_analysis `
  --train --epochs 3 --device cpu
```

Para comparar la variante que no usa `iscrowd` como positivo, conserva otro
directorio de salida y añade `--exclude-iscrowd`:

```powershell
\.\.venv\Scripts\python.exe -m analysis.run_issue6_analysis `
  --train --epochs 3 --exclude-iscrowd --output .artifacts/issue6_no_iscrowd `
  --device cpu
```

El entorno actual ya puede ejecutar CPU: aunque la versión instalada lleva
las librerías `+cu124`, `torch.cuda.is_available()` es falso y el checkpoint
se carga con `--device cpu`. No es necesario reinstalar nada para validar el
flujo. Si se quiere un entorno CPU separado en Windows, sin modificar `.venv`:

```powershell
py -3.12 -m venv .venv-cpu
.\.venv-cpu\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy==2.5.4 pillow==12.3.0 scipy==1.18.1 pycocotools==2.0.11 matplotlib==3.10.7
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python main.py detect --input fruits_part1/train --output .artifacts/inference/predictions_cpu.json --weights .artifacts/inference/detector.pth --device cpu
```

La inferencia medida en este portátil tarda aproximadamente 35–40 segundos
por imagen; por eso los cuatro folds y varios experimentos de épocas deben
ejecutarse como trabajos largos. No se deben usar las imágenes de test para
elegir umbral, épocas o pesos.
