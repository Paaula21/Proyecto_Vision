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
