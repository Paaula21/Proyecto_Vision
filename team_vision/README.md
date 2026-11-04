# Proyecto de visión: detección de frutas

Esta carpeta contiene la configuración inicial para la parte 1. El detector y
el punto de entrada del proyecto se desarrollarán en la issue #3; todavía no hay
código de aplicación en esta carpeta.

## Preparación

Se necesita Python 3.12. Desde la raíz del repositorio, en PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r team_vision/requirements.txt
```

Si la ejecución de scripts de PowerShell está deshabilitada, se puede usar
`.\.venv\Scripts\python.exe` directamente en lugar de activar el entorno.
El archivo de dependencias incluye también las necesarias para las herramientas
del starter kit. Sus versiones son las del ejemplo oficial para Python 3.12.

Obtened el starter kit en `Final project/Part 1/` del repositorio de la
asignatura `uma_cv_2027` y descomprimid los datos del Campus Virtual. Los
comandos siguientes suponen que el kit está en `Startert Kit/` y los datos en
`fruits_part1/fruits_part1/train/`, como en esta copia de trabajo. Adaptad esas
rutas si habéis guardado los archivos en otra ubicación.

## Ejecutar y evaluar el ejemplo

Desde la raíz del repositorio, con el entorno activado:

```powershell
New-Item -ItemType Directory -Force .scratch | Out-Null
python 'Startert Kit/example_submission/main.py' detect --input fruits_part1/fruits_part1/train --output .scratch/predictions.json --device cpu
python 'Startert Kit/evaluate.py' --gt fruits_part1/fruits_part1/train/annotations.json --pred .scratch/predictions.json
python 'Startert Kit/visualize.py' --images fruits_part1/fruits_part1/train/images --coco fruits_part1/fruits_part1/train/annotations.json --out .scratch/visualizaciones --limit 10
```

El ejemplo puede tardar en procesar las 200 imágenes y sus puntuaciones serán
bajas. Se ejecuta en CPU aunque se indique `cuda`; es solo una referencia de
funcionamiento. El proyecto todavía no dispone de un detector entrenado ni de
una evaluación de validación: esas tareas corresponden a las siguientes issues.

Antes de la entrega se debe ejecutar `check_submission.py` y completar este
README con los miembros, el método definitivo, resultados y recursos externos.
La carpeta de entrega no debe contener el conjunto de datos, el entorno virtual,
las visualizaciones ni los archivos temporales.
