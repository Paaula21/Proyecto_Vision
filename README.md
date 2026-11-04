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
