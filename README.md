# Preparación del entorno

Instala Python 3.12 y ejecuta estos comandos desde la raíz del proyecto en PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Si PowerShell no permite activar el entorno, instala las dependencias directamente con:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
