import multiprocessing
import os

# Configuración de ruta para tu carpeta 'src'
chdir = "src" 

# Configuración del servidor
bind = "0.0.0.0:8000"
workers = (multiprocessing.cpu_count() * 2) + 1
worker_class = "uvicorn.workers.UvicornWorker"

# Tiempos de espera (vital para IA/Azure OpenAI)
timeout = 230
accesslog = "-"
errorlog = "-"