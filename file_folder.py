import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

list_of_files = [
    "src/utils.py",
    "src/__init__.py",
    "src/medical_chatbot.py",
    "src/prompts.py",
    "experiments/experiments.ipynb",
    ".env",
    "setup.py",
    "store_index.py",
    "style",
    "templates/ui.html",
    "app.py",
]

for filepath in list_of_files:
   filepath = Path(filepath)
   filedir, filename = os.path.split(filepath)

   if filedir !="":
      os.makedirs(filedir, exist_ok=True)
      logging.info(f"Creating directory; {filedir} for the file {filename}")

   if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
      with open(filepath, 'w') as f:
         pass
         logging.info(f"Creating empty file: {filepath}")

   else:
      logging.info(f"{filename} is already created")
