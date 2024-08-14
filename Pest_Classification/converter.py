import os
import nbformat
from nbconvert import PythonExporter

# Define source and destination directories
source_dir = r'Notebooks/'
destination_dir = r'Scripts/'

# Ensure destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Loop through all files in the source directory
for filename in os.listdir(source_dir):
    if filename.endswith(".ipynb"):
        notebook_path = os.path.join(source_dir, filename)
        
        # Read the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook_content = nbformat.read(f, as_version=4)
        
        # Convert the notebook to Python script
        python_exporter = PythonExporter()
        python_script, _ = python_exporter.from_notebook_node(notebook_content)
        
        # Define the destination file path
        script_name = filename.replace('.ipynb', '.py')
        script_path = os.path.join(destination_dir, script_name)
        
        # Write the Python script to the destination directory
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(python_script)

        print(f"Converted {filename} to {script_name}")
