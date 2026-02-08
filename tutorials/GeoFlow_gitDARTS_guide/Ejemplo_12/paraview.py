import os
import numpy as np
import pandas as pd
from paraview.simple import *





import os
# Verify the current working directory
current_directory = os.getcwd()
print(f"Current working directory: {current_directory}")


# Obtener la carpeta donde se encuentra este script y Cambiar el directorio actual al del script
new_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(new_directory)
print(f"Directorio cambiado a: {os.getcwd()}")

import sys
if new_directory not in sys.path:
    sys.path.append(new_directory)


import Select_case
import importlib

importlib.reload(Select_case)
print('caso leido', Select_case.case)



data_dir = new_directory + "/vtk_output1"
filename_template = "solution_ts{}.vts"


if Select_case.case == '1':
        point1 = 666.666, 495, -50
        point2 = 1000, 495, -50
    
elif Select_case.case == '2':
        point1 = 333.333, 495, -50
        point2 = 1000, 495, -50


variable_name = "temp [°C]"

start_idx = 0
end_idx = 30


# Lista para almacenar los datos de presión de cada archivo
temp_data = []
# Para guardar las posiciones a lo largo de la línea (suponemos iguales para todos)
positions = None

for i in range(start_idx, end_idx + 1):
    filepath = os.path.join(data_dir, filename_template.format(i))
    if not os.path.isfile(filepath):
        print(f"Archivo no encontrado: {filepath}, saltando...")
        continue
    
    data = XMLStructuredGridReader(FileName=[filepath])
    
    plot_line = PlotOverLine(Input=data)
    plot_line.Point1 = point1
    plot_line.Point2 = point2
    plot_line.Resolution = 100  # Mayor resolución para mejor curva
    
    plot_line.UpdatePipeline()
    
    # Obtener la salida como un objeto de datos
    output = servermanager.Fetch(plot_line)
    
    # Extraer la variable temp del PointData
    array = output.GetPointData().GetArray(variable_name)
    
    if array is None:
        print(f"No se encontró la variable '{variable_name}' en archivo {filepath}")
        continue
    
    n_points = output.GetNumberOfPoints()
    
    # Extraer valores de temp en numpy array
    temp_values = np.array([array.GetValue(j) for j in range(n_points)])
    
    # Extraer posiciones (suponemos que la distancia es el índice o podemos calcular)
    if positions is None:
        # Extraemos la posición X para la primera coordenada como ejemplo de distancia a lo largo de la línea
        positions = np.array([output.GetPoint(j)[0] for j in range(n_points)])
    
    temp_data.append(temp_values)

# Crear DataFrame: la primera columna es posición, luego columnas por cada ts
df = pd.DataFrame()
df['position'] = positions




for idx, pdata in enumerate(temp_data):

    if Select_case.case == '1':
        print("Simulando caso 1")
        df[f'Año {idx}'] = pdata
        # Guardar CSV
        output_csv = os.path.join(new_directory, "Temp_sobre_linea_1.csv")
    
    elif Select_case.case == '2':
        print("Simulando caso 2")
        df[f'Año {idx}'] = pdata
        # Guardar CSV
        output_csv = os.path.join(new_directory, "Temp_sobre_linea_2.csv")
    
    


df.to_csv(output_csv, index=False)
print(f"Datos guardados en: {output_csv}")


############################################################

