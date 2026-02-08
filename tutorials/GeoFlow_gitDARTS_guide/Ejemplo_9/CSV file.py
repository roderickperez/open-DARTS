import pandas as pd
import matplotlib.pyplot as plt
import os


# Verify the current working directory
current_directory = os.getcwd()
print(f"Current working directory: {current_directory}")


# Obtener la carpeta donde se encuentra este script y Cambiar el directorio actual al del script
new_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(new_directory)
print(f"Directorio cambiado a: {os.getcwd()}")

#data_dir = os.path.join(new_directory, "vtk_output1")


#output_dir = data_dir

import Select_case
# Carpeta y nombre de archivo para guardar la imagen
if Select_case.case == '1':
    # Ruta del CSV generado
    csv_path = os.path.join(new_directory, "presion_sobre_linea_1.csv")
    output_img = os.path.join(new_directory, "grafico_presion_1.png")
    
elif Select_case.case == '2':
    # Ruta del CSV generado
    csv_path = os.path.join(new_directory, "presion_sobre_linea_2.csv")
    output_img = os.path.join(new_directory, "grafico_presion_2.png")





# Leer CSV
df = pd.read_csv(csv_path)

plt.figure(figsize=(12, 7))

pos = df['position']

# Plotear cada columna de presión
for col in df.columns:
    if col == 'position':
        continue
    plt.plot(pos, df[col], label=col)

plt.xlabel("Posición a lo largo de la línea (X)")
plt.ylabel("Presión")
plt.title("Distribución de presión a lo largo de la línea para todos los archivos")
plt.legend(loc='best', fontsize='small')
plt.grid(True)

# Guardar la figura antes de mostrarla
plt.savefig(output_img, dpi=300, bbox_inches='tight')

plt.show()

print(f"Imagen guardada en: {output_img}")
