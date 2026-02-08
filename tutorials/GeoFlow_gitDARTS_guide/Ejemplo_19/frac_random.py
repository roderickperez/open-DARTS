import random
import math
import matplotlib.pyplot as plt
import os



import os
# Verify the current working directory
current_directory = os.getcwd()
print(f"Current working directory: {current_directory}")


# Obtener la carpeta donde se encuentra este script y Cambiar el directorio actual al del script
new_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(new_directory)
print(f"Directorio cambiado a: {os.getcwd()}")


def generar_puntos_aleatorios(Lmin, Lmax, Xmin,Xmax,Ymin,Ymax,n, lineas, Ang1, Ang2):
    
   
    for _ in range(n):
        # Generar el primer punto aleatorio con las restricciones dadas, como enteros
        x = random.randint(int(Xmin), int(Xmax))
        y = random.randint(int(Ymin), int(Ymax))
        punto_1 = (x, y)

        # Generar la distancia asegurando que esté dentro de Lmin y Lmax
        distancia = random.uniform(Lmin, Lmax)

        # Generar un ángulo aleatorio
        angle = random.uniform(Ang1, Ang2)
        angle = angle * (math.pi / 180)
        
        #angle = random.uniform(0, 2 * math.pi)

        
        CA=distancia * math.cos(angle)
        CO=distancia * math.sin(angle)

        x1=x + CA
        y1=y + CO
        x1 = int(x1)    # Convertir las coordenadas a enteros
        y1 = int(y1)    # Convertir las coordenadas a enteros

        x2=x + CA
        y2=y - CO
        x2 = int(x2)
        y2 = int(y2)

        x3=x - CA
        y3=y - CO
        x3 = int(x3)
        y3 = int(y3)

        x4=x - CA
        y4=y + CO
        x4 = int(x4)
        y4 = int(y4)

        # Definir las coordenadas de los puntos
        p_1 = {"x": x1, "y": y1  , "dist": distancia }  # Coordenada de p_1
        p_2 = {"x": x2, "y": y2  , "dist": distancia }  # Coordenada de p_2
        p_3 = {"x": x3, "y": y3  , "dist": distancia }  # Coordenada de p_3
        p_4 = {"x": x4, "y": y4  , "dist": distancia }  # Coordenada de p_4

        # Crear la variable puntos_2 que contiene los 4 puntos
        puntos_aux = [p_1, p_2, p_3, p_4]

        for punto in puntos_aux:           

            # Asegurarse que el segundo punto esté dentro de los límites deseados        
            if not (Xmin <  punto['x'] < Xmax and Ymin < punto['y']  < Ymax):
                
                distancia_aux=distancia

                while not (Xmin <  punto['x'] < Xmax and Ymin < punto['y']  < Ymax):
                # Calcular el segundo punto
                    distancia_aux=distancia_aux-1

                    CA=distancia_aux * math.cos(angle)
                    CO=distancia_aux * math.sin(angle)

                    punto['x']= x + CA
                    punto['y']= y + CO

                    x2 = x1 + distancia_aux * math.cos(angle)
                    y2 = y1 + distancia_aux * math.sin(angle)
                
                punto['dist'] = distancia_aux
            
        
        # Encontrar la distancia máxima
        max_dist = max(punto["dist"] for punto in puntos_aux)

        # Filtrar los puntos que tienen la distancia máxima
        puntos_max = [punto for punto in puntos_aux if punto["dist"] == max_dist]

        # Escoger un punto aleatorio entre los que tienen la distancia máxima
        punto_maximo = random.choice(puntos_max)

        
        # Asignar el punto con el mayor valor de 'dist' a la variable 'punto_2'
        punto_2 = (punto_maximo['x'], punto_maximo['y'])
    

        # Guardar la línea como una tupla de puntos
        lineas.append((punto_1, punto_2))

    return lineas

def guardar_lineas_en_archivo(lineas, nombre_archivo, Ycentro, Zona1 , Zona2, Zona3):
    with open(nombre_archivo, 'w') as archivo:
        for i, (punto_1, punto_2) in enumerate(lineas, start=1):
            # Escribir ambas coordenadas en la misma línea
            archivo.write(f"{punto_1[0]} {punto_1[1]} {punto_2[0]} {punto_2[1]}\n")

def graficar_lineas(lineas):
    for punto_1, punto_2 in lineas:
        # Graficar la línea entre punto_1 y punto_2
        plt.plot([punto_1[0], punto_2[0]], [punto_1[1], punto_2[1]], color='blue')

     # Graficar las líneas punteadas horizontales
    linea1=Ycentro - (Zona1)/2
    linea2=Ycentro + (Zona1)/2
    plt.axhline(y=linea1, color='green', linestyle='--', label='y=1200')
    plt.axhline(y=linea2, color='green', linestyle='--', label='y=1800')

    linea1=Ycentro - (Zona1)/2   -  Zona2
    linea2=Ycentro + (Zona1)/2   +  Zona3
    plt.axhline(y=linea1, color='red', linestyle='--', label='y=1200')
    plt.axhline(y=linea2, color='red', linestyle='--', label='y=1800')
    
    plt.xlim(0, 4000)
    plt.ylim(0, 4000)
    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    plt.title('Líneas Aleatorias Generadas')
    plt.grid()
    plt.show()

   

class Puntos:
    def __init__(self, id, distancia, x2, y2):
        self.id = id
        self.distancia = distancia
        self.x2 = x2
        self.y = y2

lineas = []
# se agregan dos fraacturas pequeñas, las cuales representan los limites del dominio
# frac_1=(0, 0)
# frac_2=(100, 0)
# lineas.append((frac_1, frac_2))
# frac_1=(3900, 3000)
# frac_2=(4000, 3000)
# lineas.append((frac_1, frac_2))


#--------------------   definir limites de zona-----------------------------
# 
Ycentro=2000
# espesor de cada zona
Zona1= 2000   # zona central
Zona2= 1   # zona inferior
Zona3= 1   # zona superior

Len_1=800  # (m)
ang_1= 15  # (°)
rango_angulo_1=10
num_1=30   # (# de frac)


Len_2=400  # (m)
ang_2= 75  # (°)
rango_angulo_2=10
num_2=30   # (# de frac)


# 
# #--------------------   1° seccion!-----------------------------    POSIV
# 
# # Definir la longitud mínima y máxima y el número de líneas
Lmin = Len_1-10   
Lmax = Len_1+10 
Xmin=0
Xmax=4000
Ymin=Ycentro - (Zona1) / 2
Ymax=Ycentro + (Zona1) / 2 
 # Para convertir de grados a radianes, se multiplica el número de grados por π/180
Ang1=ang_1 -rango_angulo_1
Ang2=ang_1 +rango_angulo_1
n = num_1     # Cambia este valor para generar más o menos líneas

# Generar las líneas
lineas = generar_puntos_aleatorios(Lmin, Lmax, Xmin,Xmax,Ymin,Ymax, n , lineas , Ang1, Ang2)


# #--------------------   1° seccion!-----------------------------   NEGATIV
# 
# # Definir la longitud mínima y máxima y el número de líneas
Lmin = Len_1-10   
Lmax = Len_1+10 
Xmin=0
Xmax=4000
Ymin=Ycentro - (Zona1) / 2
Ymax=Ycentro + (Zona1) / 2 
 # Para convertir de grados a radianes, se multiplica el número de grados por π/180
Ang1=-ang_1 +rango_angulo_1 
Ang2=-ang_1 - rango_angulo_1
n = num_1     # Cambia este valor para generar más o menos líneas

# Generar las líneas
lineas = generar_puntos_aleatorios(Lmin, Lmax, Xmin,Xmax,Ymin,Ymax, n , lineas , Ang1, Ang2)



#--------------------   2° seccion!-----------------------------    POSIV
# 
# # Definir la longitud mínima y máxima y el número de líneas
Lmin = Len_2-10   
Lmax = Len_2+10 
#Xmin=0
#Xmax=4000
#Ymin=Ycentro - (Zona1) / 2  -  Zona2
#Ymax=Ycentro - (Zona1) / 2  
Ang1=ang_2 -rango_angulo_2         #   -15
Ang2=ang_2 +rango_angulo_2         #   -15
n = num_2     # Cambia este valor para generar más o menos líneas

# Generar las líneas
lineas = generar_puntos_aleatorios(Lmin, Lmax, Xmin,Xmax,Ymin,Ymax, n , lineas, Ang1, Ang2)


#--------------------   2° seccion!-----------------------------   NEGATIV
# 
# # Definir la longitud mínima y máxima y el número de líneas
Lmin = Len_2-10   
Lmax = Len_2+10  
#Xmin=0
#Xmax=4000
#Ymin=Ycentro - (Zona1) / 2  -  Zona2
#Ymax=Ycentro - (Zona1) / 2  
Ang1=-ang_2 + rango_angulo_2        #   -15
Ang2=-ang_2 -rango_angulo_2        #   -15
n = num_2     # Cambia este valor para generar más o menos líneas

# Generar las líneas
lineas = generar_puntos_aleatorios(Lmin, Lmax, Xmin,Xmax,Ymin,Ymax, n , lineas, Ang1, Ang2)






#--------------------   3° seccion!-----------------------------
# 
# # Definir la longitud mínima y máxima y el número de líneas
Lmin = 50   
Lmax = 150 
Xmin=0
Xmax=4000
Ymin=Ycentro + (Zona1) / 2  
Ymax=Ycentro + (Zona1) / 2  +  Zona3
Ang1=-15
Ang2=15
n = 200     # Cambia este valor para generar más o menos líneas

# Generar las líneas
# lineas = generar_puntos_aleatorios(Lmin, Lmax, Xmin,Xmax,Ymin,Ymax, n , lineas, Ang1, Ang2)

# Guardar las líneas en un archivo
nombre_archivo = 'frac.txt'
guardar_lineas_en_archivo(lineas, nombre_archivo, Ycentro, Zona1 , Zona2, Zona3)

print(f"Las coordenadas de las fracturas se han guardado en '{nombre_archivo}'.")

# Graficar las líneas
graficar_lineas(lineas)

