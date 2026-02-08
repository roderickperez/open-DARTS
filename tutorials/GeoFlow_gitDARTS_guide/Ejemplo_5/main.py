
# model: DartsModel
# Physics: Geothermal
# reservoir: StructReservoir


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from model import Model  # Se carga la clase “Model” del archivo model.py
                         # (debe de estar en la misma carpeta)

from darts.engines import  redirect_darts_output  # ,value_vector
#redirect_darts_output('dfm_model.log') # change output style
#redirect_darts_output('binary.log')    # change output styleS

import os

# Verify the current working directory
current_directory = os.getcwd()
print(f"Current working directory: {current_directory}")


# Obtener la carpeta donde se encuentra este script y Cambiar el directorio actual al del script
new_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(new_directory)
print(f"Directorio cambiado a: {os.getcwd()}")



print('\n----------------------------------------START MODEL CREATION------------------------')
print('\n')
        
print('Reservoir model: Dartsmodel')

m = Model()



print('\n----------------------------------------END WITH MODEL CREATION------------------------')
print('\n')

################################################################

print('\n----------------COMPONENTS, PHASES AND VARIABLES :')
print('\n')
print('Phases=',m.physics.phases)          
print('Number of phases=',m.physics.nph)      
print('Number of components=',m.physics.nc)     
print('Main variables=',m.physics.vars)
print('Number of variables=',m.physics.n_vars)  
print('Number of operators=',m.physics.n_ops) 

print('\n')

################################################################

print('\n------------------------MODEL INITIALIZATION------------------------')
print('\n')


m.init()

m.set_output(output_folder='output')


m.plot_init_phase_with_p_h( m.input_distribution['pressure'],   # Inicial
            m.input_distribution['enthalpy'],                   # Inicial
            m.P_iny,                                            # Inyeccion  (aproximado)
            m.inj_temp + 273.15,                                # Inyeccion
            m.P_prod,                                           # Produccion 
            m.input_distribution['enthalpy'],                   # Produccion (aproximado)
                         )



# output_dir_base = 'vtk_output'
# caso= '0'
# output_dir = output_dir_base + caso

# # output initial conditions
# prop_list = m.physics.vars + m.output.properties
# m.output.output_to_vtk(output_directory=output_dir, output_properties=prop_list)


print('\n')            

################################################################


# end_time = 30                   # End time of the simulation (years)

# print('\n------------------------TIME OF SIMULATION------------------------')
# print('\n')
# print('Simulation time (days)=',end_time*365)
# print('Simulation time (years)=',end_time)
# print('\n')

################################################################

print('\n------------------------RUN SIMULATION ------------------------')
print('\n')


x= np.linspace( m.Long/(2*m.nx), m.Long, num=m.nx)

# Crear figuras antes del bucle
fig1, ax1 = plt.subplots()  # Figura para presión
ax1.set_title("Pressure")
ax1.set_xlabel("x [m]")
ax1.set_ylabel("Pressure [bar]")

fig2, ax2 = plt.subplots()  # Figura para entalpía
ax2.set_title("Enthalpy")
ax2.set_xlabel("x [m]")
ax2.set_ylabel("Enthalpy [KJ/Kg]")

# Correr simulaciones
# for t in range(24):   #  Por 24 horas
#     m.run(0.0416667)  #  1 hora
#     resultados = np.array(m.physics.engine.X, copy=False)
#     # Extraer los valores de presión (índices impares)
#     pressure = resultados[:m.nb * 2:2]
#     # Extraer los valores de entalpía (índices pares)
#     enthalpy = resultados[1:m.nb * 2:2]
#     # Agregar curva de presión
#     ax1.plot(x, pressure, label=f'Hora {t+1}', alpha=0.7)
#     # Agregar curva de entalpía
#     ax2.plot(x, enthalpy / 18.015, label=f'Hora {t+1}', alpha=0.7)



# # Correr simulaciones
# for t in range(10):   #  Por 10 dias
#     m.run(1)          #  1 dia
#     resultados = np.array(m.physics.engine.X, copy=False)
#     # Extraer los valores de presión (índices impares)
#     pressure = resultados[:m.nb * 2:2]
#     # Extraer los valores de entalpía (índices pares)
#     enthalpy = resultados[1:m.nb * 2:2]
#     # Agregar curva de presión
#     ax1.plot(x, pressure, label=f'Dia {t+1}', alpha=0.7)
#     # Agregar curva de entalpía
#     ax2.plot(x, enthalpy / 18.015, label=f'Dia {t+1}', alpha=0.7)


# # Correr simulaciones
# for t in range(12):   #  Por 12 meses
#     m.run(30)         #  1 mes
#     resultados = np.array(m.physics.engine.X, copy=False)
#     # Extraer los valores de presión (índices impares)
#     pressure = resultados[:m.nb * 2:2]
#     # Extraer los valores de entalpía (índices pares)
#     enthalpy = resultados[1:m.nb * 2:2]
#     # Agregar curva de presión
#     ax1.plot(x, pressure, label=f'Mes {t+1}', alpha=0.7)
#     # Agregar curva de entalpía
#     ax2.plot(x, enthalpy / 18.015, label=f'Mes {t+1}', alpha=0.7)


# # Correr simulaciones
# for t in range(10):   #  Por 10 años
#     m.run(365)        #  1 año
#     resultados = np.array(m.physics.engine.X, copy=False)
#     # Extraer los valores de presión (índices impares)
#     pressure = resultados[:m.nb * 2:2]
#     # Extraer los valores de entalpía (índices pares)
#     enthalpy = resultados[1:m.nb * 2:2]
#     # Agregar curva de presión
#     ax1.plot(x, pressure, label=f'Año {t+1}', alpha=0.7)
#     # Agregar curva de entalpía
#     ax2.plot(x, enthalpy / 18.015, label=f'Año {t+1}', alpha=0.7)


# # Correr simulaciones
for t in range(10):   #  Por 100 años
    m.run(365*10)      #  1 Decada
    resultados = np.array(m.physics.engine.X, copy=False)
    # Extraer los valores de presión (índices impares)
    pressure = resultados[:m.nb * 2:2]
    # Extraer los valores de entalpía (índices pares)
    enthalpy = resultados[1:m.nb * 2:2]
    # Agregar curva de presión
    ax1.plot(x, pressure, label=f'Decada {t+1}', alpha=0.7)
    # Agregar curva de entalpía
    ax2.plot(x, enthalpy / 18.015, label=f'Decada {t+1}', alpha=0.7)


################################################################

m.print_timers()
m.print_stat()



################################## 


# Mostrar las gráficas después del bucle
ax1.legend()
fig1.savefig('pressure.png', format='png')
plt.show()

ax2.legend()
fig2.savefig('enthalpy.png', format='png')
plt.show()


################################## 

tiempo_final, estado_final = m.output.output_properties(output_properties = m.physics.property_operators[0].props_name)
    # estado_final no es una lista o un arreglo. Es un diccionario y no puede ser indexado por números como 1.
    # Contiene la informacion de las variables secundarias para cada celda
    # tiempo_final es un arreglo. Contiene solamente el tiempo de simulacion en el cual se escribieron los valores

print('\n----------------------------------------------------------------')
print(type(estado_final))           #  <class 'dict'>
print(len(estado_final))            # Número de claves en el diccionario
print(estado_final.keys())          # dict_keys(['temp', 'satA', 'satV'])
print(estado_final['temp [°C]'].shape)   #  (11, 500)   ------ (time, cells)
print(len(estado_final['temp [°C]']))    # Cantidad de elementos en la primera dimensión (tiempo)
print(len(estado_final['temp [°C]'][0])) # Cantidad de elementos en la segunda dimensión (celdas)
print(type(tiempo_final))           #  <class 'numpy.ndarray'>
print(len(tiempo_final))            # 11
print((tiempo_final))               # Tiempo en el cual se escribieron los valores

# Crear el gráfico para cada tiempo
plt.figure(figsize=(10, 6))
for t in range(len(tiempo_final)-1):  # Iterar sobre todos los tiempos 
                                      # (menos el 1°, que corresponde al estado inicial)
    plt.plot(x, estado_final['temp [°C]'][t+1], label=f'Tiempo {t+1}')

# Añadir título, etiquetas y leyenda
plt.title("Temperature")
plt.xlabel("x [m]")
plt.ylabel("Temperature [°C]")
plt.legend(loc='best')
plt.grid(True)
plt.savefig('Temperature.png', format='png')  # Guarda la figura en formato PNG


# Mostrar el gráfico
plt.show()



# Crear el gráfico para cada tiempo
plt.figure(figsize=(10, 6))
for t in range(len(tiempo_final)-1):  # Iterar sobre todos los tiempos 
                                      # (menos el 1°, que corresponde al estado inicial)
    plt.plot(x, estado_final['sat_W'][t+1], label=f'Tiempo {t+1}')

# Añadir título, etiquetas y leyenda
plt.title("Water Saturation")
plt.xlabel("x [m]")
plt.ylabel("Sw")
plt.legend(loc='best')
plt.grid(True)
plt.savefig('Sw.png', format='png')  # Guarda la figura en formato PNG

# Mostrar el gráfico
plt.show()



##################################



m.plot_final_phase_with_p_h( m.input_distribution['pressure'],  # Inicial
            m.input_distribution['enthalpy'],                   # Inicial
            pressure[0],                                         # Inyeccion  (real)
            enthalpy[0],                                        # Inyeccion (real)
            pressure[-1],                                        # Produccion (real)
            enthalpy[-1] ,                                      # Produccion (real)
                         )



################################################################

print('\n----------------------------------------WRITE EXCEL FILE------------------------')
print('\n')


# Escribimos los datos de presion y entalpia en una archivo excel


df = pd.DataFrame(pressure)
df2= pd.DataFrame(enthalpy/ 18.015)
writer = pd.ExcelWriter('Perfiles.xlsx')
df.to_excel(writer, sheet_name = 'Pressure')
df2.to_excel(writer, sheet_name = 'Entalpia')
writer.close()


################################################################
# 
# Escribimos los datos del pozo (resumidos)

td = pd.DataFrame.from_dict(m.physics.engine.time_data)
td.to_pickle("darts_time_data.pkl")
writer = pd.ExcelWriter('well_resume.xlsx')
td.to_excel(writer, 'Sheet1')
writer.close()

################################################################


# Escribimos los datos completos del pozo y los ploteamos

m.output.store_well_time_data( save_output_files=True)
m.output.plot_well_time_data()