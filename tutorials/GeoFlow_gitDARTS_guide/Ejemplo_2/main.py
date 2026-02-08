
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

m.plot_init_phase_with_p_t( m.input_distribution['pressure'],
            m.input_distribution['temperature'],
            m.P_iny,
            m.inj_temp + 273.15,
            m.P_prod,
            m.input_distribution['temperature'],
                         )




# output_dir_base = 'vtk_output'
# caso= '0'
# output_dir = output_dir_base + caso

# # output initial conditions
# prop_list = m.physics.vars + m.output.properties
# m.output.output_to_vtk(output_directory=output_dir, output_properties=prop_list)


print('\n')            

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




# Paso_de_tiempo= 1
# tiempo_inicial=0
# tiempo_total=0
# # # Correr simulaciones
# for t in range(7):   #  
#     m.run(Paso_de_tiempo)        #  
#     tiempo_total=tiempo_total + Paso_de_tiempo
#     resultados = np.array(m.physics.engine.X, copy=False)
#     # Extraer los valores de presión (índices impares)
#     pressure = resultados[:m.nb * 2:2]
#     # Extraer los valores de entalpía (índices pares)
#     enthalpy = resultados[1:m.nb * 2:2]
#     # Agregar curva de presión
#     ax1.plot(x, pressure, label=f'Dia {tiempo_total}', alpha=0.7)
#     # Agregar curva de entalpía
#     ax2.plot(x, enthalpy / 18.015, label=f'Dia {tiempo_total}', alpha=0.7)
#     Paso_de_tiempo=Paso_de_tiempo*4


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