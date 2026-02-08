
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

# m.plot_init_phase_with_p_t( m.input_distribution['pressure'],
#             m.input_distribution['temperature'],
#             m.P_iny,
#             m.inj_temp + 273.15,
#             m.P_prod,
#             m.input_distribution['temperature'],
#                          )




# output_dir_base = 'vtk_output'
# caso= '0'
# output_dir = output_dir_base + caso

# # output initial conditions
# prop_list = m.physics.vars + m.output.properties
# m.output.output_to_vtk(output_directory=output_dir, output_properties=prop_list)


print('\n')            

################################################################

end_time = 30                   # End time of the simulation (years)

print('\n------------------------TIME OF SIMULATION------------------------')
print('\n')
print('Simulation time (days)=',end_time*365)
print('Simulation time (years)=',end_time)
print('\n')

################################################################

print('\n------------------------RUN SIMULATION ------------------------')
print('\n')


m.run(end_time*365)          # End time of the simulation (days)

################################################################

m.print_timers()
m.print_stat()


################################## 

# con la siguiente linea se obtiene los resultados, pero solamente los 
# que corresponden a las variables principales (presion y entalpia)
resultados= np.array(m.physics.engine.X, copy=False)      #  Celdas + pozos

print(resultados[-8:])  # Muestra los últimos 8 valores del array 

# Extraer los valores de presión (índices impares)
pressure = resultados[:m.nb*2:2]  # Del índice 0 al 999, tomando cada 2 valores
#pressure = resultados[:1000:2]  # Del índice 0 al 999, tomando cada 2 valores

# Extraer los valores de enthalpy (índices pares)
enthalpy = resultados[1:m.nb*2:2]  # Del índice 1 al 999, tomando cada 2 valores
#enthalpy = resultados[:1000:2]  # Del índice 0 al 999, tomando cada 2 valores

x= np.linspace( m.Long/(2*m.nx), m.Long, num=m.nx)


plt.title("Pressure")
plt.plot(x, pressure, color="red")
plt.xlabel("x [m]")       
plt.ylabel("Pressure [bar]")     
plt.savefig('Pressure.png', format='png')
plt.show()

plt.title("Enthalpy")
plt.xlabel("x [m]")       
plt.ylabel("Enthalpy [KJ/Kg]")   
plt.plot(x, enthalpy / 18.015, color="red")
plt.savefig('enthalpy.png', format='png')
plt.show()

tiempo_final, estado_final = m.output.output_properties(output_properties = m.physics.property_operators[0].props_name)
# estado_final no es una lista o un arreglo. Es un diccionario y no puede ser indexado por números como 1.
# Contiene la informacion de las variables secundarias
print(estado_final.keys())
#   dict_keys(['Enthalpy_W [KJ/kmol]', 'Enthalpy_S [KJ/kmol]', 'Enthalpy_W [KJ/kg]', 'Enthalpy_S [KJ/kg]', 
#              'temp [K]', 'temp [°C]', 'sat_W', 'sat_S'])

# tiempo_final solo contiene el tiempo inicial y final

plt.title("Temperature")
plt.xlabel("x [m]")       
plt.ylabel("Temperature [°C]")  
plt.plot(x, estado_final['temp [°C]'][1], color="red")
plt.savefig('Temperature.png', format='png')
plt.show()


plt.title("Water Saturation")
plt.xlabel("x [m]")       
plt.ylabel("Sw")  
plt.plot(x, estado_final['sat_W'][1], color="red")
plt.savefig('Water Saturation.png', format='png')
plt.show()


plt.title("Water Viscosity")
plt.xlabel("x [m]")       
plt.ylabel("Water Viscosity [cP]")  
plt.plot(x, estado_final['Visco'][0], color="red")
plt.savefig('Water Viscosity.png', format='png')
plt.show()
##################################



################################################################

print('\n----------------------------------------WRITE EXCEL FILE------------------------')
print('\n')

# Escribimos los datos de presion y entalpia en una archivo excel

df = pd.DataFrame(pressure)
df2= pd.DataFrame(enthalpy/ 18.015)
writer = pd.ExcelWriter('Perfiles.xlsx')
df.to_excel(writer, sheet_name = 'Pressure')
df2.to_excel(writer, sheet_name = 'Enthalpy')
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