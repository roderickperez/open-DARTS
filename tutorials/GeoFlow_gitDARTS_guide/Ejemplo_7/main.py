
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
redirect_darts_output('binary.log')    # change output styleS

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



# output initial solution to vtk file
output_dir_base = 'vtk_output'

prop_list = m.physics.vars + m.output.properties


# ######  --------------        Ejemplo_VTK  #1          -------------- 


# m.run(end_time*365)          # End time of the simulation (days)
# caso= '1'
# output_dir = output_dir_base + caso
# m.output.output_to_vtk(output_directory=output_dir, output_properties=prop_list)


######  --------------        Ejemplo_VTK  #2          -------------- 


caso= '2'
output_dir = output_dir_base + caso

# Correr simulaciones
for t in range(end_time):

    m.run(365)
    m.output.output_to_vtk(output_directory=output_dir, output_properties=prop_list)




################################################################


m.print_timers()
m.print_stat()


################################################################

print('\n----------------------------------------WRITE EXCEL FILE------------------------')
print('\n')


#   we create AN excel files:   "full_data" 

td = pd.DataFrame.from_dict(m.physics.engine.time_data)
td.to_pickle("darts_time_data.pkl")
writer = pd.ExcelWriter('full_data.xlsx')
td.to_excel(writer, 'Sheet1')
writer.close()

################################################################
