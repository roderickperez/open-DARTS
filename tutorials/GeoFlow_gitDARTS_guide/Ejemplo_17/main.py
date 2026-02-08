
# model: DartsModel
# Physics: Geothermal
# reservoir: StructReservoir


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from darts.engines import value_vector, redirect_darts_output
from model import Model  # Se carga la clase “Model” del archivo model.py
                         # (debe de estar en la misma carpeta)


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


###################################################

from Inputs import input_data_default

input_data = input_data_default()
input_data['frac_file'] = os.path.join(new_directory, input_data['frac_file'])

print(input_data['frac_file'])   # This should return True if the file exists

print(os.path.exists(input_data['frac_file']))   # This should return True if the file exists
#input_data['frac_file'] = 'C:/Users/luis_/sorce_PAPPIT_VSC/ejemplos/frac.txt'



#########################################################################

# plotea red de fracturas



def plot_dfn(input_data):
    frac_data_raw = np.genfromtxt(input_data['frac_file'])  
    

    plt.gca().set_aspect('equal')
    for i in range(frac_data_raw.shape[0]):
        plt.plot(np.append(frac_data_raw[i, 0], frac_data_raw[i, 2]),
                 np.append(frac_data_raw[i, 1], frac_data_raw[i, 3]))
    
    wells_inj = input_data['inj_well_coords']
    plt.plot(wells_inj[0][0], wells_inj[0][1], 'o', color='b', label='inj well')
    wells_prod = input_data['prod_well_coords']
    plt.plot(wells_prod[0][0], wells_prod[0][1], 'o', color='r', label='prod well')
    #plt.plot(wells_prod[1][0], wells_prod[1][1], 'o', color='r', label='prod well #2')


    plt.xlim(input_data['x1'],input_data['x2'])
    plt.ylim(input_data['y1'],input_data['y2'])
    plt.xlabel('X, m.')
    plt.ylabel('Y, m.')
    plt.legend()
    plt.grid()
    plt.show()


plot_dfn(input_data) 


#########################################################################

from darts.tools.fracture_network.preprocessing_code import frac_preprocessing

# Read fracture tips from input_data['frac_file'] and generate a .geo text file (input for gmsh), then
# call gmesh to create a mesh and output it to .msh text file, which will be used as an input to DARTS
# These files are stored tin the 'meshes' folder, one mesh is original (raw) 
# and the second is optimized for calculation (cleaned)

def generate_mesh(input_data):
    output_dir = 'meshes'
    mesh_dir= os.path.join(new_directory, output_dir)

    import shutil
    
    if os.path.exists(mesh_dir) and os.path.isdir(mesh_dir):
        shutil.rmtree(mesh_dir)
        print(f"La carpeta '{mesh_dir}' ha sido eliminada.")
    else:
        print(f"La carpeta '{mesh_dir}' no existe o no es una carpeta válida.")



    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

   

    frac_preprocessing(output_dir=output_dir, frac_data_raw=np.genfromtxt(input_data['frac_file']),
                       char_len=input_data['char_len'], # partition_fractures_in_segms=False,
                       input_data=input_data,
                       apertures_raw=np.array([.01, .01]),
                       box_data=np.array(input_data['box_data']),
                       mesh_raw=True,
                       mesh_clean=True,
                      )
    #  ,
    #                    mesh_clean=True

############          Generacion de la malla            ###############

from datetime import datetime

t1 = datetime.now()
generate_mesh(input_data)
t2 = datetime.now()
mesh_gen_timer = (t2 - t1).total_seconds()

print('Mesh generation time:', mesh_gen_timer, 'sec.')


###################################################

print('\n----------------------------------------START MODEL CREATION------------------------')
print('\n')
        
print('Reservoir model: Dartsmodel')

m = Model(input_data)

print('Basic model: Dartsmodel')

print('\n----------------------------------------END WITH MODEL CREATION------------------------')
print('\n')

################################################################

print('\n----------------COMPONENTS, PHASES AND VARIABLES :')
print('\n')
print('Phases=',m.physics.phases)          # REVISAR, DEBE DE HABER ALGUN ERROR
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

# output initial solution to vtk file
output_dir_base = 'vtk_output'


# se utiliza:         m.output_to_vtk()
# la cual llama a:    timesteps, property_array = m.output_properties(output_properties=output_prop)
# y posteriormente:   m.reservoir.output_to_vtk()

# m.reservoir.init_vtk(output_directory=output_dir)


# ######  --------------        Ejemplo_VTK  #1          -------------- 


caso= '1'
output_dir = output_dir_base + caso
prop_list = m.physics.vars + m.output.properties

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


