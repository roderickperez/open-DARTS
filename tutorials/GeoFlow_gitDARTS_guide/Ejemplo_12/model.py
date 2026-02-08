import numpy as np
from darts.models.darts_model import DartsModel
from darts.reservoirs.struct_reservoir import StructReservoir
from darts.physics.geothermal.physics import Geothermal
from darts.physics.geothermal.property_container import PropertyContainer
from darts.engines import value_vector, sim_params, well_control_iface

import Select_case

#from re import L
#from darts.models.cicd_model import CICDModel
#from darts.physics.properties.iapws.iapws_property_vec import _Backward1_T_Ph_vec
#from darts.tools.keyword_file_tools import load_single_keyword
#from darts.physics.properties.iapws.iapws_property import *
#from darts.physics.properties.basic import ConstFunc




class Model(DartsModel):

    def __init__(self, n_points=64):


        # call base class constructor
        super().__init__()

        self.timer.node["initialization"].start()

        self.set_reservoir()         # Se llama la funcion "set_reservoir" (definida mas abajo)
                                     # con la cual se establecen las caracteristicas el yacimiento
        self.set_physics(n_points)   # Se llama la funcion "set_physics" (definida mas abajo)
                                     # con la cual se establecela fisica del problema

                                  # 1 [s]= 1.1574e-5 [days]
        self.set_sim_params(first_ts=1.1574e-5, mult_ts=2, max_ts=100, runtime=3650, tol_newton=1e-2, tol_linear=1e-4,
                            it_newton=20, it_linear=40, newton_type=sim_params.newton_global_chop,
                            newton_params=value_vector([1]))


        self.init_pressure= 150.      # [bar]
        self.init_temp= 300.         # [c]


        self.timer.node["initialization"].stop()


    def set_physics(self, n_points):
        # create pre-defined physics for geothermal

        print('Reservoir physics: Geothermal')


        self.min_p=0.                #  bar
        self.max_p=500.              #  bar
        self.min_e= 50.*18.01528      # KJ/kg ---- KJ/Kmol
        self.max_e= 3200.*18.01528   # KJ/kg ---- KJ/Kmol

 
        self.physics = Geothermal(self.timer, n_points=n_points,   
                                  min_p=self.min_p, max_p=self.max_p, min_e=self.min_e, max_e=self.max_e, cache=False)
                                                            #    min_p=150, max_p=260, min_e=9000, max_e=60000
                                                            #    min_p=150, max_p=260, min_e=0, max_e=100000
                                                            #    min_p=1, max_p=351, min_e=0, max_e=100000
        print('\n')



        property_container = PropertyContainer()


        property_container.output_props = {'Enthalpy_W [KJ/kmol]': lambda: property_container.enthalpy[0],
                                           'Enthalpy_S [KJ/kmol]': lambda: property_container.enthalpy[1],
                                           'Enthalpy_W [KJ/kg]': lambda: property_container.enthalpy[0]/18.01528,
                                           'Enthalpy_S [KJ/kg]': lambda: property_container.enthalpy[1]/18.01528,
                                           'temp [K]': lambda: property_container.temperature,
                                           'temp [°C]': lambda: property_container.temperature-273.15,
                                            'sat_W': lambda: property_container.saturation[0],
                                            'sat_S': lambda: property_container.saturation[1]
                                            }

        self.physics.add_property_region(property_container)


    def set_initial_conditions(self):

        
        # initialization with constant pressure and temperature
        input_distribution = {"pressure": self.init_pressure,              # [bar]
                              "temperature": self.init_temp + 273.15  # [c]-----[k] 
                             }
        self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh, input_distribution=input_distribution)
   

        self.state_init = value_vector([input_distribution['pressure']])               # Initial presion   (bar)   
        enth_init = self.physics.property_containers[0].enthalpy_ev['total'].evaluate(self.state_init,
                                                                                    input_distribution['temperature'] )
        
        estado=np.array([self.state_init[0],enth_init ])
        Sw_init=self.physics.property_containers[0].saturation_ev['water'].evaluate(estado)


        print('\n----------------INITIAL CONDITIONS :')
        print('\n')
        print('Initial pressure (bar)=',input_distribution['pressure'])
        print('Initial temperature (C)=',input_distribution['temperature']-273.15)
        print('Initial temperature (k)=',input_distribution['temperature'])
        print('Initial enthalpy ([kJ/kg])=',enth_init/18.01528)
        print('Initial enthalpy ([kJ/kmol])=',enth_init)
        print('Initial Saturation (water)=',Sw_init)        
        print('\n')




    def set_reservoir(self):


        print('\n----------------DEFINE RESERVOIR :')
        print('\n')



        ################################################################

        self.nx, self.ny, self.nz = 30, 30, 2
        self.nb = self.nx * self.ny * self.nz  # número total de bloques

        Long_x=1000.   # [m]
        Long_y=1000.   # [m]
        Lony_z= 100.   # [m]

        self.dx=Long_x/self.nx
        self.dy=Long_y/self.ny
        self.dz=Lony_z/self.nz                # [m] Espesor de capa


       #########################    Misma Permeabilidad  en cada capa     ##########################


        # self.reservoir = StructReservoir(self.timer, nx=self.nx, ny=self.ny, nz=self.nz, dx=self.dx, dy=self.dy, dz=self.dz,
        #                                  permx=10., permy=10., permz=10., poro=.01, 
        #                                  hcap=2470., rcond= 172.8 )

       #########################    Permeabilidad diferente en cada capa     ##########################

        # # Crear un arreglo vacío
        # perm = np.zeros(self.nb)

        # # Asignar los valores según las capas
        # perm[0:self.nx*self.ny] = 10                      #   5 [mD]
        # perm[self.nx*self.ny:2*self.nx*self.ny] = 20      #  10 [mD]
        # perm[2*self.nx*self.ny:3*self.nx*self.ny] = 40    #  15 [mD]

        # self.reservoir = StructReservoir(self.timer, nx=self.nx, ny=self.ny, nz=self.nz, dx=self.dx, dy=self.dy, dz=self.dz,
        #                                  permx=perm, permy=perm, permz=perm, poro=.01, 
        #                                  hcap=2470., rcond= 172.8 )

        
        #######################   Utilizando Start_z  para calcular profundidades y condiciones iniciales   #############

        # Crear un arreglo vacío
        perm_x = np.zeros(self.nb)
        perm_y = np.zeros(self.nb)
        perm_z = np.zeros(self.nb)

        # Asignar los valores según las capas

        perm_x[0:self.nx*self.ny] = 40                     #   [mD]
        perm_x[self.nx*self.ny:2*self.nx*self.ny] = 0      #   [mD]
        
        perm_y[0:self.nx*self.ny] = 40                     #   [mD]
        perm_y[self.nx*self.ny:2*self.nx*self.ny] = 0      #   [mD]

        perm_z[0:self.nx*self.ny] = 40                     #   [mD]
        perm_z[self.nx*self.ny:2*self.nx*self.ny] = 0      #   [mD]
        

        self.reservoir = StructReservoir(self.timer, nx=self.nx, ny=self.ny, nz=self.nz, dx=self.dx, dy=self.dy, dz=self.dz,
                                         permx=perm_x, permy=perm_y, permz=perm_z, poro=.01,
                                         hcap=2470., rcond= 800. )                                        
        #                                start_z: top reservoir depth (a single float value or nx*ny values)


        self.reservoir.boundary_volumes['xy_plus'] = 1e8



        print('Reservoir type= Structured')
        print('Reservoir dimension=',(self.nx, self.ny, self.nz))
        print('Number of blocks=',self.nb)



    def set_wells(self):


        # add well
        self.reservoir.add_well("INJ", wellbore_diameter=0.3048)

        if Select_case.case == '1':
        
            self.reservoir.add_perforation("INJ", cell_index=(20, 15, 1),  skin=0.0,
                                            well_radius=0.1524, multi_segment=True,well_indexD=0)
            
        elif Select_case.case == '2':
            self.reservoir.add_perforation("INJ", cell_index=(10, 15, 1),  skin=0.0,
                                           well_radius=0.1524, multi_segment=True,well_indexD=0)

        # add well
        self.reservoir.add_well("PRD", wellbore_diameter=0.3048)
        self.reservoir.add_perforation("PRD", cell_index=(30, 15, 1),  skin=0.0,
                                           well_radius=0.1524, multi_segment=True,well_indexD=0)


                                          # well_indexD - thermal well index (for heat loss through the wellbore)
                                              # if -1, use computed value based on cell geometry;
                                              # if 0 - no heat losses


    def set_well_controls(self):


        print('\n----------------WELL PARAMETERES :')
        mass_rate=4.0                         #     [Kg/s]
        mol_rate=mass_rate*4795.928784        #     [Kg/s]  --------> [kmol/day]
        inj_temp= 100                         #     [c]-----[k]

        print('\n---------Injector :')
        print('INJ mass rate [Kg/s]=',mass_rate)
        print('INJ mass rate [Kg/day]=',mass_rate*86400)
        print('INJ mass rate [kmol/day]=',mol_rate)
        print('INJ  temp. [c]=',inj_temp)
        print('INJ  temp. [k]=',inj_temp + 273.15)



        # initialization with constant pressure and temperature
        Inyection_state = {"pressure": self.init_pressure,              # [bar]
                              "temperature": inj_temp + 273.15             # [c]-----[k]
                             }

        self.state_init = value_vector([Inyection_state['pressure']])
        enth_init = self.physics.property_containers[0].enthalpy_ev['total'].evaluate(self.state_init,
                                                                                    Inyection_state['temperature'] )


        print('INJ enthalpy [KJ/Kg]=',enth_init/18.01528)
        print('INJ enthalpy [KJ/kmol]=',enth_init)

        P_prod=100                        #     [bar]
        #P_prod=50                        #     [bar]

        print('\n---------Producer :')
        print('PRD pressure [bar]=',P_prod)
        print('\n---------------------------')


        for i, w in enumerate(self.reservoir.wells):
            if 'INJ' in w.name:
                self.physics.set_well_controls(wctrl=w.control,
                                               control_type=well_control_iface.MOLAR_RATE,
                                               is_inj=True,
                                               target=mol_rate,
                                               phase_name='water',
                                               inj_temp=inj_temp + 273.15   #  [c]-----[k]
                                               )
            else:
                self.physics.set_well_controls(wctrl=w.control,
                                               control_type=well_control_iface.BHP,
                                               is_inj=False,
                                               target=P_prod
                                               )



