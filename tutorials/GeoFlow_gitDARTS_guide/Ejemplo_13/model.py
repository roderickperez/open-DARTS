import numpy as np
from darts.models.darts_model import DartsModel
from darts.reservoirs.struct_reservoir import StructReservoir
from darts.physics.geothermal.physics import Geothermal
from darts.physics.geothermal.property_container import PropertyContainer
from darts.engines import value_vector, sim_params, well_control_iface


#from re import L
#from darts.models.cicd_model import CICDModel
#from darts.physics.properties.iapws.iapws_property_vec import _Backward1_T_Ph_vec
#from darts.tools.keyword_file_tools import load_single_keyword
#from darts.physics.properties.iapws.iapws_property import *
#from darts.physics.properties.basic import ConstFunc

from darts.tools.keyword_file_tools import load_single_keyword
import os
from darts.reservoirs.unstruct_reservoir import UnstructReservoir
import meshio   # There are various mesh formats available for representing unstructured meshes. 
                # meshio can read and write all of the following and smoothly converts between them.



class Model(DartsModel):
    
    def __init__(self,  n_points=64):
        
        
        # call base class constructor
        super().__init__()

        self.timer.node["initialization"].start()

      
        self.set_reservoir()


        self.set_physics(n_points)   # Se llama la funcion "set_physics" (definida mas abajo) 
                                     # con la cual se establecela fisica del problema

                                  # 1 [s]= 1.1574e-5 [days] 
        self.set_sim_params(first_ts=1.1574e-5, mult_ts=2, max_ts=100, runtime=3650, tol_newton=1e-2, tol_linear=1e-4,
                            it_newton=20, it_linear=40, newton_type=sim_params.newton_global_chop,
                            newton_params=value_vector([1]))

        
        self.init_pressure= 86.      # [bar]
        self.init_temp= 300.         # [c]
        
        self.timer.node["initialization"].stop()

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
        
        

        fname = 'malla_1.msh'
        mesh_file = os.path.join('meshes', fname)
        print(mesh_file)

        permx = 5.0 # [mD]
        permy = 5.0  # [mD]
        permz = 5.0  # [mD]
        poro = 0.01 #    = 1%
        hcap = 2470.0  # [kJ/m3/K]
        rcond = 172.8   # [kJ/m/day/K]

        # initialize reservoir
        self.reservoir = UnstructReservoir(timer=self.timer, mesh_file=mesh_file,
                                      permx=permx, permy=permy, permz=permz,
                                      poro=poro,
                                      rcond=rcond,
                                      hcap=hcap)
                                        
    
        self.reservoir.physical_tags['matrix'] = [9991 ]
        self.reservoir.physical_tags['boundary'] = [2, 1, 3, 4, 5, 6]  # order: Z- (bottom); Z+ (top) ; Y-; X+; Y+; X-

        print('Reservoir type= Unstructured')

        #print('Reservoir dimension=',(self.nx, self.ny, self.nz))
        #print('Number of blocks=',self.nb)
        
    def set_physics(self, n_points):
        # create pre-defined physics for geothermal
 
        print('Reservoir physics: Geothermal')


        self.min_p=0.                #  bar
        self.max_p=500.              #  bar      
        self.min_e= 50.*18.01528      # KJ/kg ---- KJ/Kmol
        self.max_e= 3200.*18.01528   # KJ/kg ---- KJ/Kmol
        
        #######

        # self.min_p=150            #  bar
        # self.max_p=450          #  bar        #  310
        # self.min_e= 490.*18.01528   # KJ/kg ---- KJ/Kmol
        # self.max_e= 1400.*18.01528   # KJ/kg ---- KJ/Kmol

       

        
        
        self.physics = Geothermal(self.timer, n_points=n_points,                                           #  ,mass_rate=True
                                  min_p=self.min_p, max_p=self.max_p, min_e=self.min_e, max_e=self.max_e, cache=False)
                                                            #    min_p=150, max_p=260, min_e=9000, max_e=60000 
                                                            #    min_p=150, max_p=260, min_e=0, max_e=100000 
                                                            #    min_p=1, max_p=351, min_e=0, max_e=100000                
        print('\n')
        
        # self.physics.set_engine(platform = 'cpu')  #---SI sirve
        #self.physics.set_engine(platform = 'gpu')  #---NO sirve
        
        # create pre-defined physics for geothermal

        property_container = PropertyContainer()
   

        property_container.output_props = { 'temp': lambda: property_container.temperature,
                                              #'Enthalpy_W [KJ/kmol]': lambda: property_container.enthalpy[0],
        #                                    'Enthalpy_S [KJ/kmol]': lambda: property_container.enthalpy[1],
        #                                    'Enthalpy_W [KJ/kg]': lambda: property_container.enthalpy[0]/18.01528,
        #                                    'Enthalpy_S [KJ/kg]': lambda: property_container.enthalpy[1]/18.01528,
        #                                    'temp [K]': lambda: property_container.temperature,
        #                                    'temp [°C]': lambda: property_container.temperature-273.15,
        #                                     'sat_W': lambda: property_container.saturation[0],
        #                                     'sat_S': lambda: property_container.saturation[1]                                
                                            }
        
        self.physics.add_property_region(property_container) 

    def calc_well_loc(self):

        self.inj_well_coords = [[0, 1000, 0]]
        self.prod_well_coords = [[1000, 0, 0]] 

        self.bound_cond = 'wells_in_nearest_cell'
        # self.bound_cond == 'wells_in_frac'
        # self.bound_cond == 'wells_in_mat'
        
        # TODO use idx = self.reservoir.find_cell_index(wc)
        # Store number of control volumes (NOTE: in case of fractures, this includes both matrix and fractures):
        self.nb = self.reservoir.discretizer.volume_all_cells.size
        self.num_frac = self.reservoir.discretizer.frac_cells_tot
        self.num_mat = self.reservoir.discretizer.mat_cells_tot
        if self.bound_cond == 'wells_in_frac':
            offset = 0
            left_int = 0
            right_int = self.num_frac
        elif self.bound_cond == 'wells_in_mat':
            offset = self.num_frac
            left_int = self.num_frac
            right_int = self.num_frac + self.num_mat
        elif self.bound_cond == 'wells_in_nearest_cell':
            offset = 0
            left_int = 0
            right_int = self.num_frac + self.num_mat
        else:
            raise('error: wrong self.bound_cond')

        # Find closest control volume to dummy_well point:
        self.injection_wells = []
        dummy_well_inj = self.inj_well_coords

        self.store_dist_to_well_inj = np.zeros((len(dummy_well_inj),))
        self.store_coord_well_inj = np.zeros((len(dummy_well_inj), 3))
        ii = 0
        for ith_inj in dummy_well_inj:
            dist_to_well_point = np.linalg.norm(self.reservoir.discretizer.centroid_all_cells[left_int:right_int] - ith_inj,
                                                axis=1)
            cell_id = np.argmin(dist_to_well_point) + offset
            self.injection_wells.append(cell_id)

            self.store_coord_well_inj[ii, :] = self.reservoir.discretizer.centroid_all_cells[cell_id]
            self.store_dist_to_well_inj[ii] = np.min(dist_to_well_point)
            ii += 1

        self.production_wells = []
        dummy_well_prod = self.prod_well_coords

        self.store_dist_to_well_prod = np.zeros((len(dummy_well_prod),))
        self.store_coord_well_prod = np.zeros((len(dummy_well_prod), 3))
        ii = 0
        for ith_prod in dummy_well_prod:
            dist_to_well_point = np.linalg.norm(self.reservoir.discretizer.centroid_all_cells[left_int:right_int] - ith_prod,
                                                axis=1)
            cell_id = np.argmin(dist_to_well_point) + offset
            self.production_wells.append(cell_id)

            self.store_coord_well_prod[ii, :] = self.reservoir.discretizer.centroid_all_cells[cell_id]
            self.store_dist_to_well_prod[ii] = np.min(dist_to_well_point)
            ii += 1

        self.well_perf_loc = np.array([self.injection_wells, self.production_wells])
    
    def set_wells(self):
        

        self.calc_well_loc()
        
        # add well
        self.reservoir.add_well("INJ", wellbore_diameter=0.3048)
        self.reservoir.add_perforation("INJ", cell_index=self.well_perf_loc[0][0],  skin=0.0,
                                           well_radius=0.1524, multi_segment=False,well_indexD=0)

        # add well
        self.reservoir.add_well("PRD", wellbore_diameter=0.3048)
        self.reservoir.add_perforation("PRD", cell_index=self.well_perf_loc[1][0],  skin=0.0,
                                           well_radius=0.1524, multi_segment=False,well_indexD=0)
        
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
        
        P_prod=50                        #     [bar]
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
                                               inj_temp=100 + 273.15   #  [c]-----[k] 
                                               )
            else:
                self.physics.set_well_controls(wctrl=w.control,
                                               control_type=well_control_iface.BHP, 
                                               is_inj=False, 
                                               target=P_prod
                                               )
                
     
                             
      