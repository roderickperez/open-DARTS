from re import L
from darts.models.darts_model import DartsModel
#from darts.models.cicd_model import CICDModel
#from darts.physics.properties.iapws.iapws_property_vec import _Backward1_T_Ph_vec
from darts.tools.keyword_file_tools import load_single_keyword
import numpy as np
from darts.engines import value_vector, sim_params, well_control_iface

from darts.physics.geothermal.physics import Geothermal
from darts.physics.geothermal.property_container import PropertyContainer

import os

from darts.reservoirs.unstruct_reservoir import UnstructReservoir
import meshio   # There are various mesh formats available for representing unstructured meshes. 
                # meshio can read and write all of the following and smoothly converts between them.

########################

#from darts.physics.properties.iapws.iapws_property import *
#from darts.physics.properties.basic import ConstFunc


class Model(DartsModel):
    
    def __init__(self, input_data,  n_points=64):
        
        self.input_data=input_data
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

        self.timer.node["initialization"].stop()
        
            
    def set_reservoir(self):
        

        print('\n----------------DEFINE RESERVOIR :')
        print('\n')
        
        
        # fname = '_' + 'raw_lc' + '_' + '80' + '.msh'
        # mesh_file = os.path.join('meshes', 'case_1' + fname)
        # print(mesh_file)

        #fname = 'output_raw_lc_300.msh'

        fname = f"output_raw_lc_{self.input_data['char_len']}.msh"
        mesh_file = os.path.join('meshes', fname)
        print(mesh_file)


        permx = 100.0 # [mD]
        permy = 100.0  # [mD]
        permz = 100.0  # [mD]
        poro = 0.005 #   0.005 = .5%
        hcap = 2470.0  # [kJ/m3/K]
        rcond = 172.8   # [kJ/m/day/K]

        # frac_aper= 0.01  # (initial) fracture aperture [m] =  1e-5 [m] =  0.001 [m] = 10 [mm]
        frac_aper= 0.0001  # (initial) fracture aperture [m] =  1e-3 [m] =  0.001 [m] = .1 [mm]
        # initialize reservoir
        self.reservoir = UnstructReservoir(timer=self.timer, mesh_file=mesh_file,
                                      permx=permx, permy=permy, permz=permz,
                                      poro=poro,
                                      rcond=rcond,
                                      hcap=hcap,
                                      frac_aper=frac_aper)
                                        
        # read mesh to get the number of fractures for tags specification
        # assume mesh is extruded and fractures have a quad shape
        # fracture tags start from 90000 according to .geo file generation code
        msh = meshio.read(mesh_file)
        c = msh.cell_data_dict['gmsh:physical']
        
        # cuenta cuántas fracturas hay. obtiene todos los tags únicos asignados a celdas cuadriláteras y filtra
        # solo los que corresponden a fracturas (mayores a 90000).
        n_fractures = (np.unique(c['quad']) >= 90000).sum()
        # n_fractures = n_fractures * (1 + int(input_data['overburden_layers']>0) + int(input_data['underburden_layers']>0))

        # Asigna los tags físicos [9991, 9992, 9993, 9994, 9995] para la matriz de roca.
        # 9991 - rsv, 9992 - overburden, 9993 - underburden, 9994 - overburden2, 9995 - underburden2
        # •	9991: matriz principal (reservorio)
        # •	9992: sobrecarga
        # •	9993: bajo carga
        # •	9994 y 9995: capas extra si las hay
        self.reservoir.physical_tags['matrix'] = [9991 + i for i in range(5)]
        # multiplied by 3 because physical surfaces for fracture are also in underburden and overburden
        self.reservoir.physical_tags['fracture'] = [90000 + i for i in range(n_fractures)]
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

       

        self.physics = Geothermal(self.timer, n_points=n_points,
                                  min_p=self.min_p, max_p=self.max_p, min_e=self.min_e, max_e=self.max_e, cache=False)
                                                            #    min_p=150, max_p=260, min_e=9000, max_e=60000 
                                                            #    min_p=150, max_p=260, min_e=0, max_e=100000 
                                                            #    min_p=1, max_p=351, min_e=0, max_e=100000                
        print('\n')
        
        # self.physics.set_engine(platform = 'cpu')  #---SI sirve
        #self.physics.set_engine(platform = 'gpu')  #---NO sirve
        
        
        property_container = PropertyContainer()

        
        self.physics.add_property_region(property_container)      

        property_container.output_props = {'temp [K]': lambda: property_container.temperature                             
                                            }
     
    def set_initial_conditions(self):

            # initialization with constant pressure and temperature
            self.input_distribution = {"pressure": 150.,              # [bar]
                                    "temperature": 300. + 273.15  # [c]-----[k] 

                                    }
            self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh,
                                                                input_distribution=self.input_distribution)
        

            state_init = value_vector([self.input_distribution['pressure']])               # Initial presion   (bar)   

            enth_init = self.physics.property_containers[0].enthalpy_ev['total'].evaluate(state_init,
                                                                                            self.input_distribution['temperature'] )
                
            estado=np.array([state_init[0],enth_init ])
            Sw_init=self.physics.property_containers[0].saturation_ev['water'].evaluate(estado)


            print('\n----------------INITIAL CONDITIONS :')
            print('\n')
            print('Initial pressure (bar)=',self.input_distribution['pressure'])
            print('Initial temperature (C)=',self.input_distribution['temperature']-273.15)
            print('Initial enthalpy ([kJ/kg])=',enth_init/18.01528)
            print('Initial enthalpy ([kJ/kmol])=',enth_init)
            print('Initial Saturation (water)=',Sw_init)        
            print('\n')    
                    
    def calc_well_loc(self):

        self.bound_cond = 'wells_in_nearest_cell'
        # self.bound_cond == 'wells_in_frac'
        # self.bound_cond == 'wells_in_mat'

        self.inj_well_coords = self.input_data['inj_well_coords'] 
        self.prod_well_coords = self.input_data['prod_well_coords']

      
        
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
        self.Mass_iny=4.0        #     [Kg/s]  
        self.mol_rate_iny=self.Mass_iny*4795.928784     #     [Kg/s]  --------> [kmol/day]
        self.inj_temp= 100                  #     [c]
        print('\n---------Injector :')
        print('INJ mass rate [Kg/s]=',self.Mass_iny)
        print('INJ mass rate [Kg/day]=',self.Mass_iny*60*60*24)
        print('INJ mass rate [kmol/day]=',self.mol_rate_iny)
        print('INJ  temp. [c]=',self.inj_temp)
                       
        
        self.P_prod=50                        #     [bar]
        #self.P_prod=50                        #     [bar] 
        
        print('\n---------Producer :')
        print('PRD pressure [bar]=',self.P_prod)
        print('\n---------------------------')
        
        for i, w in enumerate(self.reservoir.wells):
            if 'INJ' in w.name:
                #w.control = self.physics.new_mass_rate_water_inj(mass_rate,ent_inj)
                self.physics.set_well_controls(wctrl=w.control,
                                               control_type=well_control_iface.MOLAR_RATE,
                                               is_inj=True,
                                               target=self.mol_rate_iny,
                                               phase_name='water', 
                                               inj_temp=self.inj_temp + 273.15   #  [c]-----[k] 
                                               )
            else:
                #w.control = self.physics.new_bhp_prod(P_prod)
                self.physics.set_well_controls(wctrl=w.control,
                                               control_type=well_control_iface.BHP, 
                                               is_inj=False, 
                                               target=self.P_prod
                                               )
                
      