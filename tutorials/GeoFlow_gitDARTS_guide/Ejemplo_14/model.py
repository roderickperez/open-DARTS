import numpy as np
from darts.models.darts_model import DartsModel
#from darts.reservoirs.struct_reservoir import StructReservoir
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
# import meshio   # There are various mesh formats available for representing unstructured meshes. 
                # meshio can read and write all of the following and smoothly converts between them.

from darts.engines import conn_mesh,  index_vector, value_vector
from darts.reservoirs.mesh.unstruct_discretizer import UnstructDiscretizer


from scipy.interpolate import interp1d



from dataclasses import dataclass

@dataclass
class Layer_prop:
    poro: float
    perm: float
    anisotropy: list = None
    hcap: float = 2200
    rcond: float = 181.44





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

    def set_physics(self, n_points):
        # create pre-defined physics for geothermal
 
        print('Reservoir physics: Geothermal')


        self.min_p=0.                #  bar
        self.max_p=500.              #  bar      
        self.min_e= 50.*18.01528      # KJ/kg ---- KJ/Kmol
        self.max_e= 3200.*18.01528   # KJ/kg ---- KJ/Kmol
        
        #######



        
        self.physics = Geothermal(self.timer, n_points=n_points,                                           
                                  min_p=self.min_p, max_p=self.max_p, min_e=self.min_e, max_e=self.max_e, cache=False)
                                                            #    min_p=150, max_p=260, min_e=9000, max_e=60000 
                                                            #    min_p=150, max_p=260, min_e=0, max_e=100000 
                                                            #    min_p=1, max_p=351, min_e=0, max_e=100000                
        print('\n')
        

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

    def set_initial_conditions(self):
        
        #input_depth = [1516.666, 1550, 1583.333]
        input_depth = [1516.666, 1550, 1583.333]

        input_distribution = {            
            "pressure": [86., 95, 105],      
            "temperature": [300.0 + 273.15, 305.0 + 273.15, 310.0 + 273.15]  # K
        }

        self.physics.set_initial_conditions_from_depth_table(mesh=self.reservoir.mesh, 
                                                             input_distribution=input_distribution, input_depth=input_depth)
        
        
        
        
        # # initialization with constant pressure and temperature
        # input_distribution = {"pressure": self.init_pressure,              # [bar]
        #                       "temperature": self.init_temp + 273.15  # [c]-----[k] 
        #                      }
        # self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh, input_distribution=input_distribution)
   

        # self.state_init = value_vector([input_distribution['pressure']])               # Initial presion   (bar)   
        # enth_init = self.physics.property_containers[0].enthalpy_ev['total'].evaluate(self.state_init,
        #                                                                             input_distribution['temperature'] )
        
        # estado=np.array([self.state_init[0],enth_init ])
        # Sw_init=self.physics.property_containers[0].saturation_ev['water'].evaluate(estado)


        # print('\n----------------INITIAL CONDITIONS :')
        # print('\n')
        # print('Initial pressure (bar)=',input_distribution['pressure'])
        # print('Initial temperature (C)=',input_distribution['temperature']-273.15)
        # print('Initial temperature (k)=',input_distribution['temperature'])
        # print('Initial enthalpy ([kJ/kg])=',enth_init/18.01528)
        # print('Initial enthalpy ([kJ/kmol])=',enth_init)
        # print('Initial Saturation (water)=',Sw_init)        
        # print('\n')

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
                                        
    
        self.reservoir.physical_tags['matrix'] = [9991, 9992, 9993   ]  # 
        self.reservoir.physical_tags['boundary'] = [1,2,30,31,32,40,41,42,50,51,52,60,61,62]  # order: Z- (bottom); Z+ (top) ; Y-; X+; Y+; X-


        print('Reservoir type= Unstructured')


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
                                               inj_temp=inj_temp + 273.15   #  [c]-----[k] 
                                               )
            else:
                self.physics.set_well_controls(wctrl=w.control,
                                               control_type=well_control_iface.BHP, 
                                               is_inj=False, 
                                               target=P_prod
                                               )
                
     
                             
def patched_set_initial_conditions_from_depth_table(
    self,
    mesh,
    input_distribution: dict,
    input_depth,
    global_to_local=None,
):
    """
    Parche: igual que la original pero corrige el loop con .items()
    """

    # Chequeos
    assert "pressure" in input_distribution.keys() and (
        "temperature" in input_distribution.keys()
        or "enthalpy" in input_distribution.keys()
    )
    input_depth = (
        input_depth if not np.isscalar(input_depth) else np.array([input_depth])
    )

    # 🔧 Corrección aquí: usamos .items()
    for key, input_values in input_distribution.items():
        input_values = (
            input_values
            if not np.isscalar(input_values)
            else np.ones(len(input_depth)) * input_values
        )
        assert len(input_values) == len(input_depth)

    depths = np.asarray(mesh.depth)[: mesh.n_res_blocks]
    if global_to_local is not None:
        depths = depths[global_to_local]

    mesh.initial_state.resize(mesh.n_res_blocks * self.n_vars)

    for ith_var, variable in enumerate(self.vars):
        if variable == "enthalpy" and "enthalpy" not in input_distribution.keys():
            p_itor = interp1d(
                input_depth,
                input_distribution["pressure"],
                kind="linear",
                fill_value="extrapolate",
            )
            pressure = p_itor(depths)

            t_itor = interp1d(
                input_depth,
                input_distribution["temperature"],
                kind="linear",
                fill_value="extrapolate",
            )
            temperature = t_itor(depths)

            values = np.empty(mesh.n_res_blocks)
            for j in range(mesh.n_res_blocks):
                state_pt = np.array([pressure[j], temperature[j]])
                values[j] = self.property_containers[0].compute_total_enthalpy(
                    state_pt
                )
        else:
            itor = interp1d(
                input_depth,
                input_distribution[variable],
                kind="linear",
                fill_value="extrapolate",
            )
            values = itor(depths)

        np.asarray(mesh.initial_state)[ith_var :: self.n_vars] = values


def patched_set_layer_properties(self):
        
        
        layer_props = {9991: Layer_prop( poro=0.01, perm=10., anisotropy=[1, 1, 1]) ,  # capa de hasta abjo
               9992: Layer_prop(poro=0.01, perm=20., anisotropy=[1, 1, 1]),    
                9993: Layer_prop(poro=0.01, perm=40., anisotropy=[1, 1, 1]) }
        

        # Extract layer type of each cell
        self.layers = {}
        self.seal = []
        for tag in self.discretizer.physical_tags['matrix']:
            self.layers[tag] = []

        for geometry, tags in sorted(list(self.discretizer.mesh_data.cell_data_dict['gmsh:physical'].items()),
                                     key=lambda x: -x[1][0]):
            # Main loop over different existing geometries
            for ith_cell, nodes_to_cell in enumerate(self.discretizer.mesh_data.cells_dict[geometry]):
                tag = tags[ith_cell]
                if tag in self.discretizer.physical_tags['matrix']:
                    self.layers[tags[ith_cell]].append(ith_cell)
                

        # Assign properties to layers
        n_layers = len(layer_props)   #   31
        layer_op_num = np.zeros(n_layers)
        layer_poro = np.zeros(n_layers)
        layer_hcap = np.zeros(n_layers)
        layer_rcond = np.zeros(n_layers)
        layer_perm = np.zeros((n_layers, 3))
        for i, (layer, porperm) in enumerate(layer_props.items()):
            layer_op_num[i] = 0                #          revisar!!
            layer_poro[i] = porperm.poro
            layer_hcap[i] = porperm.hcap
            layer_rcond[i] = porperm.rcond
            for j in range(3):
                layer_perm[i, j] = porperm.perm * porperm.anisotropy[j]

        self.poro = np.zeros(self.discretizer.volume_all_cells.size)
        self.hcap = np.zeros(self.discretizer.volume_all_cells.size)
        self.rcond = np.zeros(self.discretizer.volume_all_cells.size)
        self.op_num = np.zeros(self.discretizer.volume_all_cells.size)

        ith_layer = -1
        for tag, layer in self.layers.items():
            ith_layer += 1
            for ith_cell in layer:
                self.discretizer.mat_cell_info_dict[ith_cell].permeability = layer_perm[ith_layer, :]
                self.poro[ith_cell] = layer_poro[ith_layer]
                self.hcap[ith_cell] = layer_hcap[ith_layer]
                self.rcond[ith_cell] = layer_rcond[ith_layer]
                self.op_num[ith_cell] = layer_op_num[ith_layer]

        return


def patched_UnstructReservoir_discretize(self, cache: bool = False, verbose: bool = False):
        self.discretizer = UnstructDiscretizer(mesh_file=self.mesh_file, physical_tags=self.physical_tags, verbose=verbose)

        self.discretizer.n_dim = 3

        
        ##############        Lineas omitidas de la clase original     #################

        # if self.frac_aper is not None and self.sh_max is not None:
        #     self.discretizer.calc_frac_aper_by_stress(
        #         self.frac_aper,
        #         self.sh_max,
        #         self.sh_min,
        #         self.sh_max_azimuth,
        #         self.sigma_c,
        #     )
        ##################################################################################


        # Use class method load_mesh to load the GMSH file specified above:
        self.discretizer.load_mesh(permx=self.permx, permy=self.permy, permz=self.permz, frac_aper=0, cache=cache)


        
        ##############3        Lineas Agregadas      #################

        start_z= 1500.

        for ith_cell in self.discretizer.mat_cell_info_dict:
            self.discretizer.mat_cell_info_dict[ith_cell].depth += start_z
            
            # Sumar al último elemento de centroid (z en un vector 3x1)
            # MODIFICAR EL CENTROIDE CAMBIA COMPLETAMENTE LA SIMULACION, EVITARLO  !!!
            #self.discretizer.mat_cell_info_dict[ith_cell].centroid[-1] += start_z


         ################################################################# 


        # Store volumes and depth to single numpy arrays:
        self.discretizer.store_volume_all_cells()
        self.discretizer.store_depth_all_cells()
        self.discretizer.store_centroid_all_cells()

        # Assign layer properties
        self.set_layer_properties()

        # Perform discretization:
        cell_m, cell_p, tran, tran_thermal = self.discretizer.calc_connections_all_cells(cache=cache)

        # Initialize mesh with all four parameters (cell_m, cell_p, trans, trans_D):
        # Create mesh object (C++ object used by DARTS for all mesh related quantities):
        self.mesh = conn_mesh()
        self.mesh.init(index_vector(cell_m), index_vector(cell_p), value_vector(tran), value_vector(tran_thermal))

        # Create numpy arrays wrapped around mesh data (no copying, this will severely slow down the process!)
        np.array(self.mesh.poro, copy=False)[:] = self.poro
        np.array(self.mesh.rock_cond, copy=False)[:] = self.rcond
        np.array(self.mesh.heat_capacity, copy=False)[:] = self.hcap
        np.array(self.mesh.op_num, copy=False)[:] = self.op_num
        n_elements=self.discretizer.mat_cells_tot + self.discretizer.frac_cells_tot
        np.array(self.mesh.depth, copy=False)[:] = self.discretizer.depth_all_cells[:n_elements]
        np.array(self.mesh.volume, copy=False)[:] = self.discretizer.volume_all_cells[:n_elements]

        
        ##############3                         Lineas agregadas                                    #################
        ##############3   (para que el arhcivo mesh.vts contenga las permeabilidades correctas ?)   #################
        
        # número de celdas
        n_cells = len(self.discretizer.mat_cell_info_dict)

        # arreglo vacío 1D
        self.permx = np.zeros(n_cells)
        self.permy = np.zeros(n_cells)
        self.permz = np.zeros(n_cells)

        # llenar con kx de cada celda
        for i, ith_cell in enumerate(self.discretizer.mat_cell_info_dict):
            self.permx[i] = self.discretizer.mat_cell_info_dict[ith_cell].permeability[0]  
            self.permy[i] = self.discretizer.mat_cell_info_dict[ith_cell].permeability[1] 
            self.permz[i] = self.discretizer.mat_cell_info_dict[ith_cell].permeability[2] 


        return self.mesh

    


# 🔗 Parcheamos el método de Geothermal
Geothermal.set_initial_conditions_from_depth_table = (
    patched_set_initial_conditions_from_depth_table
)

# 🔗 Parcheamos el método
UnstructReservoir.discretize = (
    patched_UnstructReservoir_discretize
)

# 🔗 Parcheamos el método
UnstructReservoir.set_layer_properties = (
    patched_set_layer_properties
)


