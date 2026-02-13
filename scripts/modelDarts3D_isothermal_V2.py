# modelDarts3D_isothermal_V2.py - Version 2 (Vertical Heterogeneity)
import os
import warnings
import numpy as np
import pandas as pd
try:
    import pyvista as pv
except ImportError:
    pv = None
    print("Warning: PyVista not found. 3D visualization will be disabled.")
import matplotlib.pyplot as plt

# Force software rendering for headless environments
os.environ['MESA_LOADER_DRIVER_OVERRIDE'] = 'swrast'
os.environ['GALLIUM_DRIVER'] = 'llvmpipe'
os.environ['DISPLAY'] = ':99.0'

# Suppress PyVista deprecation warnings
if pv:
    from pyvista.plotting.utilities.xvfb import PyVistaDeprecationWarning
    warnings.filterwarnings("ignore", category=PyVistaDeprecationWarning)
    try:
        pv.start_xvfb()
    except (OSError, RuntimeError, ImportError):
        pv.OFF_SCREEN = True
        print("Xvfb not found or failed to start. Using off-screen rendering fallback.")

from darts.models.darts_model import DartsModel
from darts.reservoirs.struct_reservoir import StructReservoir
from darts.physics.super.physics import Compositional, PhysicsBase
from darts.physics.super.property_container import PropertyContainer
from darts.physics.properties.basic import ConstFunc
from darts.physics.properties.density import DensityBasic
from darts.engines import value_vector, redirect_darts_output, well_control_iface, sim_params

# --- SIMULATION CONFIGURATION ---
NX, NY, NZ = 100, 100, 5           
DX, DY, DZ = 10.0, 10.0, 10.0       
TOTAL_DAYS = 100                    
USE_GPU = False                     

# Well Locations (0-based grid indices)
INJ_COORD  = [4, 4]    
PROD_COORD = [94, 94]  

# Operation Parameters
INJ_BHP  = 300.0   
PROD_BHP = 150.0   

# Fluid Properties
WATER_DENS_0 = 1000.0   
WATER_COMPR  = 1e-5     
WATER_VISC   = 0.5      

# ANSI Colors
C_RED = '\033[1;91m'
C_CYAN = '\033[1;96m'
C_WHITE = '\033[1;97m'
C_END = '\033[0m'

DARTS_BANNER = f"""
{C_RED} ██████╗   █████╗  ██████╗  ████████╗ ███████╗
 ██╔══██╗ ██╔══██╗ ██╔══██╗ ╚══██╔══╝ ██╔════╝
 ██║  ██║ ███████║ ██████╔╝    ██║    ███████╗
 ██║  ██║ ██╔══██║ ██╔══██╗    ██║    ╚════██║
 ██████╔╝ ██║  ██║ ██║  ██║    ██║    ███████║
 ╚═════╝  ╚═╝  ╚═╝ ╚═╝  ╚═╝    ╚═╝    ╚══════╝{C_END}

{C_CYAN} >>>>>>>>> {C_WHITE}[ GEOTHERMAL: ISOTHERMAL - VERSION 2 (HETEROGENEOUS) ]{C_CYAN} <<<<<<<<<<<< {C_END}
"""

# --- HETEROGENEITY SETUP ---
def generate_property_arrays(nx, ny, nz):
    """
    Generates 1D arrays (flattened 3D) for DARTS StructReservoir.
    DARTS orders cells: Loop X, then Y, then Z.
    """
    # Define properties per layer (Top -> Bottom)
    layer_data = [
        # Layer 1: Moderate Rock (User example: 10% poro, 200 perm)
        {'poro': 0.10, 'perm': 200.0,  'hcap': 2300.0, 'rcond': 160.0},
        
        # Layer 2: High Quality Reservoir (25% poro -> Very high perm)
        {'poro': 0.25, 'perm': 1500.0, 'hcap': 2500.0, 'rcond': 190.0}, 
        
        # Layer 3: Transition Zone (15% poro -> Moderate perm)
        {'poro': 0.15, 'perm': 500.0,  'hcap': 2400.0, 'rcond': 175.0},
        
        # Layer 4: Tight Seal/Barrier (5% poro -> Almost no flow)
        {'poro': 0.05, 'perm': 5.0,    'hcap': 2100.0, 'rcond': 140.0},
        
        # Layer 5: Deep Tight Sand (10% poro -> Low perm)
        {'poro': 0.10, 'perm': 80.0,   'hcap': 2250.0, 'rcond': 160.0}
    ]
    
    total_cells = nx * ny * nz
    
    # Initialize empty arrays
    poro_arr = np.zeros(total_cells)
    permx_arr = np.zeros(total_cells)
    permy_arr = np.zeros(total_cells)
    permz_arr = np.zeros(total_cells)
    hcap_arr = np.zeros(total_cells)
    rcond_arr = np.zeros(total_cells)

    cells_per_layer = nx * ny
    
    for k in range(nz):
        props = layer_data[k] if k < len(layer_data) else layer_data[-1]
        
        start_idx = k * cells_per_layer
        end_idx = (k + 1) * cells_per_layer
        
        poro_arr[start_idx:end_idx]  = props['poro']
        permx_arr[start_idx:end_idx] = props['perm']
        permy_arr[start_idx:end_idx] = props['perm']
        permz_arr[start_idx:end_idx] = props['perm'] * 0.1 
        hcap_arr[start_idx:end_idx]  = props['hcap']
        rcond_arr[start_idx:end_idx] = props['rcond']

    return poro_arr, permx_arr, permy_arr, permz_arr, hcap_arr, rcond_arr

class ModelProperties(PropertyContainer):
    def __init__(self, phases_name, components_name, min_z=1e-11):
        self.nph = len(phases_name)
        self.nc  = len(components_name)
        Mw = np.array([18.015] * self.nc) 
        super().__init__(phases_name=phases_name, components_name=components_name, Mw=Mw, min_z=min_z, temperature=350.0)

    def get_state(self, state):
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]
        zc = np.append(vec_state_as_np[1:], 1 - np.sum(vec_state_as_np[1:]))
        temperature = self.temperature 
        return pressure, temperature, zc

    def evaluate(self, state):
        pressure, temperature, zc = self.get_state(state)
        self.clean_arrays()
        self.ph = np.array([0], dtype=np.intp)
        for i in range(self.nc):
            self.x[0, i] = zc[i]
        for j in self.ph:
            M = np.sum(self.x[j, :] * self.Mw) + 1e-20
            self.dens[j] = self.density_ev[self.phases_name[j]].evaluate(pressure, temperature)
            self.dens_m[j] = self.dens[j] / M
            self.mu[j] = self.viscosity_ev[self.phases_name[j]].evaluate(pressure, temperature)
            self.enthalpy[j] = self.enthalpy_ev[self.phases_name[j]].evaluate(pressure, temperature)
            self.cond[j] = self.conductivity_ev[self.phases_name[j]].evaluate(pressure, temperature)
        self.sat[0] = 1.0
        for j in self.ph:
            self.kr[j] = 1.0 
            self.pc[j] = 0

class SimulationModel(DartsModel):
    def __init__(self):
        super().__init__()
        self.timer.node["initialization"].start()
        self.nx, self.ny, self.nz = NX, NY, NZ
        
        # 1. GENERATE THE LAYERED PROPERTIES
        poro, permx, permy, permz, hcap, rcond = generate_property_arrays(NX, NY, NZ)

        # 2. INITIALIZE RESERVOIR WITH ARRAYS
        self.reservoir = StructReservoir(self.timer, nx=NX, ny=NY, nz=NZ, dx=DX, dy=DY, dz=DZ,
                                         permx=permx, permy=permy, permz=permz,
                                         poro=poro, hcap=hcap, rcond=rcond, 
                                         depth=2000)
        
        self.set_physics()
        if USE_GPU:
            self.params.linear_type = sim_params.linear_solver_t.gpu_gmres_cpr_amgx_ilu
        else:
            self.params.linear_type = sim_params.linear_solver_t.cpu_superlu
        self.set_sim_params(first_ts=0.005, mult_ts=1.5, max_ts=5.0, runtime=TOTAL_DAYS, tol_newton=1e-3, tol_linear=1e-5)
        self.timer.node["initialization"].stop()
        
    def set_physics(self):
        zero = 1e-13
        components = ["H2O_Cold", "H2O_Hot"] 
        phases = ["wat"]
        self.inj_comp = value_vector([1.0 - zero]) 
        self.ini_comp = value_vector([0.5])         
        property_container = ModelProperties(phases_name=phases, components_name=components)
        property_container.density_ev = dict([('wat', DensityBasic(compr=WATER_COMPR, dens0=WATER_DENS_0))])
        property_container.viscosity_ev = dict([('wat', ConstFunc(WATER_VISC))])
        property_container.enthalpy_ev = dict([('wat', ConstFunc(100.0))])
        property_container.rock_hcap_ev = ConstFunc(1000.0)
        property_container.conductivity_ev = dict([('wat', ConstFunc(10.0))])
        self.physics = Compositional(components, phases, self.timer, n_points=50, 
                                     min_p=100, max_p=500, min_z=zero, max_z=1 - zero,
                                     state_spec=PhysicsBase.StateSpecification.P)
        self.physics.add_property_region(property_container)
        
    def set_wells(self):
        self.reservoir.add_well("INJ")
        for k in range(1, NZ+1):
            self.reservoir.add_perforation("INJ", cell_index=(INJ_COORD[0]+1, INJ_COORD[1]+1, k))
        self.reservoir.add_well("PROD")
        for k in range(1, NZ+1):
            self.reservoir.add_perforation("PROD", cell_index=(PROD_COORD[0]+1, PROD_COORD[1]+1, k))
            
    def set_initial_conditions(self):
        input_distribution = {self.physics.vars[0]: 200.0,
                              self.physics.vars[1]: self.ini_comp[0]}
        self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh, input_distribution=input_distribution)

    def set_well_controls(self):
        for w in self.reservoir.wells:
            if "INJ" in w.name:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP, 
                                               is_inj=True, target=INJ_BHP, inj_composition=self.inj_comp)
            else:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP, 
                                               is_inj=False, target=PROD_BHP)

def export_wells_to_vtk(reservoir, output_directory):
    if not pv: return
    wells_data = []
    for well in reservoir.wells:
        points = []
        for perf in well.perforations:
            res_idx = perf[1]
            if res_idx >= 0:
                coords = reservoir.discretizer.centroids_all_cells[res_idx]
                coords[2] -= 2000.0 
                points.append(coords)
        if points:
            line = pv.MultipleLines(points=np.array(points))
            line["name"] = well.name
            wells_data.append(line)
    if wells_data:
        combined = wells_data[0]
        if len(wells_data) > 1:
            combined = wells_data[0].merge(wells_data[1:])
        os.makedirs(output_directory, exist_ok=True)
        combined.save(os.path.join(output_directory, "wells.vtp"))

def plot_reservoir_maps(nx, ny, nz, run_dir):
    poro, permx, permy, permz, hcap, rcond = generate_property_arrays(nx, ny, nz)
    poro_3d = poro.reshape(nz, ny, nx)
    perm_3d = permx.reshape(nz, ny, nx)
    layer_idx = 1
    poro_map, perm_map = poro_3d[layer_idx, :, :], perm_3d[layer_idx, :, :]
    mid_y = ny // 2
    poro_xs, perm_xs = poro_3d[:, mid_y, :], perm_3d[:, mid_y, :]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    im1 = axes[0,0].imshow(poro_map, origin='lower', cmap='Blues', extent=[0, nx*DX, 0, ny*DY])
    axes[0,0].scatter(INJ_COORD[0]*DX, INJ_COORD[1]*DY, c='c', s=100, edgecolors='k', label='Inj')
    axes[0,0].scatter(PROD_COORD[0]*DX, PROD_COORD[1]*DY, c='g', s=100, marker='^', edgecolors='k', label='Prod')
    axes[0,0].set_title(f'Map View: Porosity (Layer {layer_idx+1})'); fig.colorbar(im1, ax=axes[0,0]); axes[0,0].legend()
    im2 = axes[0,1].imshow(perm_map, origin='lower', cmap='viridis', extent=[0, nx*DX, 0, ny*DY])
    axes[0,1].set_title(f'Map View: Permeability (Layer {layer_idx+1})'); fig.colorbar(im2, ax=axes[0,1])
    im3 = axes[1,0].imshow(poro_xs, origin='upper', cmap='Blues', extent=[0, nx*DX, nz*DZ, 0])
    axes[1,0].set_title(f'Cross Section: Porosity'); axes[1,0].set_aspect('auto'); fig.colorbar(im3, ax=axes[1,0])
    im4 = axes[1,1].imshow(perm_xs, origin='upper', cmap='plasma', extent=[0, nx*DX, nz*DZ, 0])
    axes[1,1].set_title(f'Cross Section: Permeability'); axes[1,1].set_aspect('auto'); fig.colorbar(im4, ax=axes[1,1])
    plt.suptitle('Geological Model: 5-Layer Heterogeneity (Isothermal V2)', fontsize=16)
    os.makedirs(os.path.join(run_dir, 'figures'), exist_ok=True)
    plt.savefig(os.path.join(run_dir, 'figures/geothermal_map_isothermal.png'), bbox_inches='tight')
    plt.close()

def main():
    print(DARTS_BANNER)
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join('output/runs', f'isothermal_V2_{ts}')
    vtk_path = os.path.join(run_dir, 'vtk_files')
    os.makedirs(vtk_path, exist_ok=True)
    plot_reservoir_maps(NX, NY, NZ, run_dir)
    
    class DartsSilence:
        def __enter__(self):
            self.stdout_fd, self.stderr_fd = os.dup(1), os.dup(2)
            self.devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(self.devnull, 1); os.dup2(self.devnull, 2)
        def __exit__(self, exc_type, exc_val, exc_tb):
            os.dup2(self.stdout_fd, 1); os.dup2(self.stderr_fd, 2)
            os.close(self.stdout_fd); os.close(self.stderr_fd); os.close(self.devnull)

    log_file = os.path.join(run_dir, 'simulation.log')
    redirect_darts_output(log_file)
    m = SimulationModel()
    with DartsSilence(): m.init()
    m.set_output(output_folder=vtk_path)
    export_wells_to_vtk(m.reservoir, vtk_path)
    
    print(f"Starting {TOTAL_DAYS}-day Heterogeneous Simulation...")
    # Add geological properties to output list
    prop_list = m.physics.vars + ['poro', 'permx', 'permy', 'permz']
    with DartsSilence(): m.output.output_to_vtk(output_properties=prop_list, ith_step=0)
    
    for i in range(1, TOTAL_DAYS + 1):
        with DartsSilence():
            m.run(1.0)
            m.output.output_to_vtk(output_properties=prop_list, ith_step=i)
        if i % 10 == 0: print(f"Progress: {i}/{TOTAL_DAYS} days")
        
    print(f"\nSimulation [ISOTHERMAL V2] complete.")
    df_history = pd.DataFrame.from_dict(m.physics.engine.time_data)
    df_history.to_csv(os.path.join(run_dir, 'history_isothermal.csv'))
    print(f"Results: {run_dir}")

if __name__ == "__main__":
    main()
