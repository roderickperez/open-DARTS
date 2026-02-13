# modelDarts3D_thermal_V2.py - Version 2 (Vertical Heterogeneity)
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- VISUALIZATION SETUP ---
try:
    import pyvista as pv
except ImportError:
    pv = None

# Force software rendering for headless environments
os.environ['MESA_LOADER_DRIVER_OVERRIDE'] = 'swrast'
os.environ['GALLIUM_DRIVER'] = 'llvmpipe'
os.environ['DISPLAY'] = ':99.0'

if pv:
    from pyvista.plotting.utilities.xvfb import PyVistaDeprecationWarning
    warnings.filterwarnings("ignore", category=PyVistaDeprecationWarning)
    try:
        pv.start_xvfb()
    except (OSError, RuntimeError, ImportError):
        pv.OFF_SCREEN = True

# --- DARTS IMPORTS ---
from darts.models.darts_model import DartsModel
from darts.reservoirs.struct_reservoir import StructReservoir
from darts.engines import value_vector, redirect_darts_output, well_control_iface, sim_params
from darts.engines import property_evaluator_iface

# Geothermal Physics Imports
from darts.physics.geothermal.physics import Geothermal as GeothermalPhysicsBase
from darts.physics.geothermal.geothermal import GeothermalIAPWSProperties
from darts.physics.properties.iapws.custom_rock_property import custom_rock_compaction_evaluator, custom_rock_energy_evaluator
from darts.physics.properties.iapws.iapws_property import (
    iapws_temperature_evaluator,
    iapws_water_enthalpy_evaluator, iapws_steam_enthalpy_evaluator, iapws_total_enthalpy_evalutor,
    iapws_water_density_evaluator, iapws_steam_density_evaluator,
    iapws_water_viscosity_evaluator, iapws_steam_viscosity_evaluator,
    iapws_water_saturation_evaluator, iapws_steam_saturation_evaluator,
    iapws_water_relperm_evaluator, iapws_steam_relperm_evaluator
)
from iapws.iapws97 import _Region1
from darts.physics.properties.basic import ConstFunc

# --- CONFIGURATION ---
NX, NY, NZ = 100, 100, 5    
DX, DY, DZ = 10.0, 10.0, 10.0 
TOTAL_DAYS = 100

# Well Locations (0-based)
INJ_COORD  = [4, 4]      
PROD_COORD = [94, 94]    

# Operation
INJ_RATE = 500.0        # m3/day
INJ_TEMP = 310.0        # K (~37C)
PROD_BHP = 150.0        # bar

# Default Compaction Params
ROCK_COMPR = 1e-5
ROCK_COMPR_REF_P = 200.0
ROCK_COMPR_REF_T = 350.0
# Note: Average RCOND is used for the physics constant, though heterogeneity is passed to the grid
AVG_RCOND = 172.8 

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

{C_CYAN} >>>>>>>>> {C_WHITE}[ GEOTHERMAL: 5-LAYER HETEROGENEOUS MODEL ]{C_CYAN} <<<<<<<<<<<< {C_END}
"""

# --- HELPER CLASSES ---
class FixedEnthalpyRegion1(property_evaluator_iface):
    def __init__(self, temp):
        super().__init__()
        self.temperature = temp
    def evaluate(self, state):
        return _Region1(self.temperature, float(state[0]) * 0.1)['h'] * 18.015

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

# --- CUSTOM PHYSICS CLASS ---
class CustomGeothermalPhysics(GeothermalPhysicsBase):
    def __init__(self, timer, n_points=64, min_p=1.0, max_p=1000.0, min_e=1000, max_e=100000):
        super().__init__(timer, n_points, min_p, max_p, min_e, max_e, cache=False)
        
        property_container = GeothermalIAPWSProperties()
        property_container.Mw = [18.015]
        
        # Rock Properties
        property_container.rock = [value_vector([ROCK_COMPR_REF_P, ROCK_COMPR, ROCK_COMPR_REF_T])]
        property_container.rock_compaction_ev = custom_rock_compaction_evaluator(property_container.rock)
        property_container.rock_energy_ev = custom_rock_energy_evaluator(property_container.rock)
        
        # Fluid Properties (IAPWS-97)
        property_container.temperature_ev = iapws_temperature_evaluator()
        property_container.enthalpy_ev = {
            'water': iapws_water_enthalpy_evaluator(),
            'steam': iapws_steam_enthalpy_evaluator(),
            'total': iapws_total_enthalpy_evalutor()
        }
        property_container.density_ev = {
            'water': iapws_water_density_evaluator(),
            'steam': iapws_steam_density_evaluator()
        }
        property_container.viscosity_ev = {
            'water': iapws_water_viscosity_evaluator(),
            'steam': iapws_steam_viscosity_evaluator()
        }
        property_container.saturation_ev = {
            'water': iapws_water_saturation_evaluator(),
            'steam': iapws_steam_saturation_evaluator()
        }
        property_container.relperm_ev = {
            'water': iapws_water_relperm_evaluator(),
            'steam': iapws_steam_relperm_evaluator()
        }
        property_container.conduction_ev = {
            'water': ConstFunc(AVG_RCOND), 
            'steam': ConstFunc(0.0)
        }
        self.add_property_region(property_container)

# --- SIMULATION MODEL ---
class SimulationModel(DartsModel):
    def __init__(self):
        super().__init__()
        self.timer.node["initialization"].start()
        self.nx, self.ny, self.nz = NX, NY, NZ
        
        # 1. GENERATE THE LAYERED PROPERTIES
        poro, permx, permy, permz, hcap, rcond = generate_property_arrays(NX, NY, NZ)

        # 2. PASS ARRAYS INSTEAD OF SCALARS
        self.reservoir = StructReservoir(self.timer, nx=NX, ny=NY, nz=NZ, dx=DX, dy=DY, dz=DZ,
                                         permx=permx, permy=permy, permz=permz,
                                         poro=poro, hcap=hcap, rcond=rcond, 
                                         depth=2000)
        
        self.physics = CustomGeothermalPhysics(self.timer, n_points=100, 
                                               min_p=1.0, max_p=500.0, 
                                               min_e=1000.0, max_e=100000.0) 
        
        self.params.linear_type = sim_params.linear_solver_t.cpu_superlu
        self.set_sim_params(first_ts=1e-3, mult_ts=1.2, max_ts=1.0, runtime=TOTAL_DAYS,
                            tol_newton=1e-3, tol_linear=1e-5)
        
        self.timer.node["initialization"].stop()

    def set_wells(self):
        self.reservoir.add_well("INJ")
        # Perforate all layers for injection
        for k in range(1, NZ+1):
            self.reservoir.add_perforation("INJ", cell_index=(INJ_COORD[0]+1, INJ_COORD[1]+1, k))
        
        self.reservoir.add_well("PROD")
        # Perforate all layers for production
        for k in range(1, NZ+1):
            self.reservoir.add_perforation("PROD", cell_index=(PROD_COORD[0]+1, PROD_COORD[1]+1, k))

    def set_well_controls(self):
        for w in self.reservoir.wells:
            if "INJ" in w.name:
                self.physics.set_well_controls(wctrl=w.control, 
                                               control_type=well_control_iface.VOLUMETRIC_RATE,
                                               is_inj=True, target=INJ_RATE, 
                                               phase_name='water', inj_composition=[], inj_temp=INJ_TEMP) 
            else:
                self.physics.set_well_controls(wctrl=w.control, 
                                               control_type=well_control_iface.BHP,
                                               is_inj=False, target=PROD_BHP)

    def set_initial_conditions(self):
        input_dist = {'pressure': 200.0, 'temperature': 350.0}
        self.physics.set_initial_conditions_from_array(self.reservoir.mesh, input_dist)

# --- VISUALIZATION HELPERS ---
def plot_reservoir_maps(nx, ny, nz, run_dir):
    # Regenerate properties just for plotting
    poro, permx, permy, permz, hcap, rcond = generate_property_arrays(nx, ny, nz)
    
    # Reshape back to 3D for slicing: (NZ, NY, NX)
    poro_3d = poro.reshape(nz, ny, nx)
    perm_3d = permx.reshape(nz, ny, nx)
    
    # SLICE 1: Map View of Layer 2 (The High Permeability Aquifer)
    # Index 1 corresponds to Layer 2 (0-based index)
    layer_idx = 1
    poro_map = poro_3d[layer_idx, :, :]
    perm_map = perm_3d[layer_idx, :, :]
    
    # SLICE 2: Cross Section (XZ) taken at the middle of Y (ny//2)
    mid_y = ny // 2
    poro_xs = poro_3d[:, mid_y, :]
    perm_xs = perm_3d[:, mid_y, :]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # -- Row 1: Map View (Layer 2) --
    # Porosity
    im1 = axes[0,0].imshow(poro_map, origin='lower', cmap='Blues', extent=[0, nx*DX, 0, ny*DY])
    axes[0,0].set_title(f'Map View: Porosity (Layer {layer_idx+1})', fontweight='bold')
    axes[0,0].scatter(INJ_COORD[0]*DX, INJ_COORD[1]*DY, c='c', s=100, edgecolors='k', label='Inj')
    axes[0,0].scatter(PROD_COORD[0]*DX, PROD_COORD[1]*DY, c='g', s=100, marker='^', edgecolors='k', label='Prod')
    fig.colorbar(im1, ax=axes[0,0], fraction=0.046, pad=0.04)
    axes[0,0].legend()
    
    # Permeability
    im2 = axes[0,1].imshow(perm_map, origin='lower', cmap='viridis', extent=[0, nx*DX, 0, ny*DY])
    axes[0,1].set_title(f'Map View: Permeability (Layer {layer_idx+1})', fontweight='bold')
    fig.colorbar(im2, ax=axes[0,1], fraction=0.046, pad=0.04)

    # -- Row 2: Cross Section (XZ) --
    # Note: imshow aspect ratio needs adjustment for Z vs X
    aspect_ratio = (nx * DX) / (nz * DZ) * 0.15 # Exaggerate Z for visibility
    
    # Porosity XS
    im3 = axes[1,0].imshow(poro_xs, origin='upper', cmap='Blues', extent=[0, nx*DX, nz*DZ, 0])
    axes[1,0].set_title(f'Cross Section (Y={mid_y}): Porosity', fontweight='bold')
    axes[1,0].set_xlabel('Distance X (m)'); axes[1,0].set_ylabel('Depth Z (m)')
    axes[1,0].set_aspect('auto') # Fill the plot area
    fig.colorbar(im3, ax=axes[1,0], fraction=0.046, pad=0.04)

    # Perm XS
    im4 = axes[1,1].imshow(perm_xs, origin='upper', cmap='plasma', extent=[0, nx*DX, nz*DZ, 0])
    axes[1,1].set_title(f'Cross Section (Y={mid_y}): Permeability', fontweight='bold')
    axes[1,1].set_xlabel('Distance X (m)')
    axes[1,1].set_aspect('auto')
    fig.colorbar(im4, ax=axes[1,1], fraction=0.046, pad=0.04)

    plt.suptitle('Geological Model: 5-Layer Heterogeneity', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'figures/geothermal_map_thermal.png'), bbox_inches='tight')
    plt.close()

def plot_history(df, run_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    # BHP
    cols = [c for c in df.columns if 'BHP' in c]
    for c in cols: axes[0].plot(df['time'], df[c], '--' if 'INJ' in c else '-', label=c)
    axes[0].set_title('Bottom Hole Pressure'); axes[0].set_ylabel('bar'); axes[0].legend()
    # Rates
    cols = [c for c in df.columns if 'rate' in c and ':' in c]
    for c in cols: axes[1].plot(df['time'], abs(df[c]), '--' if 'INJ' in c else '-', label=c)
    axes[1].set_title('Volumetric Rates'); axes[1].set_ylabel('m3/day'); axes[1].legend()
    # Temperatures
    cols = [c for c in df.columns if 'BHT' in c]
    if not cols: 
        cols = [c for c in df.columns if 'temperature' in c.lower() and ':' in c]
    for c in cols: axes[2].plot(df['time'], df[c], label=c)
    axes[2].set_title('Well Temperatures'); axes[2].set_ylabel('K'); axes[2].legend()
    # Heat Rates
    cols = [c for c in df.columns if 'energy' in c.lower() and ':' in c]
    for c in cols: axes[3].plot(df['time'], abs(df[c]), label=c)
    axes[3].set_title('Heat Rates'); axes[3].set_ylabel('kJ/day'); axes[3].legend()
    for ax in axes: ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'figures/geothermal_well_performance_thermal.png'))
    plt.close()

def export_wells(reservoir, run_dir):
    if not pv: return
    points = [reservoir.discretizer.centroids_all_cells[p[1]] for w in reservoir.wells for p in w.perforations if p[1]>=0]
    points = np.array(points); points[:, 2] -= 2000.0
    merged = pv.MultipleLines(points=points)
    os.makedirs(os.path.join(run_dir, 'vtk_files'), exist_ok=True)
    merged.save(os.path.join(run_dir, 'vtk_files/wells.vtp'))

# --- MAIN ---
def main():
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"output/runs/thermal_final_{ts}"
    os.makedirs(os.path.join(run_dir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'vtk_files'), exist_ok=True)
    
    print(DARTS_BANNER)
    print(f"Run Directory: {run_dir}")
    
    # Pass NZ to plotter
    plot_reservoir_maps(NX, NY, NZ, run_dir)
    
    class DartsSilence:
        def __enter__(self):
            self.stdout_fd, self.stderr_fd = os.dup(1), os.dup(2)
            self.devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(self.devnull, 1); os.dup2(self.devnull, 2)
        def __exit__(self, *args):
            os.dup2(self.stdout_fd, 1); os.dup2(self.stderr_fd, 2)
            os.close(self.stdout_fd); os.close(self.stderr_fd); os.close(self.devnull)

    def print_progress(curr, total):
        pct = (curr / total) * 100
        bar = '=' * int(30 * curr // total) + '-' * (30 - int(30 * curr // total))
        print(f"\r\033[K\033[96mSimulation Progress: [{bar}] {pct:4.1f}% ({curr}/{total} days)\033[0m", end='', flush=True)

    log_file = os.path.join(run_dir, 'simulation.log')
    redirect_darts_output(log_file)
    
    m = SimulationModel()
    with DartsSilence(): m.init()
    m.set_output(os.path.join(run_dir, 'vtk_files'))
    export_wells(m.reservoir, run_dir)
    
    print(f"Starting {TOTAL_DAYS}-day Heterogeneous Geothermal Run...")
    # Add geological properties manually to the output
    poro, permx, permy, permz, hcap, rcond = generate_property_arrays(NX, NY, NZ)
    vars_to_eval = m.physics.vars + ['temperature']
    
    # OUTPUT STEP 0
    with DartsSilence():
        timesteps, property_array = m.output.output_properties(output_properties=vars_to_eval, timestep=0, engine=True)
        # Inject static properties
        for name, arr in [('poro', poro), ('permx', permx), ('permy', permy), ('permz', permz)]:
            property_array[name] = np.array([arr])
        m.output.output_to_vtk(output_data=[timesteps, property_array], ith_step=0)
    
    print_progress(0, TOTAL_DAYS)
    
    for i in range(1, TOTAL_DAYS + 1):
        with DartsSilence():
            m.run(1.0)
            timesteps, property_array = m.output.output_properties(output_properties=vars_to_eval, timestep=i, engine=True)
            # Inject static properties again for each step
            for name, arr in [('poro', poro), ('permx', permx), ('permy', permy), ('permz', permz)]:
                property_array[name] = np.array([arr])
            m.output.output_to_vtk(output_data=[timesteps, property_array], ith_step=i)
        print_progress(i, TOTAL_DAYS)
        
    print(f"\n\n\033[92mSimulation [THERMAL] success!\033[0m")
    df = pd.DataFrame(m.physics.engine.time_data)
    df.to_csv(os.path.join(run_dir, 'history_thermal.csv'))
    plot_history(df, run_dir)
    print(f"Results saved to {run_dir}")

if __name__ == "__main__":
    main()