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

# --- GEOLOGICAL GENERATOR ---
def generate_layered_properties(nx, ny, nz):
    """
    Generates 3D arrays for rock properties based on a 5-layer geological model.
    """
    # Initialize 3D arrays (flattened order for DARTS: x varies fastest, then y, then z)
    # However, it is easier to build as (NZ, NY, NX) and then flatten.
    
    poro  = np.zeros((nz, ny, nx))
    permx = np.zeros((nz, ny, nx))
    permy = np.zeros((nz, ny, nx))
    permz = np.zeros((nz, ny, nx))
    hcap  = np.zeros((nz, ny, nx))
    rcond = np.zeros((nz, ny, nx))

    # --- DEFINE LAYERS (Top to Bottom) ---
    # Layer 0 (Top): Overburden/Tight Cap
    # Low Poro/Perm. High Thermal Cond (dense rock).
    poro[0,:,:]  = 0.10
    permx[0,:,:] = 50.0
    hcap[0,:,:]  = 2200.0
    rcond[0,:,:] = 180.0

    # Layer 1: MAIN RESERVOIR (High Quality Sandstone)
    # High Poro (25%), High Perm (500 mD).
    poro[1,:,:]  = 0.25
    permx[1,:,:] = 500.0
    hcap[1,:,:]  = 2470.0  # Higher water content usually implies higher effective heat cap
    rcond[1,:,:] = 170.0

    # Layer 2: Transition Zone
    # Medium qualities.
    poro[2,:,:]  = 0.15
    permx[2,:,:] = 150.0
    hcap[2,:,:]  = 2300.0
    rcond[2,:,:] = 175.0

    # Layer 3: FLOW BARRIER / SEAL
    # Very low Perm (1 mD) to baffle flow.
    poro[3,:,:]  = 0.05
    permx[3,:,:] = 1.0     # Effectively a seal
    hcap[3,:,:]  = 2100.0
    rcond[3,:,:] = 200.0   # Dense rock conducts heat well

    # Layer 4 (Bottom): Deep Unit
    # Tight/Deep rock.
    poro[4,:,:]  = 0.10
    permx[4,:,:] = 80.0
    hcap[4,:,:]  = 2200.0
    rcond[4,:,:] = 180.0

    # Isotropy/Anisotropy
    permy = permx.copy()           # X = Y permeability
    permz = permx.copy() * 0.1     # Z is 10% of X (Vertical Anisotropy)

    # Flatten arrays for DARTS (C-order: last index changes fastest is standard for numpy flatten)
    # DARTS StructReservoir usually expects a list/array of size NX*NY*NZ corresponding to blocks.
    # We must ensure the order matches the grid generation. 
    # StructReservoir iterates k, then j, then i usually? 
    # Actually, let's keep it simple: DARTS typically takes a 1D array of length Nx*Ny*Nz.
    # We will pass the flattened arrays.
    
    return {
        'poro': poro.flatten(),
        'permx': permx.flatten(),
        'permy': permy.flatten(),
        'permz': permz.flatten(),
        'hcap': hcap.flatten(),
        'rcond': rcond.flatten()
    }

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
        
        # 1. GENERATE HETEROGENEOUS PROPERTIES
        geo_props = generate_layered_properties(NX, NY, NZ)
        
        # 2. INITIALIZE RESERVOIR WITH ARRAYS
        self.reservoir = StructReservoir(self.timer, nx=NX, ny=NY, nz=NZ, dx=DX, dy=DY, dz=DZ,
                                         permx=geo_props['permx'], 
                                         permy=geo_props['permy'], 
                                         permz=geo_props['permz'],
                                         poro=geo_props['poro'], 
                                         hcap=geo_props['hcap'], 
                                         rcond=geo_props['rcond'], 
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
    # Regenerate properties just for plotting (to avoid passing the object around)
    props = generate_layered_properties(nx, ny, nz)
    
    # Reshape back to 3D for slicing: (NZ, NY, NX)
    # Note: The flatten() was C-order (row-major), so we reshape (NZ, NY, NX)
    poro_3d = props['poro'].reshape(nz, ny, nx)
    perm_3d = props['permx'].reshape(nz, ny, nx)
    
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
    # Add geological properties to output list
    prop_list = m.physics.vars + ['temperature', 'poro', 'permx', 'permy', 'permz']
    with DartsSilence(): m.output.output_to_vtk(ith_step=0, output_properties=prop_list)
    print_progress(0, TOTAL_DAYS)
    
    for i in range(1, TOTAL_DAYS + 1):
        with DartsSilence():
            m.run(1.0)
            m.output.output_to_vtk(ith_step=i, output_properties=prop_list)
        print_progress(i, TOTAL_DAYS)
        
    print(f"\n\n\033[92mSimulation [THERMAL] success!\033[0m")
    df = pd.DataFrame(m.physics.engine.time_data)
    df.to_csv(os.path.join(run_dir, 'history_thermal.csv'))
    plot_history(df, run_dir)
    print(f"Results saved to {run_dir}")

if __name__ == "__main__":
    main()