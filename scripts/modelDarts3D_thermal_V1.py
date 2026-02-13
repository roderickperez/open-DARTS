# modelDarts3D_thermal_V1.py - Version 1 (Uniform Properties)
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
NX, NY, NZ = 100, 100, 5    # Matches isothermal
DX, DY, DZ = 10.0, 10.0, 10.0 
TOTAL_DAYS = 3650 # 10 years 365 * 10

# Well Locations (0-based) - DIAGONAL (Matches isothermal)
INJ_COORD  = [4, 4]     # (5, 5)
PROD_COORD = [94, 94]   # (95, 95)

# Operation
INJ_RATE = 500.0        # m3/day
INJ_TEMP = 310.0        # K (~37C) - Cold injection for thermal front
PROD_BHP = 150.0        # bar (Matches isothermal)

# Rock Properties (Based on Guia_DARTS.pdf)
ROCK_PORO = 0.25
ROCK_PERM = 500.0
ROCK_HCAP = 2470.0      # kJ/(m3*K)
ROCK_RCOND = 172.8      # kJ/(m*day*K)
ROCK_COMPR = 1e-5
ROCK_COMPR_REF_P = 200.0
ROCK_COMPR_REF_T = 350.0

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

{C_CYAN} >>>>>>>>> {C_WHITE}[ GEOTHERMAL: THERMAL (IAPWS) - VERSION 1 (UNIFORM) ]{C_CYAN} <<<<<<<<<<<< {C_END}
{C_CYAN}  _____  _   _  _____  ____   __  __    _    _      {C_END}
{C_CYAN} |_   _|| | | || ____||  _ \\ |  \\/  |  / \\  | |     {C_END}
{C_CYAN}   | |  | |_| ||  _|  | |_) || |\\/| | / _ \\ | |     {C_END}
{C_CYAN}   | |  |  _  || |___ |  _ < | |  | |/ ___ \\| |___  {C_END}
{C_CYAN}   |_|  |_| |_||_____||_| \\_\\|_|  |_/_/   \\_\\_____| {C_END}
"""

# --- HELPER CLASSES ---
class FixedEnthalpyRegion1(property_evaluator_iface):
    def __init__(self, temp):
        super().__init__()
        self.temperature = temp
    def evaluate(self, state):
        return _Region1(self.temperature, float(state[0]) * 0.1)['h'] * 18.015

# --- CUSTOM PHYSICS CLASS (Stripped of idata dependency) ---
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
            'water': ConstFunc(ROCK_RCOND), 
            'steam': ConstFunc(0.0)
        }
        self.add_property_region(property_container)

# --- SIMULATION MODEL ---
class SimulationModel(DartsModel):
    def __init__(self):
        super().__init__()
        self.timer.node["initialization"].start()
        self.nx, self.ny, self.nz = NX, NY, NZ
        
        self.reservoir = StructReservoir(self.timer, nx=NX, ny=NY, nz=NZ, dx=DX, dy=DY, dz=DZ,
                                         permx=ROCK_PERM, permy=ROCK_PERM, permz=ROCK_PERM*0.1,
                                         poro=ROCK_PORO, hcap=ROCK_HCAP, rcond=ROCK_RCOND, depth=2000)
        
        self.physics = CustomGeothermalPhysics(self.timer, n_points=100, 
                                               min_p=1.0, max_p=500.0, 
                                               min_e=1000.0, max_e=100000.0) 
        
        self.params.linear_type = sim_params.linear_solver_t.cpu_superlu
        self.set_sim_params(first_ts=1e-3, mult_ts=1.2, max_ts=1.0, runtime=TOTAL_DAYS,
                            tol_newton=1e-3, tol_linear=1e-5)
        
        self.timer.node["initialization"].stop()

    def set_wells(self):
        self.reservoir.add_well("INJ")
        for k in range(1, NZ+1):
            self.reservoir.add_perforation("INJ", cell_index=(INJ_COORD[0]+1, INJ_COORD[1]+1, k))
        self.reservoir.add_well("PROD")
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
def plot_reservoir_maps(nx, ny, run_dir):
    poro_map = np.full((ny, nx), ROCK_PORO)
    perm_map = np.full((ny, nx), ROCK_PERM)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    maps = [(poro_map, 'Porosity', 'Blues'), (perm_map, 'Permeability (mD)', 'viridis')]
    for i, (data, name, cmap) in enumerate(maps):
        ax = axes[i]
        im = ax.imshow(data, origin='lower', cmap=cmap, extent=[0, nx-1, 0, ny-1])
        ax.scatter(INJ_COORD[0], INJ_COORD[1], color='cyan', s=150, marker='v', edgecolors='black', label='Injector')
        ax.scatter(PROD_COORD[0], PROD_COORD[1], color='green', s=150, marker='^', edgecolors='black', label='Producer')
        ax.set_title(name, fontweight='bold'); ax.legend()
        fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.2)
    plt.suptitle('Thermal Model Configuration (Diagonal Wells)', fontsize=14)
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
    if not cols: # Fallback to looking for temperature in well results if engine name differs
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
    # Native centroids might need shift
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
    
    plot_reservoir_maps(NX, NY, run_dir)
    
    class DartsSilence:
        def __enter__(self):
            self.stdout_fd, self.stderr_fd = os.dup(1), os.dup(2)
            self.devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(self.devnull, 1); os.dup2(self.devnull, 2)
        def __exit__(self, *_):
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
    
    print(f"Starting {TOTAL_DAYS}-day Peak Geothermal Run...")
    prop_list = m.physics.vars + ['temperature']
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
