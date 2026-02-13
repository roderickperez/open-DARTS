# modelDarts3D_isothermal_V1.py - Version 1 (Uniform Properties)
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

# Force software rendering to avoid MESA/ZINK errors in headless environments
os.environ['MESA_LOADER_DRIVER_OVERRIDE'] = 'swrast'
os.environ['GALLIUM_DRIVER'] = 'llvmpipe'
os.environ['DISPLAY'] = ':99.0'

# Suppress PyVista deprecation warnings
from pyvista.plotting.utilities.xvfb import PyVistaDeprecationWarning
warnings.filterwarnings("ignore", category=PyVistaDeprecationWarning)

if pv:
    try:
        # Attempt to start Xvfb if available
        pv.start_xvfb()
    except (OSError, RuntimeError, ImportError):
        # If Xvfb fails, fall back to off-screen rendering
        pv.OFF_SCREEN = True
        print("Xvfb not found or failed to start. Using off-screen rendering fallback.")
else:
    print("Visualization libraries missing. Skipping Xvfb/rendering setup.")

from darts.models.darts_model import DartsModel
from darts.reservoirs.struct_reservoir import StructReservoir
from darts.physics.super.physics import Compositional, PhysicsBase
from darts.physics.super.property_container import PropertyContainer
from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic
from darts.engines import value_vector, redirect_darts_output, well_control_iface, sim_params

# --- SIMULATION CONFIGURATION ---
# --- SIMULATION CONFIGURATION ---
NX, NY, NZ = 100, 100, 5           # Grid resolution (100x100x5 as requested)
DX, DY, DZ = 10.0, 10.0, 10.0       # Cell dimensions (m) - 1km x 1km x 50m model
TOTAL_DAYS = 100                    # Simulation duration
USE_GPU = False                     # Set to True to use NVIDIA GPU acceleration (Requires GPU-enabled package)

# Well Locations (0-based grid indices)
# Well Locations (0-based grid indices)
# User request: (5,5) and (95,95)
INJ_COORD  = [4, 4]    # Injector at (5, 5)
PROD_COORD = [94, 94]  # Producer at (95, 95)

# Well Completions (List of layers [1-10] to open)
# By default, all layers are open.
# Well Completions
# Well Completions
LAYERS_INJ  = range(1, 6)   # Layers perforated for injectors (All 5 layers)
LAYERS_PROD = range(1, 6)   # Layers perforated for producer (All 5 layers)

# Operation Parameters
INJ_BHP  = 300.0   # Injector Bottom Hole Pressure (bar)
PROD_BHP = 150.0   # Producer Bottom Hole Pressure (bar)
INJ_TEMP_COLD = 310.0  # K (Used in Thermal case)
INJ_TEMP_HOT  = 380.0  # K (Used in Thermal case)

# Fluid & Rock Properties
WATER_DENS_0 = 1000.0   # Surface Density (kg/m3)
WATER_COMPR  = 1e-5     # Compressibility (1/bar)
WATER_VISC   = 0.5      # Viscosity (cP)
ROCK_PORO    = 0.25     # Porosity
ROCK_PERM    = 500.0    # Permeability (mD)
# --------------------------------

# ANSI Color Constants for Terminal UI
C_GREEN = '\033[1;92m'
C_CYAN = '\033[1;96m'
C_WHITE = '\033[1;97m'
C_END = '\033[0m'

DARTS_BANNER = f"""
{C_GREEN}       _____          _____ _______  _____ 
      |||  __ \\   /\\   |  __ \\__   __|/ ____|
      |||  ||| /  \\  |||__) | | |  | (___  
      |||  |||/ /\\ \\ |  _  /  | |   \\___ \\ 
      |||__||/ ____ \\| | \\ \\  | |   ____) |
      |||___/_/    \\_\\_|  \\_\\ |_|  |_____/ {C_END}

{C_CYAN} >>>>>>>>> {C_WHITE}[ GEOTHERMAL: ISOTHERMAL - VERSION 1 (UNIFORM) ]{C_CYAN} <<<<<<<<<<<< {C_END}
{C_CYAN}  ___  ____   ___   _____  _   _  _____  ____   __  __    _    _     {C_END}
{C_CYAN} |_ _|| ___| / _ \\ |_   _|| | | || ____||  _ \\ |  \\/  |  / \\  | |    {C_END}
{C_CYAN}  | | \\___ \\| | | |  | |  | |_| ||  _|  | |_) || |\\/| | / _ \\ | |    {C_END}
{C_CYAN}  | |  ___) | |_| |  | |  |  _  || |___ |  _ < | |  | |/ ___ \\| |___ {C_END}
{C_CYAN} |___||____/ \\___/   |_|  |_| |_||_____||_| \\_\\|_|  |_/_/   \\_\\_____|{C_END}
"""

class ModelProperties(PropertyContainer):
    def __init__(self, phases_name, components_name, min_z=1e-11):
        # Call base class constructor
        self.nph = len(phases_name)
        self.nc  = len(components_name)
        Mw = np.array([18.015] * self.nc) # Water molecular weight (kg/kmol)
        # temperature=350.0 means isothermal simulation at this temperature (Surface)
        super().__init__(phases_name=phases_name, components_name=components_name, Mw=Mw, min_z=min_z, temperature=350.0)

    def get_state(self, state):
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]
        # In P spec, state is [pressure, component1, ..., componentN-1]
        zc = np.append(vec_state_as_np[1:], 1 - np.sum(vec_state_as_np[1:]))
        temperature = self.temperature # Constant 350K (Surface)
        return pressure, temperature, zc

    def evaluate(self, state):
        pressure, temperature, zc = self.get_state(state)
        self.clean_arrays()
        
        # All components are in the single liquid water phase
        self.ph = np.array([0], dtype=np.intp)
        for i in range(self.nc):
            self.x[0, i] = zc[i]

        for j in self.ph:
            M = np.sum(self.x[j, :] * self.Mw) + 1e-20
            self.dens[j] = self.density_ev[self.phases_name[j]].evaluate(pressure, temperature)
            self.dens_m[j] = self.dens[j] / M
            self.mu[j] = self.viscosity_ev[self.phases_name[j]].evaluate(pressure, temperature)

            # Define thermal properties for OBL consistency (even if isothermal)
            self.enthalpy[j] = self.enthalpy_ev[self.phases_name[j]].evaluate(pressure, temperature)
            self.cond[j] = self.conductivity_ev[self.phases_name[j]].evaluate(pressure, temperature)

        self.nu = np.array([1.0]) # Single phase saturation is 1.0
        self.sat[0] = 1.0
        for j in self.ph:
            self.kr[j] = 1.0 # Single phase rel-perm is 1.0
            self.pc[j] = 0

    def evaluate_thermal(self, state):
        pressure, temperature, zc = self.get_state(state)
        for j in self.ph:
            self.enthalpy[j] = self.enthalpy_ev[self.phases_name[j]].evaluate(pressure, temperature)
            self.cond[j] = self.conductivity_ev[self.phases_name[j]].evaluate(pressure, temperature)

class SimulationModel(DartsModel):
    def __init__(self):
        super().__init__()
        self.timer.node["initialization"].start()
        
        self.nx, self.ny, self.nz = NX, NY, NZ
        
        # Reservoir definition: 100x100x10 grid
        self.reservoir = StructReservoir(self.timer, nx=NX, ny=NY, nz=NZ, dx=DX, dy=DY, dz=DZ,
                                         permx=ROCK_PERM, permy=ROCK_PERM, permz=ROCK_PERM*0.1, 
                                         poro=ROCK_PORO, hcap=0, rcond=0, depth=2000)
        
        self.set_physics()
        
        
        # SETTINGS (Adaptive solver selection)
        if USE_GPU:
            self.params.linear_type = sim_params.linear_solver_t.gpu_gmres_cpr_amgx_ilu
        else:
            self.params.linear_type = sim_params.linear_solver_t.cpu_superlu

        self.set_sim_params(first_ts=0.005, mult_ts=1.5, max_ts=5.0, runtime=TOTAL_DAYS, tol_newton=1e-3, tol_linear=1e-5)
        
        self.timer.node["initialization"].stop()
        
    def set_physics(self):
        zero = 1e-13
        components = ["H2O_Cold", "H2O_Hot"] # Components used to track hot/cold source
        phases = ["wat"]
        
        self.inj_comp      = value_vector([1.0 - zero]) # Injection composition
        self.ini_comp      = value_vector([0.5])         # Initial is mixture
        
        property_container = ModelProperties(phases_name=phases, components_name=components)
        property_container.density_ev = dict([('wat', DensityBasic(compr=WATER_COMPR, dens0=WATER_DENS_0))])
        property_container.viscosity_ev = dict([('wat', ConstFunc(WATER_VISC))])
        
        # Thermal properties (defined for compatibility)
        property_container.enthalpy_ev = dict([('wat', ConstFunc(100.0))])
        property_container.rock_hcap_ev = ConstFunc(1000.0)
        property_container.conductivity_ev = dict([('wat', ConstFunc(10.0))])
        
        n_points = 50 
        self.physics = Compositional(components, phases, self.timer, n_points=n_points, 
                                     min_p=100, max_p=500, min_z=zero, max_z=1 - zero,
                                     state_spec=PhysicsBase.StateSpecification.P)
        self.physics.add_property_region(property_container)
        
            
    def set_wells(self):
        # Add Injector
        self.reservoir.add_well("INJ")
        for k in LAYERS_INJ:
            self.reservoir.add_perforation("INJ", cell_index=(INJ_COORD[0] + 1, INJ_COORD[1] + 1, k))
            
        # Add Producer
        self.reservoir.add_well("PROD")
        for k in LAYERS_PROD:
            self.reservoir.add_perforation("PROD", cell_index=(PROD_COORD[0] + 1, PROD_COORD[1] + 1, k))
            
    def set_initial_conditions(self):
        # Initial condition with respect to Water-only physics variables
        input_distribution = {self.physics.vars[0]: 200.0,
                              self.physics.vars[1]: self.ini_comp[0]}
        self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh, input_distribution=input_distribution)

    def set_well_controls(self):
        for i, w in enumerate(self.reservoir.wells):
            if "INJ" in w.name:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP, 
                                               is_inj=True, target=INJ_BHP, inj_composition=self.inj_comp)
            else:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP, 
                                               is_inj=False, target=PROD_BHP)

def export_wells_to_vtk(reservoir, output_directory):
    wells_data = []
    for well in reservoir.wells:
        points = []
        for perf in well.perforations:
            res_idx = perf[1]
            if res_idx >= 0:
                coords = reservoir.discretizer.centroids_all_cells[res_idx]
                # Offset Z to match reservoir depth (negative convention)
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
        output_file = os.path.join(output_directory, "wells.vtp")
        combined.save(output_file)
        print(f"\033[93mWell trajectories exported to: {os.path.abspath(output_file)}\033[0m")

def plot_reservoir_properties(nx, ny, run_dir):
    poro_map = np.full((ny, nx), ROCK_PORO)
    permx_map = np.full((ny, nx), ROCK_PERM)
    permz_map = np.full((ny, nx), ROCK_PERM * 0.1)
    depth_map = np.full((ny, nx), 2000.0)

    fig, axes = plt.subplots(1, 4, figsize=(22, 7))
    properties = [
        (poro_map, 'Porosity', 'Blues', 'fraction'),
        (permx_map, 'PermX', 'viridis', 'mD'),
        (permz_map, 'PermZ', 'plasma', 'mD'),
        (depth_map, 'Reservoir Depth', 'terrain', 'meters')
    ]

    for i, (data, name, cmap, unit) in enumerate(properties):
        ax = axes[i]
        im = ax.imshow(data, origin='lower', cmap=cmap, extent=[0, nx-1, 0, ny-1])
        ax.scatter(INJ_COORD[0], INJ_COORD[1], color='cyan', s=150, marker='v', edgecolors='black', label='Injector', zorder=5)
        ax.scatter(PROD_COORD[0], PROD_COORD[1], color='green', s=150, marker='^', edgecolors='black', label='Producer', zorder=5)
        ax.set_title(f"{name}\n({unit})", fontsize=12, fontweight='bold')
        ax.set_xlabel('Grid X')
        if i == 0: 
            ax.set_ylabel('Grid Y')
            ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.15, shrink=0.8)
        cbar.ax.tick_params(labelsize=8)

    plt.suptitle('Geothermal Reservoir - Geological Properties & Well Locations', fontsize=16)
    fig_dir = os.path.join(run_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    save_path = os.path.join(fig_dir, 'geothermal_map_isothermal.png')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\033[96mStatic reservoir maps saved to {save_path}\033[0m")
    plt.close()

def plot_well_history(df_history, run_dir):
    categories = [
        ('Pressure', 'BHP', 'Pressure (bar)', None),
        ('Temperature', 'BHT', 'Temperature (K)', None),
        ('Volumetric Rates', 'rate', 'Rate (m3/day)', None),
        ('Heat Rates', 'energy', 'Heat Rate (kJ/day)', None)
    ]

    found_categories = []
    for name, pattern, ylabel, secondary in categories:
        cols = [c for c in df_history.columns if pattern in c]
        if secondary:
            cols = [c for c in cols if any(s in c.lower() for s in secondary.split('|'))]
        if cols:
            found_categories.append({'name': name, 'cols': cols, 'ylabel': ylabel})

    if found_categories:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        for i, cat in enumerate(found_categories):
            if i >= 4: break
            ax = axes[i]
            for col in cat['cols']:
                plt_style = '--' if 'INJ' in col else '-'
                ax.plot(df_history['time'], abs(df_history[col]), plt_style, label=col)
            ax.set_title(cat['name'], fontweight='bold')
            ax.set_xlabel('Time (days)')
            ax.set_ylabel(cat['ylabel'])
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, linestyle=':', alpha=0.7)
        
        for j in range(len(found_categories), 4):
            axes[j].axis('off')
        
        plt.tight_layout()
        fig_dir = os.path.join(run_dir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)
        save_path = os.path.join(fig_dir, 'geothermal_well_performance_isothermal.png')
        plt.savefig(save_path, bbox_inches='tight')
        print(f"\033[96mWell performance plots saved to {save_path}\033[0m")
        plt.close()

def main():
    print(DARTS_BANNER)
    
    # Setup paths with timestamped run folder
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join('/home/roderickperez/DataScienceProjects/openDARTS/output/runs', f'isothermal_{timestamp}')
    vtk_output_path = os.path.join(run_dir, 'vtk_files')
    os.makedirs(vtk_output_path, exist_ok=True)
    
    print(f"\n\033[94mInitializing 3D Geothermal Isothermal Simulation...\033[0m")
    print(f"\033[94mGrid Size: {NX}x{NY}x{NZ}\033[0m")
    print(f"\033[94mRun Directory: {run_dir}\033[0m")
    
    # Plot initial map in the run directory
    plot_reservoir_properties(NX, NY, run_dir)
    
    # Context manager to silence C++ output from DARTS engine
    class DartsSilence:
        def __enter__(self):
            try:
                self.stdout_fd = os.dup(1)
                self.stderr_fd = os.dup(2)
                self.devnull = os.open(os.devnull, os.O_WRONLY)
                os.dup2(self.devnull, 1)
                os.dup2(self.devnull, 2)
            except Exception:
                pass
        def __exit__(self, exc_type, exc_val, exc_tb):
            try:
                os.dup2(self.stdout_fd, 1)
                os.dup2(self.stderr_fd, 2)
                os.close(self.stdout_fd)
                os.close(self.stderr_fd)
                os.close(self.devnull)
            except Exception:
                pass

    # Simple manual progress bar
    def print_progress(current, total):
        pct = (current / total) * 100
        bar_len = 30
        filled_len = int(bar_len * current // total)
        progress_bar = '=' * filled_len + '-' * (bar_len - filled_len)
        # Clear line and print in Cyan
        print(f"\r\033[K\033[96mSimulation Progress: [{progress_bar}] {pct:4.1f}% ({current}/{total} days)\033[0m", end='', flush=True)

    # Initialize DARTS
    log_file = os.path.join(run_dir, 'simulation_isothermal.log')
    redirect_darts_output(log_file)
    m = SimulationModel()
    print("\033[93mBuilding OBL tables and initializing engine...\033[0m")
    with DartsSilence():
        m.init(platform='gpu' if USE_GPU else 'cpu')
    
    # Configure output folder
    m.set_output(output_folder=vtk_output_path)
    
    # Export wells (shifted to match reservoir depth)
    export_wells_to_vtk(m.reservoir, vtk_output_path)
    
    # Simulation settings
    report_step = 1    # Daily reports for better progress visibility
    steps = TOTAL_DAYS // report_step
    
    print(f"\n\033[92mStarting {TOTAL_DAYS}-day simulation (Daily reporting)...\033[0m")
    
    # Export initial state
    prop_list = m.physics.vars + m.output.properties
    with DartsSilence():
        m.output.output_to_vtk(output_properties=prop_list, ith_step=0)
    print_progress(0, TOTAL_DAYS)
    
    # Simulation loop
    for i in range(1, steps + 1):
        target_time = i * report_step
        with DartsSilence():
            m.run(report_step, verbose=False)
            m.output.output_to_vtk(output_properties=prop_list, ith_step=i)
        print_progress(target_time, TOTAL_DAYS)
        
    print(f"\n\n\033[92mSimulation [ISOTHERMAL] complete. Processing history data...\033[0m")
    
    # Save history and plot
    df_history = pd.DataFrame.from_dict(m.physics.engine.time_data)
    history_csv = os.path.join(run_dir, 'simulation_history_isothermal.csv')
    df_history.to_csv(history_csv, index=False)
    plot_well_history(df_history, run_dir)
    
    print(f"\n\033[95mResults summary:\033[0m")
    print(f"- Run Directory: {run_dir}")
    print(f"- PVD File (ParaView): {os.path.join(vtk_output_path, 'solution.pvd')}")
    print(f"- CSV History: {history_csv}")
    print(f"- Figures: {os.path.join(run_dir, 'figures/')}")
    print("\n\033[1;92m[DONE]\033[0m")

if __name__ == "__main__":
    main()
