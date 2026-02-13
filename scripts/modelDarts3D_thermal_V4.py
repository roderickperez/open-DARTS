# modelDarts3D_thermal_V4.py - Version 4 (Stochastic Ensemble)
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# --- VISUALIZATION SETUP ---
try:
    import pyvista as pv
except ImportError:
    pv = None

# Force software rendering for headless environments (consistent for all runs)
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
TOTAL_DAYS = 2
ENSEMBLE_SIZE = 1

# Base Operation Params
INJ_RATE = 500.0        
INJ_TEMP = 310.0        
PROD_BHP = 150.0        

# Compaction Default
ROCK_COMPR = 1e-5
ROCK_COMPR_REF_P = 200.0
ROCK_COMPR_REF_T = 350.0
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

{C_CYAN} >>>>>>>>> {C_WHITE}[ GEOTHERMAL: V4 - STOCHASTIC ENSEMBLE ]{C_CYAN} <<<<<<<<<<<< {C_END}
"""

# --- HELPER CLASSES ---
class FixedEnthalpyRegion1(property_evaluator_iface):
    def __init__(self, temp):
        super().__init__()
        self.temperature = temp
    def evaluate(self, state):
        return _Region1(self.temperature, float(state[0]) * 0.1)['h'] * 18.015

# --- STOCHASTIC GENERATORS (V4) ---
# --- STOCHASTIC GENERATORS (V4) ---
def generate_property_arrays(nx, ny, nz, realization_seed=42):
    """
    Generates Spatially Correlated properties with stochastic parameters.
    Returns: poro, permx, permy, permz, hcap, rcond, layer_stats
    """
    np.random.seed(realization_seed)
    
    # Randomly jitter the target means and sigmas for this realization (+/- 10%)
    def jitter(val): return float(val * np.random.uniform(0.9, 1.1))

    layer_configs = [
        {'m_poro': 0.10, 'm_permx': 200.0,  'sigma': 2.0},
        {'m_poro': 0.25, 'm_permx': 1500.0, 'sigma': 8.0},
        {'m_poro': 0.15, 'm_permx': 500.0,  'sigma': 4.0},
        {'m_poro': 0.05, 'm_permx': 5.0,    'sigma': 1.0},
        {'m_poro': 0.10, 'm_permx': 80.0,   'sigma': 3.0}
    ]
    
    poro_3d  = np.zeros((nz, ny, nx))
    permx_3d = np.zeros((nz, ny, nx))
    permy_3d = np.zeros((nz, ny, nx))
    permz_3d = np.zeros((nz, ny, nx))
    
    layer_stats = []

    for k in range(nz):
        cfg = layer_configs[k]
        noise = np.random.normal(0, 1, (ny, nx))
        
        # Jitter parameters for THIS specific realization and layer
        m_poro = jitter(cfg['m_poro'])
        m_permx = jitter(cfg['m_permx'])
        sigma = jitter(cfg['sigma'])
        
        # Anisotropy: PermY = PermX * jittered_factor, PermZ = PermX * 0.1 * jittered_factor
        m_permy = m_permx * np.random.uniform(0.8, 1.2)
        m_permz = m_permx * 0.1 * np.random.uniform(0.5, 1.5)
        
        smoothed = gaussian_filter(noise, sigma=sigma)
        
        poro_layer = m_poro + (smoothed * (m_poro * 0.2))
        poro_layer = np.clip(poro_layer, 0.01, 0.45)
        
        # Log-linear perm from smoothed field
        log_perm = np.log10(m_permx) + smoothed * 0.5 
        px = 10**log_perm
        py = px * (m_permy / m_permx)
        pz = px * (m_permz / m_permx)
        
        poro_3d[k,:,:]  = poro_layer
        permx_3d[k,:,:] = px
        permy_3d[k,:,:] = py
        permz_3d[k,:,:] = pz
        
        layer_stats.append({
            'layer': k + 1,
            'm_poro': m_poro,
            'm_permx': m_permx,
            'm_permy': m_permy,
            'm_permz': m_permz,
            'sigma': sigma
        })

    hcap_val = 2400.0 
    rcond_val = 170.0

    return (poro_3d.flatten().astype(np.float64), 
            permx_3d.flatten().astype(np.float64), 
            permy_3d.flatten().astype(np.float64), 
            permz_3d.flatten().astype(np.float64), 
            np.full(nx*ny*nz, hcap_val, dtype=np.float64), 
            np.full(nx*ny*nz, rcond_val, dtype=np.float64), 
            layer_stats)

# --- CUSTOM PHYSICS CLASS ---
# --- CUSTOM PHYSICS CLASS ---
class CustomGeothermalPhysics(GeothermalPhysicsBase):
    def __init__(self, timer, n_points, min_p, max_p, min_e, max_e, 
                 rock_compr, rock_ref_p, rock_ref_t, avg_rcond):
        super().__init__(timer, n_points, min_p, max_p, min_e, max_e, cache=False)
        
        property_container = GeothermalIAPWSProperties()
        property_container.Mw = [18.015]
        
        # Rock Properties (Randomized per realization)
        property_container.rock = [value_vector([rock_ref_p, rock_compr, rock_ref_t])]
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
            'water': ConstFunc(avg_rcond), 
            'steam': ConstFunc(0.0)
        }
        self.add_property_region(property_container)

# --- SIMULATION MODEL ---
class SimulationModel(DartsModel):
    def __init__(self, nx, ny, nz, poro, permx, permy, permz, hcap, rcond, 
                 inj_coord, prod_coord, perf_layers_inj, perf_layers_prod,
                 op_params, rock_params):
        super().__init__()
        self.timer.node["initialization"].start()
        self.nx, self.ny, self.nz = nx, ny, nz
        self.inj_coord = inj_coord
        self.prod_coord = prod_coord
        self.perf_layers_inj = perf_layers_inj
        self.perf_layers_prod = perf_layers_prod
        self.op_params = op_params
        
        # Initialize reservoir with passed arrays
        self.reservoir = StructReservoir(self.timer, nx=nx, ny=ny, nz=nz, dx=DX, dy=DY, dz=DZ,
                                         permx=permx, permy=permy, permz=permz,
                                         poro=poro, hcap=hcap, rcond=rcond, 
                                         depth=2000)
        
        self.physics = CustomGeothermalPhysics(self.timer, n_points=100, 
                                               min_p=1.0, max_p=500.0, 
                                               min_e=1000.0, max_e=100000.0,
                                               rock_compr=rock_params['rock_compr'],
                                               rock_ref_p=rock_params['rock_ref_p'],
                                               rock_ref_t=rock_params['rock_ref_t'],
                                               avg_rcond=rock_params['avg_rcond']) 
        
        self.params.linear_type = sim_params.linear_solver_t.cpu_superlu
        self.set_sim_params(first_ts=1e-3, mult_ts=1.2, max_ts=1.0, runtime=TOTAL_DAYS,
                            tol_newton=1e-3, tol_linear=1e-5)
        
        self.timer.node["initialization"].stop()

    def set_wells(self):
        self.reservoir.add_well("INJ")
        for k in self.perf_layers_inj:
            self.reservoir.add_perforation("INJ", cell_index=(self.inj_coord[0]+1, self.inj_coord[1]+1, k))
        
        self.reservoir.add_well("PROD")
        for k in self.perf_layers_prod:
            self.reservoir.add_perforation("PROD", cell_index=(self.prod_coord[0]+1, self.prod_coord[1]+1, k))

    def set_well_controls(self):
        for w in self.reservoir.wells:
            if "INJ" in w.name:
                self.physics.set_well_controls(wctrl=w.control, 
                                               control_type=well_control_iface.VOLUMETRIC_RATE,
                                               is_inj=True, target=self.op_params['inj_rate'], 
                                               phase_name='water', inj_composition=[], 
                                               inj_temp=self.op_params['inj_temp']) 
            else:
                self.physics.set_well_controls(wctrl=w.control, 
                                               control_type=well_control_iface.BHP,
                                               is_inj=False, target=self.op_params['prod_bhp'])

    def set_initial_conditions(self):
        input_dist = {'pressure': 200.0, 'temperature': 350.0}
        self.physics.set_initial_conditions_from_array(self.reservoir.mesh, input_dist)

# --- VISUALIZATION HELPERS ---
def plot_reservoir_maps(nx, ny, nz, run_dir, poro, permx, inj_coord, prod_coord):
    poro_3d = poro.reshape(nz, ny, nx)
    perm_3d = permx.reshape(nz, ny, nx)
    
    layer_idx = 1
    poro_map = poro_3d[layer_idx, :, :]
    perm_map = perm_3d[layer_idx, :, :]
    mid_y = ny // 2
    poro_xs = poro_3d[:, mid_y, :]
    perm_xs = perm_3d[:, mid_y, :]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    im1 = axes[0,0].imshow(poro_map, origin='lower', cmap='Blues', extent=[0, nx*DX, 0, ny*DY])
    axes[0,0].set_title(f'Map View: Porosity (Layer {layer_idx+1})', fontweight='bold')
    axes[0,0].scatter(inj_coord[0]*DX, inj_coord[1]*DY, c='c', s=100, edgecolors='k', label='Inj')
    axes[0,0].scatter(prod_coord[0]*DX, prod_coord[1]*DY, c='g', s=100, marker='^', edgecolors='k', label='Prod')
    fig.colorbar(im1, ax=axes[0,0], fraction=0.046, pad=0.04)
    axes[0,0].legend()
    
    im2 = axes[0,1].imshow(perm_map, origin='lower', cmap='viridis', extent=[0, nx*DX, 0, ny*DY])
    axes[0,1].set_title(f'Map View: Permeability (Layer {layer_idx+1})', fontweight='bold')
    fig.colorbar(im2, ax=axes[0,1], fraction=0.046, pad=0.04)

    im3 = axes[1,0].imshow(poro_xs, origin='upper', cmap='Blues', extent=[0, nx*DX, nz*DZ, 0])
    axes[1,0].set_title(f'Cross Section (Y={mid_y}): Porosity', fontweight='bold')
    axes[1,0].set_xlabel('Distance X (m)'); axes[1,0].set_ylabel('Depth Z (m)')
    axes[1,0].set_aspect('auto')
    fig.colorbar(im3, ax=axes[1,0], fraction=0.046, pad=0.04)

    im4 = axes[1,1].imshow(perm_xs, origin='upper', cmap='plasma', extent=[0, nx*DX, nz*DZ, 0])
    axes[1,1].set_title(f'Cross Section (Y={mid_y}): Permeability', fontweight='bold')
    axes[1,1].set_xlabel('Distance X (m)')
    axes[1,1].set_aspect('auto')
    fig.colorbar(im4, ax=axes[1,1], fraction=0.046, pad=0.04)

    plt.suptitle('Geological Model: Stochastic Ensemble V4', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'figures/geothermal_map_thermal_V4.png'), bbox_inches='tight')
    plt.close()

def plot_history(df, run_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    cols = [c for c in df.columns if 'BHP' in c]
    for c in cols: axes[0].plot(df['time'], df[c], '--' if 'INJ' in c else '-', label=c)
    axes[0].set_title('Bottom Hole Pressure'); axes[0].set_ylabel('bar'); axes[0].legend()
    cols = [c for c in df.columns if 'rate' in c and ':' in c]
    for c in cols: axes[1].plot(df['time'], abs(df[c]), '--' if 'INJ' in c else '-', label=c)
    axes[1].set_title('Volumetric Rates'); axes[1].set_ylabel('m3/day'); axes[1].legend()
    cols = [c for c in df.columns if 'BHT' in c]
    if not cols: 
        cols = [c for c in df.columns if 'temperature' in c.lower() and ':' in c]
    for c in cols: axes[2].plot(df['time'], df[c], label=c)
    axes[2].set_title('Well Temperatures'); axes[2].set_ylabel('K'); axes[2].legend()
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
    import time
    from datetime import datetime
    
    ensemble_root = f"output/ensemble_thermal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(ensemble_root, exist_ok=True)
    
    print(DARTS_BANNER)
    print(f"Starting Ensemble Simulation: {ENSEMBLE_SIZE} realizations")
    print(f"Results Root: {ensemble_root}\n")
    
    for i in range(ENSEMBLE_SIZE):
        seed = 42 + i
        np.random.seed(seed)
        
        # 1. Randomize Operational & Rock Parameters
        def jitter(val): return float(val * np.random.uniform(0.9, 1.1))
        
        op_params = {
            'inj_rate': jitter(INJ_RATE),
            'inj_temp': float(np.random.uniform(300.0, 340.0)),
            'prod_bhp': float(np.random.uniform(50.0, 200.0))
        }
        rock_params = {
            'rock_compr': float(ROCK_COMPR * np.random.uniform(0.5, 2.0)),
            'rock_ref_p': float(ROCK_COMPR_REF_P * np.random.uniform(0.8, 1.2)),
            'rock_ref_t': float(ROCK_COMPR_REF_T * np.random.uniform(0.9, 1.1)),
            'avg_rcond': float(AVG_RCOND * np.random.uniform(0.8, 1.2))
        }

        # 2. Randomize Well Locations (dist >= 20)
        max_attempts = 100
        for _ in range(max_attempts):
            inj_coord = [int(np.random.randint(5, 95)), int(np.random.randint(5, 95))]
            prod_coord = [int(np.random.randint(5, 95)), int(np.random.randint(5, 95))]
            dist = np.sqrt((inj_coord[0]-prod_coord[0])**2 + (inj_coord[1]-prod_coord[1])**2)
            if dist >= 20.0: break
        
        # 3. Randomize Perforations
        possible_layers = list(range(1, NZ + 1))
        perf_inj = [int(k) for k in sorted(np.random.choice(possible_layers, size=np.random.randint(2, 6), replace=False))]
        perf_prod = [int(k) for k in sorted(np.random.choice(possible_layers, size=np.random.randint(2, 6), replace=False))]
        
        # 4. Randomize Geological Properties
        poro, permx, permy, permz, hcap, rcond, layer_stats = generate_property_arrays(NX, NY, NZ, realization_seed=seed)
        
        # Expanded Metadata Logging
        for entry in layer_stats:
            entry.update({
                'inj_x': inj_coord[0], 'inj_y': inj_coord[1],
                'prod_x': prod_coord[0], 'prod_y': prod_coord[1],
                'inj_rate': op_params['inj_rate'], 'inj_temp': op_params['inj_temp'],
                'prod_bhp': op_params['prod_bhp'], 'rock_compr': rock_params['rock_compr'],
                'avg_rcond': rock_params['avg_rcond']
            })
        
        real_dir = os.path.join(ensemble_root, f"realization_{i}")
        os.makedirs(os.path.join(real_dir, 'figures'), exist_ok=True)
        os.makedirs(os.path.join(real_dir, 'vtk_files'), exist_ok=True)
        
        pd.DataFrame(layer_stats).to_csv(os.path.join(real_dir, 'geological_params.csv'), index=False)
        
        print(f"Realization {i}: Wells at {inj_coord} & {prod_coord} (Dist: {dist:.1f})")
        
        plot_reservoir_maps(NX, NY, NZ, real_dir, poro, permx, inj_coord, prod_coord)

        def print_progress(curr, total, real_idx):
            pct = (curr / total) * 100
            bar = '█' * int(20 * curr // total) + '░' * (20 - int(20 * curr // total))
            print(f"\r\033[K\033[96mRealization {real_idx}: {bar} {pct:4.1f}% ({curr}/{total} days)\033[0m", end='', flush=True)

        redirect_darts_output(os.path.join(real_dir, 'simulation.log'))
        
        m = SimulationModel(NX, NY, NZ, poro, permx, permy, permz, hcap, rcond, 
                            inj_coord, prod_coord, perf_inj, perf_prod, op_params, rock_params)
        m.init()
        m.set_output(os.path.join(real_dir, 'vtk_files'))
        export_wells(m.reservoir, real_dir)
        
        vars_to_eval = m.physics.vars + ['temperature']
        timesteps, property_array = m.output.output_properties(output_properties=vars_to_eval, timestep=0, engine=True)
        for name, arr in [('poro', poro), ('permx', permx), ('permy', permy), ('permz', permz)]:
            property_array[name] = np.array([arr])
        m.output.output_to_vtk(output_data=[timesteps, property_array], ith_step=0)
        
        print_progress(0, TOTAL_DAYS, i)
        for d in range(1, TOTAL_DAYS + 1):
            m.run(1.0)
            if d % 10 == 0 or d == TOTAL_DAYS:
                timesteps, property_array = m.output.output_properties(output_properties=vars_to_eval, timestep=d, engine=True)
                for name, arr in [('poro', poro), ('permx', permx), ('permy', permy), ('permz', permz)]:
                    property_array[name] = np.array([arr])
                m.output.output_to_vtk(output_data=[timesteps, property_array], ith_step=d)
            print_progress(d, TOTAL_DAYS, i)
        
        print(f"\nRealization {i} complete.")
        df = pd.DataFrame(m.physics.engine.time_data)
        df.to_csv(os.path.join(real_dir, 'history_thermal.csv'))
        plot_history(df, real_dir)
        del m
        
    print(f"\n\033[92mEnsemble simulation success!\033[0m")
    print(f"All results saved in: {ensemble_root}")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()