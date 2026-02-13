# modelDarts3D_isothermal_V4.py - Version 4 (Stochastic Ensemble)
import os
import warnings
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
try:
    import pyvista as pv
except ImportError:
    pv = None
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
ENSEMBLE_SIZE = 10

# Base Operation Parameters
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

{C_CYAN} >>>>>>>>> {C_WHITE}[ GEOTHERMAL: ISOTHERMAL - V4 (STOCHASTIC) ]{C_CYAN} <<<<<<<<<<<< {C_END}
"""

# --- STOCHASTIC GENERATORS (V4) ---
def generate_property_arrays(nx, ny, nz, realization_seed=42):
    """
    Generates Spatially Correlated properties with stochastic parameters.
    """
    np.random.seed(realization_seed)
    def jitter(val): return val * np.random.uniform(0.9, 1.1)
    
    layer_data = [
        {'m_poro': jitter(0.10), 'm_perm': jitter(200.0),  'sigma': jitter(2.0)},
        {'m_poro': jitter(0.25), 'm_perm': jitter(1500.0), 'sigma': jitter(8.0)},
        {'m_poro': jitter(0.15), 'm_perm': jitter(500.0),  'sigma': jitter(4.0)},
        {'m_poro': jitter(0.05), 'm_perm': jitter(5.0),    'sigma': jitter(1.0)},
        {'m_poro': jitter(0.10), 'm_perm': jitter(80.0),   'sigma': jitter(3.0)}
    ]
    
    poro_3d  = np.zeros((nz, ny, nx))
    permx_3d = np.zeros((nz, ny, nx))

    for k in range(nz):
        data = layer_data[k]
        noise = np.random.normal(0, 1, (ny, nx))
        smoothed = gaussian_filter(noise, sigma=data['sigma'])
        
        poro_layer = data['m_poro'] + (smoothed * (data['m_poro'] * 0.2))
        poro_layer = np.clip(poro_layer, 0.01, 0.45)
        
        log_perm = np.log10(data['m_perm']) + smoothed * 0.5 
        perm_layer = 10**log_perm
        
        poro_3d[k,:,:]  = poro_layer
        permx_3d[k,:,:] = perm_layer

    return (poro_3d.flatten(), permx_3d.flatten(), permx_3d.flatten(), 
            permx_3d.flatten() * 0.1, np.full(nx*ny*nz, 2400.0), np.full(nx*ny*nz, 170.0), layer_data)

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
        
# --- SIMULATION MODEL ---
class SimulationModel(DartsModel):
    def __init__(self, nx, ny, nz, poro, permx, permy, permz, hcap, rcond,
                 inj_coord, prod_coord, perf_layers_inj, perf_layers_prod):
        super().__init__()
        self.timer.node["initialization"].start()
        self.nx, self.ny, self.nz = nx, ny, nz
        self.inj_coord = inj_coord
        self.prod_coord = prod_coord
        self.perf_layers_inj = perf_layers_inj
        self.perf_layers_prod = perf_layers_prod
        
        # Initialize reservoir with passed arrays
        self.reservoir = StructReservoir(self.timer, nx=nx, ny=ny, nz=nz, dx=DX, dy=DY, dz=DZ,
                                         permx=permx, permy=permy, permz=permz,
                                         poro=poro, hcap=hcap, rcond=rcond, 
                                         depth=2000)
        
        self.physics = CustomPhysics(self.timer, n_points=100, min_p=1.0, max_p=500.0)
        
        self.params.linear_type = sim_params.linear_solver_t.cpu_superlu
        self.set_sim_params(first_ts=1e-3, mult_ts=1.2, max_ts=1.0, runtime=TOTAL_DAYS,
                            tol_newton=1e-3, tol_linear=1e-4)
        
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
                self.physics.set_well_controls(w.control, well_control_iface.BHP, INJ_BHP)
            else:
                self.physics.set_well_controls(w.control, well_control_iface.BHP, PROD_BHP)

    def set_initial_conditions(self):
        self.physics.set_initial_conditions(self.reservoir.mesh, [200.0])

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
        if len(wells_data) > 1: combined = wells_data[0].merge(wells_data[1:])
        os.makedirs(output_directory, exist_ok=True)
        combined.save(os.path.join(output_directory, "wells_isothermal.vtp"))

def plot_reservoir_maps(nx, ny, nz, run_dir, poro, permx, inj_coord, prod_coord):
    poro_3d = poro.reshape(nz, ny, nx)
    perm_3d = permx.reshape(nz, ny, nx)
    layer_idx = 1
    poro_map, perm_map = poro_3d[layer_idx, :, :], perm_3d[layer_idx, :, :]
    mid_y = ny // 2
    poro_xs, perm_xs = poro_3d[:, mid_y, :], perm_3d[:, mid_y, :]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    im1 = axes[0,0].imshow(poro_map, origin='lower', cmap='Blues', extent=[0, nx*DX, 0, ny*DY])
    axes[0,0].scatter(inj_coord[0]*DX, inj_coord[1]*DY, c='c', s=100, edgecolors='k', label='Inj')
    axes[0,0].scatter(prod_coord[0]*DX, prod_coord[1]*DY, c='g', s=100, marker='^', edgecolors='k', label='Prod')
    axes[0,0].set_title(f'Map View: Porosity (Layer {layer_idx+1})'); fig.colorbar(im1, ax=axes[0,0]); axes[0,0].legend()
    
    im2 = axes[0,1].imshow(perm_map, origin='lower', cmap='viridis', extent=[0, nx*DX, 0, ny*DY])
    axes[0,1].set_title(f'Map View: Permeability (Layer {layer_idx+1})'); fig.colorbar(im2, ax=axes[0,1])
    
    im3 = axes[1,0].imshow(poro_xs, origin='upper', cmap='Blues', extent=[0, nx*DX, nz*DZ, 0])
    axes[1,0].set_title(f'Cross Section: Porosity'); axes[1,0].set_aspect('auto'); fig.colorbar(im3, ax=axes[1,0])
    
    im4 = axes[1,1].imshow(perm_xs, origin='upper', cmap='plasma', extent=[0, nx*DX, nz*DZ, 0])
    axes[1,1].set_title(f'Cross Section: Permeability'); axes[1,1].set_aspect('auto'); fig.colorbar(im4, ax=axes[1,1])
    
    plt.suptitle('Geological Model: Stochastic Ensemble (Isothermal V4)', fontsize=16)
    os.makedirs(os.path.join(run_dir, 'figures'), exist_ok=True)
    plt.savefig(os.path.join(run_dir, 'figures/geothermal_map_isothermal_V4.png'), bbox_inches='tight')
    plt.close()

def main():
    from datetime import datetime
    
    ensemble_root = f"output/ensemble_isothermal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(ensemble_root, exist_ok=True)
    
    print(DARTS_BANNER)
    print(f"Starting Isothermal Ensemble: {ENSEMBLE_SIZE} realizations")
    print(f"Results Root: {ensemble_root}\n")
    
    for i in range(ENSEMBLE_SIZE):
        seed = 100 + i
        np.random.seed(seed)
        
        real_dir = os.path.join(ensemble_root, f"realization_{i}")
        os.makedirs(os.path.join(real_dir, 'figures'), exist_ok=True)
        os.makedirs(os.path.join(real_dir, 'vtk_files'), exist_ok=True)
        
        # 1. Randomize Properties
        poro, permx, permy, permz, hcap, rcond, layer_metadata = generate_property_arrays(NX, NY, NZ, seed)
        pd.DataFrame(layer_metadata).to_csv(os.path.join(real_dir, 'geological_params.csv'), index=False)
        
        # 2. Randomize Wells/Perfs
        inj_coord = [4 + np.random.randint(-2, 10), 4 + np.random.randint(-2, 10)]
        prod_coord = [94 - np.random.randint(0, 10), 94 - np.random.randint(0, 10)]
        perf_inj = sorted(np.random.choice(range(1, NZ+1), size=np.random.randint(2, 6), replace=False))
        perf_prod = sorted(np.random.choice(range(1, NZ+1), size=np.random.randint(2, 6), replace=False))
        
        print(f"Realization {i}: Wells at {inj_coord} (perfs {perf_inj}) and {prod_coord} (perfs {perf_prod})")
        plot_reservoir_maps(NX, NY, NZ, real_dir, poro, permx, inj_coord, prod_coord)

        class DartsSilence:
            def __enter__(self):
                self.stdout_fd, self.stderr_fd = os.dup(1), os.dup(2)
                self.devnull = os.open(os.devnull, os.O_WRONLY)
                os.dup2(self.devnull, 1); os.dup2(self.devnull, 2)
            def __exit__(self, *args):
                os.dup2(self.stdout_fd, 1); os.dup2(self.stderr_fd, 2)
                os.close(self.stdout_fd); os.close(self.stderr_fd); os.close(self.devnull)

        def print_progress(curr, total, real_idx):
            pct = (curr / total) * 100
            bar = '=' * int(20 * curr // total) + '-' * (20 - int(20 * curr // total))
            print(f"\r\033[K\033[96mRealization {real_idx}: [{bar}] {pct:4.1f}% ({curr}/{total} days)\033[0m", end='', flush=True)

        redirect_darts_output(os.path.join(real_dir, 'simulation.log'))
        
        m = SimulationModel(NX, NY, NZ, poro, permx, permy, permz, hcap, rcond, 
                            inj_coord, prod_coord, perf_inj, perf_prod)
        with DartsSilence(): m.init()
        vtk_path = os.path.join(real_dir, 'vtk_files')
        m.set_output(vtk_path)
        export_wells_to_vtk(m.reservoir, vtk_path)
        
        vars_to_eval = m.physics.vars
        timesteps, property_array = m.output.output_properties(output_properties=vars_to_eval, timestep=0, engine=True)
        for name, arr in [('poro', poro), ('permx', permx), ('permy', permy), ('permz', permz)]:
            property_array[name] = np.array([arr])
        m.output.output_to_vtk(output_data=[timesteps, property_array], ith_step=0)
        
        for d in range(1, TOTAL_DAYS + 1):
            with DartsSilence():
                m.run(1.0)
                if d % 20 == 0 or d == TOTAL_DAYS:
                    timesteps, property_array = m.output.output_properties(output_properties=vars_to_eval, timestep=d, engine=True)
                    for name, arr in [('poro', poro), ('permx', permx), ('permy', permy), ('permz', permz)]:
                        property_array[name] = np.array([arr])
                    m.output.output_to_vtk(output_data=[timesteps, property_array], ith_step=d)
            print_progress(d, TOTAL_DAYS, i)
        
        print(f"Realization {i} complete.")
        pd.DataFrame(m.physics.engine.time_data).to_csv(os.path.join(real_dir, 'history_isothermal.csv'))
        
    print(f"\n\033[92mSimulation [ISOTHERMAL V4 ENSEMBLE] success!\033[0m")
    print(f"All results saved in: {ensemble_root}")

if __name__ == "__main__":
    main()
