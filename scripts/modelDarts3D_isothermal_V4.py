# modelDarts3D_isothermal_V4.py - Version 4 (Stochastic Ensemble)
import os
import warnings
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

# Silence DARTS specific solver warning
warnings.filterwarnings("ignore", message=".*number of cells looks too big to use a direct linear solver.*")
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
REPORT_FREQ = 10

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
    Returns: poro, permx, permy, permz, hcap, rcond, layer_stats
    """
    np.random.seed(realization_seed)
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
        
        m_poro = jitter(cfg['m_poro'])
        m_permx = jitter(cfg['m_permx'])
        sigma = jitter(cfg['sigma'])
        
        m_permy = m_permx * np.random.uniform(0.8, 1.2)
        m_permz = m_permx * 0.1 * np.random.uniform(0.5, 1.5)
        
        smoothed = gaussian_filter(noise, sigma=sigma)
        
        poro_layer = m_poro + (smoothed * (m_poro * 0.2))
        poro_layer = np.clip(poro_layer, 0.01, 0.45)
        
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

    return (poro_3d.flatten().astype(np.float64), 
            permx_3d.flatten().astype(np.float64), 
            permy_3d.flatten().astype(np.float64), 
            permz_3d.flatten().astype(np.float64), 
            np.full(nx*ny*nz, 2400.0, dtype=np.float64), 
            np.full(nx*ny*nz, 170.0, dtype=np.float64), 
            layer_stats)

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
                 inj_coord, prod_coord, perf_layers_inj, perf_layers_prod,
                 op_params):
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
        
        self.physics = CustomPhysics(self.timer, n_points=100, min_p=1.0, max_p=500.0,
                                     compr=op_params['water_compr'], viscos=op_params['water_visc'])
        
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
                self.physics.set_well_controls(w.control, well_control_iface.BHP, True, self.op_params['inj_bhp'],
                                               inj_composition=[1.0 - 1e-12])
            else:
                self.physics.set_well_controls(w.control, well_control_iface.BHP, False, self.op_params['prod_bhp'])

    def set_initial_conditions(self):
        self.physics.set_initial_conditions_from_array(self.reservoir.mesh, 
                                                       {'pressure': 200.0, 'H2O_Cold': 0.5})

class CustomPropertyContainer(PropertyContainer):
    def __init__(self, phases_name, components_name, compr, viscos):
        mw = np.array([18.015] * len(components_name))
        super().__init__(phases_name, components_name, Mw=mw)
        self.density_ev = {'water': DensityBasic(dens0=WATER_DENS_0, compr=compr)}
        self.viscosity_ev = {'water': ConstFunc(viscos)}
        self.rel_perm_ev = {'water': ConstFunc(1.0)}
        self.temperature = 350.0

    def evaluate(self, state):
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]
        # In P-z state, zc is state[1:]
        zc = np.append(vec_state_as_np[1:], 1 - np.sum(vec_state_as_np[1:]))
        
        self.clean_arrays()
        self.ph = np.array([0], dtype=np.intp) # Always water phase
        self.sat[0] = 1.0
        self.kr[0] = 1.0
        
        for i in range(self.nc):
            self.x[0, i] = zc[i]
            
        M = np.sum(self.x[0, :] * self.Mw) + 1e-20
        self.dens[0] = self.density_ev['water'].evaluate(pressure, self.temperature)
        self.dens_m[0] = self.dens[0] / M
        self.mu[0] = self.viscosity_ev['water'].evaluate(pressure, self.temperature)
        self.enthalpy[0] = 100.0 # dummy
        self.cond[0] = 10.0 # dummy

class CustomPhysics(Compositional):
    def __init__(self, timer, n_points, min_p, max_p, compr, viscos):
        components = ['H2O_Cold', 'H2O_Hot']
        phases = ['water']
        zero = 1e-12
        property_container = CustomPropertyContainer(phases, components, compr, viscos)
        
        super().__init__(components, phases, timer, n_points, min_p, max_p, 
                         min_z=zero, max_z=1.0 - zero, cache=False)
        self.add_property_region(property_container)

def export_wells_to_vtk(reservoir, output_directory):
    if not pv: return
    wells_data = []
    for well in reservoir.wells:
        points = []
        perfs = [p for p in well.perforations if p[1] >= 0]
        if not perfs: continue
        
        # 1. Add surface point (Z=0)
        first_perf_idx = perfs[0][1]
        surf_coords = reservoir.discretizer.centroids_all_cells[first_perf_idx].copy()
        surf_coords[2] = 0.0 # Surface
        points.append(surf_coords)
        
        # 2. Add perforation points (offset by -2000m)
        for perf in perfs:
            coords = reservoir.discretizer.centroids_all_cells[perf[1]].copy()
            coords[2] -= 2000.0 
            points.append(coords)
            
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
        
        # 1. Randomization parameters
        def jitter(val): return float(val * np.random.uniform(0.9, 1.1))
        op_params = {
            'inj_bhp': float(np.random.uniform(250, 450)),
            'prod_bhp': float(np.random.uniform(50, 150)),
            'water_compr': float(WATER_COMPR * np.random.uniform(0.5, 2.0)),
            'water_visc': float(WATER_VISC * np.random.uniform(0.8, 1.2))
        }

        # 2. Randomize Wells (Distance >= 20)
        max_attempts = 100
        for _ in range(max_attempts):
            inj_coord = [int(np.random.randint(5, 95)), int(np.random.randint(5, 95))]
            prod_coord = [int(np.random.randint(5, 95)), int(np.random.randint(5, 95))]
            dist = np.sqrt((inj_coord[0]-prod_coord[0])**2 + (inj_coord[1]-prod_coord[1])**2)
            if dist >= 20.0: break
        
        perf_inj = [int(k) for k in sorted(np.random.choice(range(1, NZ+1), size=np.random.randint(2, 6), replace=False))]
        perf_prod = [int(k) for k in sorted(np.random.choice(range(1, NZ+1), size=np.random.randint(2, 6), replace=False))]
        
        # 3. Randomize Properties
        poro, permx, permy, permz, hcap, rcond, layer_stats = generate_property_arrays(NX, NY, NZ, seed)
        
        for entry in layer_stats:
            entry.update({
                'inj_x': inj_coord[0], 'inj_y': inj_coord[1],
                'prod_x': prod_coord[0], 'prod_y': prod_coord[1],
                'inj_bhp': op_params['inj_bhp'], 'prod_bhp': op_params['prod_bhp'],
                'water_compr': op_params['water_compr'], 'water_visc': op_params['water_visc']
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
                            inj_coord, prod_coord, perf_inj, perf_prod, op_params)
        m.init()
        vtk_path = os.path.join(real_dir, 'vtk_files')
        m.set_output(vtk_path)
        export_wells_to_vtk(m.reservoir, vtk_path)
        
        vars_to_eval = m.physics.vars
        timesteps, property_array = m.output.output_properties(output_properties=vars_to_eval, timestep=0, engine=True)
        for name, arr in [('poro', poro), ('permx', permx), ('permy', permy), ('permz', permz)]:
            property_array[name] = np.array([arr])
        m.output.output_to_vtk(output_data=[timesteps, property_array], ith_step=0)
        
        print_progress(0, TOTAL_DAYS, i)
        for d in range(1, TOTAL_DAYS + 1):
            m.run(1.0)
            if d % REPORT_FREQ == 0 or d == TOTAL_DAYS:
                timesteps, property_array = m.output.output_properties(output_properties=vars_to_eval, timestep=d, engine=True)
                # Re-inject geological properties for ParaView (they are not tracked by the engine)
                for name, arr in [('poro', poro), ('permx', permx), ('permy', permy), ('permz', permz)]:
                    property_array[name] = np.array([arr])
                m.output.output_to_vtk(output_data=[timesteps, property_array], ith_step=d)
            print_progress(d, TOTAL_DAYS, i)
        
        print(f"\nRealization {i} complete.")
        pd.DataFrame(m.physics.engine.time_data).to_csv(os.path.join(real_dir, 'history_isothermal.csv'))
        del m
        
    print(f"\n\033[92mSimulation [ISOTHERMAL V4 ENSEMBLE] success!\033[0m")
    print(f"All results saved in: {ensemble_root}")

if __name__ == "__main__":
    main()
