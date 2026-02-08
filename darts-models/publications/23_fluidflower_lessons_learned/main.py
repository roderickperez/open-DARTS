import numpy as np
import pandas as pd

from model import Model, PorPerm, Corey
from darts.engines import value_vector, redirect_darts_output, sim_params, well_control_iface#, set_gpu_device
from visualization.output import CSVResults

from fluidflower_str import FluidFlowerStruct
from fluidflower_unstr import FluidFlowerUnstruct


# region PorPerm
# Regions: 0 - very-coarse, 1 - coarse, 2 - fine, 3 - very-fine
# property_regions = {0: ["E", "F", "G", "W"], 1: ["D"], 2: ["C"], 3: ["ESF"]}
property_regions = [0, 1, 2, 3]
layers_to_regions = {"ESF": 3, "C": 2, "D": 1, "E": 0, "F": 0, "G": 0, "W": 0}

# define the Corey parameters for each layer, estimated corey parameters from paper Abaci (2006)
# Paper + description Combined
corey = {
    0: Corey(nw=2.0, ng=1.5, swc=0.11, sgc=0.06, krwe=0.80, krge=0.85, labda=2., p_entry=0.),
    1: Corey(nw=2.0, ng=1.5, swc=0.12, sgc=0.08, krwe=0.93, krge=0.95, labda=2., p_entry=1e-3),
    2: Corey(nw=2.5, ng=2.0, swc=0.14, sgc=0.10, krwe=0.93, krge=0.95, labda=2., p_entry=3e-3),
    3: Corey(nw=2.5, ng=2.0, swc=0.32, sgc=0.14, krwe=0.71, krge=0.75, labda=2., p_entry=15e-3)
}

# Standard deviation of Corey parameters (fractions)
stdCorey = {
    0: Corey(nw=0.1, ng=0.1, swc=0.5, sgc=0.5, krwe=0.05, krge=0.05, labda=2., p_entry=0.),
    1: Corey(nw=0.1, ng=0.1, swc=0.5, sgc=0.5, krwe=0.05, krge=0.05, labda=2., p_entry=0.),
    2: Corey(nw=0.1, ng=0.1, swc=0.5, sgc=0.5, krwe=0.05, krge=0.05, labda=2., p_entry=0.),
    3: Corey(nw=0.1, ng=0.1, swc=0.5, sgc=0.5, krwe=0.05, krge=0.05, labda=2., p_entry=0.)
}

layer_props = {900001: PorPerm(type='G', poro=0.44, perm=2276750.01, anisotropy=[1, 1, 1]),
               900002: PorPerm(type='F', poro=0.45, perm=3246353.97, anisotropy=[1, 1, 1]),
               900003: PorPerm(type='ESF', poro=0.43, perm=44000.0, anisotropy=[1, 1, 0.747972992]),
               900004: PorPerm(type='E', poro=0.45, perm=1441352.77, anisotropy=[1, 1, 0.9]),
               900005: PorPerm(type='D', poro=0.44, perm=1878990.31, anisotropy=[1, 1, 0.8]),
               900006: PorPerm(type='C', poro=0.44, perm=385597.374, anisotropy=[1, 1, 0.7]),
               900007: PorPerm(type='G', poro=0.44, perm=2276750.01, anisotropy=[1, 1, 1]),
               900008: PorPerm(type='D', poro=0.44, perm=1878990.31, anisotropy=[1, 1, 0.8]),
               900009: PorPerm(type='G', poro=0.44, perm=2276750.01, anisotropy=[1, 1, 1]),
               900010: PorPerm(type='E', poro=0.45, perm=1441352.77, anisotropy=[1, 1, 0.9]),
               900011: PorPerm(type='G', poro=0.44, perm=2276750.01, anisotropy=[1, 1, 1]),
               900012: PorPerm(type='F', poro=0.45, perm=3246353.97, anisotropy=[1, 1, 1]),
               900013: PorPerm(type='ESF', poro=0.43, perm=44000.0, anisotropy=[1, 1, 0.747972992]),
               900014: PorPerm(type='E', poro=0.45, perm=1441352.77, anisotropy=[1, 1, 0.9]),
               900015: PorPerm(type='D', poro=0.44, perm=1878990.31, anisotropy=[1, 1, 0.8]),
               900016: PorPerm(type='F', poro=0.45, perm=3246353.97, anisotropy=[1, 1, 1]),
               900017: PorPerm(type='E', poro=0.45, perm=1441352.77, anisotropy=[1, 1, 0.9]),
               900018: PorPerm(type='D', poro=0.44, perm=1878990.31, anisotropy=[1, 1, 0.8]),
               900019: PorPerm(type='C', poro=0.44, perm=385597.374, anisotropy=[1, 1, 0.7]),
               900020: PorPerm(type='F', poro=0.45, perm=3246353.97, anisotropy=[1, 1, 1]),
               900021: PorPerm(type='E', poro=0.45, perm=1441352.77, anisotropy=[1, 1, 0.9]),
               900022: PorPerm(type='D', poro=0.44, perm=1878990.31, anisotropy=[1, 1, 0.8]),
               900023: PorPerm(type='C', poro=0.44, perm=385597.374, anisotropy=[1, 1, 0.7]),
               900024: PorPerm(type='G', poro=0.44, perm=2276750.01, anisotropy=[1, 1, 1]),
               900025: PorPerm(type='F', poro=0.45, perm=3246353.97, anisotropy=[1, 1, 1]),
               900026: PorPerm(type='E', poro=0.45, perm=1441352.77, anisotropy=[1, 1, 0.9]),
               900027: PorPerm(type='E', poro=0.45, perm=1441352.77, anisotropy=[1, 1, 0.9]),
               900028: PorPerm(type='D', poro=0.44, perm=1878990.31, anisotropy=[1, 1, 0.8]),
               900029: PorPerm(type='C', poro=0.44, perm=385597.374, anisotropy=[1, 1, 0.7]),
               900030: PorPerm(type='ESF', poro=0.43, perm=44000.0, anisotropy=[1, 1, 0.747972992]),
               900031: PorPerm(type='W', poro=0.44, perm=100000000.0, anisotropy=[1, 1, 1]),
               }

# endregion

redirect_darts_output('run.log')

"""Define realization ID"""
# Specify here the specific model:
model_specs = [
    {'structured': True, 'thickness': False, 'curvature': False, 'tpfa': True, 'capillary': True, 'nx': 150, 'nz': 75, 'output_dir': 'str_coarse'},
    # {'structured': True, 'thickness': False, 'curvature': False, 'tpfa': True, 'capillary': True, 'nx': 300, 'nz': 150, 'output_dir': 'str_fine'},
    # {'structured': False, 'thickness': False, 'curvature': False, 'tpfa': True, 'capillary': True, 'lc': [0.03, 0.003], 'output_dir': 'unstr_coarse'},
    # {'structured': False, 'thickness': False, 'curvature': False, 'tpfa': True, 'capillary': True, 'lc': [0.015, 0.003], 'output_dir': 'unstr_fine'},
    # {'structured': True, 'thickness': False, 'curvature': False, 'tpfa': True, 'capillary': True, 'nx': 600, 'nz': 300, 'output_dir': 'str_fine'},
    # {'structured': False, 'thickness': False, 'curvature': False, 'tpfa': True, 'capillary': True, 'lc': [0.008, 0.003], 'output_dir': 'unstr_finer'},
]

for j, specs in enumerate(model_specs):
    structured = specs['structured']

    m = Model()

    """Define reservoir and wells"""
    well_centers = {
        "I1": [0.925, 0.0, 0.32942],
        "I2": [1.72806, 0.0, 0.72757]
    }

    if structured:
        m.reservoir = FluidFlowerStruct(timer=m.timer, layer_properties=layer_props, layers_to_regions=layers_to_regions,
                                        model_specs=specs, well_centers=well_centers)
    else:
        m.reservoir = FluidFlowerUnstruct(timer=m.timer, layer_properties=layer_props, layers_to_regions=layers_to_regions,
                                          model_specs=specs, well_centers=well_centers)

    """Define physics"""
    zero = 1e-10
    m.set_physics(corey=corey, zero=zero, temperature=296., n_points=10001)

    # Some newton parameters for non-linear solution:
    m.set_sim_params(first_ts=1e-8, mult_ts=2, max_ts=0.00694, tol_linear=1e-3, tol_newton=1e-3,
                     it_linear=200, it_newton=15, newton_type=sim_params.newton_local_chop)
    # m.params.newton_params[0] = 0.2
    m.params.linear_type = sim_params.cpu_superlu

    m.pres_in = 1.01325
    m.inj_comp = [zero]
    m.inj_temp = 293.
    m.inj_rate = 10 * 24 * 60 / 1E6  # ml/min to m3/day
    # m.reservoir.min_depth = 0.1367
    m.p_prod = m.pres_in + m.reservoir.min_depth * 0.1 if hasattr(m.reservoir, "min_depth") else pres_in + 0.01367
    m.zero = zero

    m.init(discr_type='tpfa', platform='cpu')
    m.set_output(save_initial = False, verbose = True)

    # m.p_prod = pres_in + m.reservoir.min_depth * 0.1

    output_prop = ["pressure", "satV", "xCO2", "rhoV", "rho_mA"]
    output_prop += ["temperature"] if m.physics.thermal else []

    # Initialize CSV reporting object
    output_dir = specs['output_dir']
    csv = CSVResults(output_dir)

    if structured:
        csv.centroids = m.reservoir.centroids
        v = m.reservoir.discretizer.volume.flatten()[m.reservoir.global_to_local]
        csv.volume = np.ones(m.reservoir.mesh.n_res_blocks) * v
    else:
        csv.centroids = m.reservoir.discretizer.centroid_all_cells
        csv.volume = m.reservoir.discretizer.volume_all_cells

    csv.poro = np.array(m.reservoir.mesh.poro, copy=False)
    csv.op_num = np.array(m.reservoir.mesh.op_num, copy=False)
    csv.corey = corey
    csv.seal = m.reservoir.seal
    # csv.conn_0 = m.reservoir.conn_0
    # csv.conn_1 = m.reservoir.conn_1
    csv.regions = property_regions
    csv.op_num = np.array(m.reservoir.mesh.op_num, copy=False)

    csv.find_output_cells(specs['curvature'], cache=1)

    # Time-related properties:
    # Specify some other time-related properties (NOTE: all time parameters are in [days])
    size_report_step = 5 / (24 * 60)  # Half Size of the reporting step (when output is writen to .vtk format)
    num_report_steps = int(5. / size_report_step)

    if 1:
        m.max_dt = 1e-7
        m.run_python_my(1e-6, )  # pre-equilibration
        m.output.save_data_to_h5(kind='reservoir')
    else:
        m.load_restart_data(filename='output/solution.h5')

    # Output properties and export to vtk
    m.output.output_to_vtk(ith_step=0, output_directory=output_dir, output_properties=output_prop)

    timesteps, property_array = m.output.output_properties(output_properties=output_prop, timestep=0)
    csv.update_time_series(property_array)
    csv.write_time_series()
    csv.ith_step_ts += 1

    # Run over all reporting time-steps:
    event1 = True
    event2 = True
    eps = 1e-6
    factor = 5  # factor to timestep due to refinement (=1 for coarse model)

    max_dt = (size_report_step + eps) / 5
    m.max_dt = max_dt
    m.params.max_ts = max_dt
    m.physics.set_well_controls(wctrl=m.reservoir.wells[0].control, is_inj=True, control_type=well_control_iface.VOLUMETRIC_RATE,
                                target=m.inj_rate, phase_name='V', inj_composition=m.inj_comp, inj_temp=m.inj_temp)
    first_ts = 1e-8
    for i in range(num_report_steps):

        print('\n---------------------------SELF-PRINT #%d---------------------------' % (i + 1))

        if m.physics.engine.t > 0.09375 - eps and event1:
            print("Injector 2 started")
            m.physics.set_well_controls(wctrl=m.reservoir.wells[1].control, is_inj=True, control_type=well_control_iface.VOLUMETRIC_RATE,
                                        target=m.inj_rate, phase_name='V', inj_composition=m.inj_comp, inj_temp=m.inj_temp)
            event1 = False

        if m.physics.engine.t > 0.20833 - eps and event2:
            print("Both injectors stopped")
            m.physics.set_well_controls(wctrl=m.reservoir.wells[0].control, is_inj=True, control_type=well_control_iface.VOLUMETRIC_RATE,
                                        target=0., phase_name='V', inj_composition=m.inj_comp, inj_temp=m.inj_temp)
            m.physics.set_well_controls(wctrl=m.reservoir.wells[1].control, is_inj=True, control_type=well_control_iface.VOLUMETRIC_RATE,
                                        target=0., phase_name='V', inj_composition=m.inj_comp, inj_temp=m.inj_temp)
            event2 = False

        print('Run simulation from %f to %f days with max dt = %f'
              % (m.physics.engine.t, m.physics.engine.t + size_report_step, m.max_dt))
        if m.physics.engine.t < 1:
            max_res = 1e6
        else:
            max_res = 1e10

        m.run_python_my(size_report_step, restart_dt=first_ts, max_res=max_res)
        # first_ts = 0
        m.output.save_data_to_h5(kind='reservoir')

        timesteps, property_array = m.output.output_properties(output_properties=output_prop, timestep=i+1)

        if i % 2 != 0.:
            csv.update_time_series(property_array)
            csv.write_time_series()
            csv.ith_step_ts += 1

            if (i + 1) % int(1 / size_report_step) == 0.:
                t = str(24 * int((i + 1) / int(1 / size_report_step))) + 'h'
                csv.write_spatial_map(property_array, t)

            m.output.output_to_vtk(ith_step = i+1, output_directory=output_dir, output_properties=output_prop)

    # from visualization.visualize_spatial_maps import visualizeBoxC
    # timestamps = np.array([4, 8, 12, 16, 20, 24]) * 3600
    # visualizeBoxC(output_dir, timestamps)

    # # Output spatial maps and time series
    # from visualization.visualize_spatial_maps import visualizeSpatialMaps
    # visualizeSpatialMaps(output_dir)
    # from visualization.visualize_time_series import visualizeTimeSeries
    # visualizeTimeSeries(output_dir)

    # After the simulation, print some of the simulation timers and statistics,
    # newton iters, etc., how much time spent where:
    m.print_timers()
    m.print_stat()

    # time_data = pd.DataFrame.from_dict(m.physics.engine.time_data)
    # writer = pd.ExcelWriter('time_data.xlsx')
    # time_data.to_excel(writer, 'Sheet1')
    # writer.save()
