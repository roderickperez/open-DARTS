import numpy as np
import pandas as pd

from model import Model, PorPerm, Corey
from darts.engines import value_vector, redirect_darts_output, sim_params#, set_gpu_device
from visualization.output import CSVResults

from fluidflower_str_b import FluidFlowerStruct
from fluidflower_unstr_b import FluidFlowerUnstruct


# Facies PorPerm
# Facies: 1-7
# property_regions = {0: ["E", "F", "G", "W"], 1: ["D"], 2: ["C"], 3: ["ESF"]}
property_regions = [0, 1, 2, 3, 4, 5, 6]
layers_to_regions = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6}

# define the Corey parameters for each layer, estimated corey parameters from paper Abaci (2006)
# Paper + description Combined

# need to fix krwe, krge, labda, and p_entry?)

corey = {
    0: Corey(nw=1.5, ng=1.5, swc=0.32, sgc=0.10, krwe=0.80, krge=0.85, labda=2., p_entry=1.935314),
    1: Corey(nw=1.5, ng=1.5, swc=0.14, sgc=0.10, krwe=0.93, krge=0.95, labda=2., p_entry=0.08655),
    2: Corey(nw=1.5, ng=1.5, swc=0.12, sgc=0.10, krwe=0.93, krge=0.95, labda=2., p_entry=0.0612),
    3: Corey(nw=1.5, ng=1.5, swc=0.12, sgc=0.10, krwe=0.71, krge=0.75, labda=2., p_entry=0.038706),
    4: Corey(nw=1.5, ng=1.5, swc=0.12, sgc=0.10, krwe=0.71, krge=0.75, labda=2., p_entry=0.0306),
    5: Corey(nw=1.5, ng=1.5, swc=0.10, sgc=0.10, krwe=0.71, krge=0.75, labda=2., p_entry=0.025602),
    6: Corey(nw=1.5, ng=1.5, swc=1e-8, sgc=0.10, krwe=0.71, krge=0.75, labda=2., p_entry=15)
}

# Standard deviation of Corey parameters (fractions)
stdCorey = {
    0: Corey(nw=0.1, ng=0.1, swc=0.5, sgc=0.5, krwe=0.05, krge=0.05, labda=2., p_entry=0.),
    1: Corey(nw=0.1, ng=0.1, swc=0.5, sgc=0.5, krwe=0.05, krge=0.05, labda=2., p_entry=0.),
    2: Corey(nw=0.1, ng=0.1, swc=0.5, sgc=0.5, krwe=0.05, krge=0.05, labda=2., p_entry=0.),
    3: Corey(nw=0.1, ng=0.1, swc=0.5, sgc=0.5, krwe=0.05, krge=0.05, labda=2., p_entry=0.),
    4: Corey(nw=0.1, ng=0.1, swc=0.5, sgc=0.5, krwe=0.05, krge=0.05, labda=2., p_entry=0.),
    5: Corey(nw=0.1, ng=0.1, swc=0.5, sgc=0.5, krwe=0.05, krge=0.05, labda=2., p_entry=0.),
    6: Corey(nw=0.1, ng=0.1, swc=0.5, sgc=0.5, krwe=0.05, krge=0.05, labda=2., p_entry=0.)
}

layer_props = {900001: PorPerm(type='7', poro=1e-6, perm=1e-6, anisotropy=[1, 1, 0.1]),
               900002: PorPerm(type='5', poro=0.25, perm=1013.24997, anisotropy=[1, 1, 0.1]),
               900003: PorPerm(type='5', poro=0.25, perm=1013.24997, anisotropy=[1, 1, 0.1]),
               900004: PorPerm(type='5', poro=0.25, perm=1013.24997, anisotropy=[1, 1, 0.1]),
               900005: PorPerm(type='5', poro=0.25, perm=1013.24997, anisotropy=[1, 1, 0.1]),
               900006: PorPerm(type='5', poro=0.25, perm=1013.24997, anisotropy=[1, 1, 0.1]),
               900007: PorPerm(type='1', poro=0.1, perm=0.101324997, anisotropy=[1, 1, 0.1]),
               900008: PorPerm(type='1', poro=0.1, perm=0.101324997, anisotropy=[1, 1, 0.1]),
               900009: PorPerm(type='1', poro=0.1, perm=0.101324997, anisotropy=[1, 1, 0.1]),
               900010: PorPerm(type='4', poro=0.2, perm=506.624985, anisotropy=[1, 1, 0.1]),
               900011: PorPerm(type='4', poro=0.2, perm=506.624985, anisotropy=[1, 1, 0.1]),
               900012: PorPerm(type='4', poro=0.2, perm=506.624985, anisotropy=[1, 1, 0.1]),
               900013: PorPerm(type='4', poro=0.2, perm=506.624985, anisotropy=[1, 1, 0.1]),
               900014: PorPerm(type='4', poro=0.2, perm=506.624985, anisotropy=[1, 1, 0.1]),
               900015: PorPerm(type='4', poro=0.2, perm=506.624985, anisotropy=[1, 1, 0.1]),
               900016: PorPerm(type='3', poro=0.2, perm=202.649994, anisotropy=[1, 1, 0.1]),
               900017: PorPerm(type='3', poro=0.2, perm=202.649994, anisotropy=[1, 1, 0.1]),
               900018: PorPerm(type='3', poro=0.2, perm=202.649994, anisotropy=[1, 1, 0.1]),
               900019: PorPerm(type='3', poro=0.2, perm=202.649994, anisotropy=[1, 1, 0.1]),
               900020: PorPerm(type='3', poro=0.2, perm=202.649994, anisotropy=[1, 1, 0.1]),
               900021: PorPerm(type='3', poro=0.2, perm=202.649994, anisotropy=[1, 1, 0.1]),
               900022: PorPerm(type='4', poro=0.2, perm=506.624985, anisotropy=[1, 1, 0.1]),
               900023: PorPerm(type='6', poro=0.35, perm=2026.49994, anisotropy=[1, 1, 0.1]),
               900024: PorPerm(type='6', poro=0.35, perm=2026.49994, anisotropy=[1, 1, 0.1]),
               900025: PorPerm(type='6', poro=0.35, perm=2026.49994, anisotropy=[1, 1, 0.1]),
               900026: PorPerm(type='2', poro=0.2, perm=101.324997, anisotropy=[1, 1, 0.1]),
               900027: PorPerm(type='2', poro=0.2, perm=101.324997, anisotropy=[1, 1, 0.1]),
               900028: PorPerm(type='2', poro=0.2, perm=101.324997, anisotropy=[1, 1, 0.1]),
               900029: PorPerm(type='2', poro=0.2, perm=101.324997, anisotropy=[1, 1, 0.1]),
               900030: PorPerm(type='2', poro=0.2, perm=101.324997, anisotropy=[1, 1, 0.1]),
               900031: PorPerm(type='7', poro=1e-6, perm=1e-6, anisotropy=[1, 1, 0.1]),
               900032: PorPerm(type='1', poro=0.1, perm=0.101324997, anisotropy=[1, 1, 0.1]),
               }

# endregion

redirect_darts_output('run.log')

"""Define realization ID"""
# Specify here the specific model:
model_specs = [
    # {'structured': True, 'thickness': False, 'curvature': False, 'tpfa': True, 'capillary': True, 'nx': 150, 'nz': 75, 'output_dir': 'str_coarse_b'},
    # {'structured': True, 'thickness': False, 'curvature': False, 'tpfa': True, 'capillary': True, 'nx': 300, 'nz': 150, 'output_dir': 'str_fine_b'},
    {'structured': False, 'thickness': False, 'curvature': False, 'tpfa': True, 'capillary': True, 'lc': [30, 3], 'output_dir': 'unstr_coarse_b'},
    # {'structured': False, 'thickness': False, 'curvature': False, 'tpfa': True, 'capillary': True, 'lc': [0.015, 0.003], 'output_dir': 'unstr_fine_b'},
    # {'structured': True, 'thickness': False, 'curvature': False, 'tpfa': True, 'capillary': True, 'nx': 600, 'nz': 300, 'output_dir': 'str_finer_b'},
    # {'structured': False, 'thickness': False, 'curvature': False, 'tpfa': True, 'capillary': True, 'lc': [0.008, 0.003], 'output_dir': 'unstr_finer_b'},
]

for j, specs in enumerate(model_specs):
    structured = specs['structured']

    m = Model()

    """Define reservoir and wells"""
    well_centers = {
        "I1": [2700.0, 0.0, 300.0],
        "I2": [5100.0, 0.0, 700.0]
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

    pres_in = 210 # initial pressure at well 1
    m.initial_values = {"pressure": pres_in,
                        "H2O": 1. - zero,
                        "temperature": 393.15}
    m.gradient = {"pressure": 0.09995}

    m.inj_stream = [1.-zero]
    if m.physics.thermal:
        m.inj_stream.append(t_inj)
    m.inj_rate = 10.294 * 24 * 60 / 1E6  # ml/min to m3/day, need to improve kg/s*m to ml/min conversion
    # m.reservoir.min_depth = 2000
    # m.p_prod = pres_in # + m.reservoir.min_depth * 0.1

    m.init(discr_type='tpfa', platform='cpu')

    # m.p_prod = pres_in # + m.reservoir.min_depth * 0.1
    # m.reservoir.wells[2].control = m.physics.new_bhp_prod(m.p_prod)

    output_prop = m.physics.vars + m.physics.property_operators[0].props_name

    # Initialize CSV reporting object
    output_dir = specs['output_dir']
    # csv = CSVResults(output_prop, output_dir)
    #
    # if structured:
    #     csv.centroids = m.reservoir.centroids
    #     v = m.reservoir.discretizer.volume.flatten()[m.reservoir.global_to_local]
    #     csv.volume = np.ones(m.reservoir.mesh.n_res_blocks) * v
    # else:
    #     csv.centroids = m.reservoir.discretizer.centroid_all_cells
    #     csv.volume = m.reservoir.discretizer.volume_all_cells
    #
    # csv.poro = np.array(m.reservoir.mesh.poro, copy=False)
    # csv.op_num = np.array(m.reservoir.mesh.op_num, copy=False)
    # csv.corey = corey
    # csv.seal = m.reservoir.seal
    # # csv.conn_0 = m.reservoir.conn_0
    # # csv.conn_1 = m.reservoir.conn_1
    # csv.regions = property_regions
    # csv.op_num = np.array(m.reservoir.mesh.op_num, copy=False)
    #
    # csv.find_output_cells(specs['curvature'], cache=1)

    # Time-related properties:
    # Specify some other time-related properties (NOTE: all time parameters are in [days])
    size_report_step = 5 / (24 * 60)  # Half Size of the reporting step (when output is writen to .vtk format)
    num_report_steps = int(5. / size_report_step)

    if 1:
        m.max_dt = 1e-7
        m.run_python_my(1e-6)  # pre-equilibration
        m.save_restart_data('post_equil.pkl')
    else:
        m.load_restart_data('post_equil.pkl')

    # Output properties and export to vtk
    m.output_to_vtk(ith_step=0, output_directory=output_dir)

    property_array = m.output_properties()
    # csv.update_time_series(property_array)
    # csv.write_time_series()
    # csv.ith_step_ts += 1

    # Run over all reporting time-steps:
    event1 = True
    event2 = True
    eps = 1e-6
    factor = 1  # factor to timestep due to refinement (=1 for coarse model)

    max_dt = (size_report_step + eps) / 5
    m.max_dt = max_dt
    m.params.max_ts = max_dt
    m.reservoir.wells[0].control = m.physics.new_rate_inj(m.inj_rate, m.inj_stream, 0)
    first_ts = 1e-8
    for i in range(num_report_steps):

        print('\n---------------------------SELF-PRINT #%d---------------------------' % (i + 1))

        if m.physics.engine.t > 0.09375 - eps and event1:
            print("Injector 2 started")
            m.reservoir.wells[1].control = m.physics.new_rate_inj(m.inj_rate, m.inj_stream, 0)
            event1 = False

        if m.physics.engine.t > 0.20833 - eps and event2:
            print("Both injectors stopped")
            m.reservoir.wells[0].control = m.physics.new_rate_inj(0, m.inj_stream, 0)
            m.reservoir.wells[1].control = m.physics.new_rate_inj(0, m.inj_stream, 0)
            event2 = False
            m.save_restart_data('end_of_injection.pkl')

        print('Run simulation from %f to %f days with max dt = %f'
              % (m.physics.engine.t, m.physics.engine.t + size_report_step, m.max_dt))
        if m.physics.engine.t < 1:
            max_res = 1e6
        else:
            max_res = 1e10

        m.run_python_my(size_report_step, restart_dt=first_ts, max_res=max_res)
        first_ts = 0

        property_array = m.output_properties()

        if i % 2 != 0.:
            # csv.update_time_series(property_array)
            # csv.write_time_series()
            # csv.ith_step_ts += 1
            #
            # if (i + 1) % int(1 / size_report_step) == 0.:
            #     t = str(24 * int((i + 1) / int(1 / size_report_step))) + 'h'
            #     csv.write_spatial_map(property_array, t)

            m.output_to_vtk(ith_step=i+1, output_directory=output_dir)

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
