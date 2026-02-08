import numpy as np
import pandas as pd

from model import Model, PorPerm, Corey
from darts.engines import value_vector, redirect_darts_output, well_control_iface#, set_gpu_device
from visualization.output import CSVResults


# region PorPerm
# define the Corey parameters for each layer, estimated corey parameters from paper Abaci (2006)
# Paper + description Combined
corey = {
    "very-coarse":  Corey(nw=2.0, ng=1.5, swc=0.11, sgc=0.06, krwe=0.80, krge=0.85, labda=2., p_entry=0.0),
    "coarse":       Corey(nw=2.0, ng=1.5, swc=0.12, sgc=0.08, krwe=0.93, krge=0.95, labda=2., p_entry=0.01),
    "fine":         Corey(nw=2.5, ng=2.0, swc=0.14, sgc=0.10, krwe=0.93, krge=0.95, labda=2., p_entry=0.03),
    "very-fine":    Corey(nw=2.5, ng=2.0, swc=0.32, sgc=0.14, krwe=0.71, krge=0.75, labda=2., p_entry=0.1)
}

# Standard deviation of Corey parameters (fractions)
stdCorey = {
    "very-coarse":  Corey(nw=0.1, ng=0.1, swc=0.5, sgc=0.5, krwe=0.05, krge=0.05, labda=2., p_entry=0.),
    "coarse":       Corey(nw=0.1, ng=0.1, swc=0.5, sgc=0.5, krwe=0.05, krge=0.05, labda=2., p_entry=0.),
    "fine":         Corey(nw=0.1, ng=0.1, swc=0.5, sgc=0.5, krwe=0.05, krge=0.05, labda=2., p_entry=0.),
    "very-fine":    Corey(nw=0.1, ng=0.1, swc=0.5, sgc=0.5, krwe=0.05, krge=0.05, labda=2., p_entry=0.)
}

permg = {
  "G": 2e4,
  "E": 2e4,
  "D": 5e3,
  "C": 1e3,
  "F": 1e3,
  "ESF": 5e2
}

poro = 0.2

layer_props = {900001: PorPerm(type='G', poro=poro, perm=permg['G'], anisotropy=[1, 1, 1]),
               900002: PorPerm(type='F', poro=poro, perm=permg['F'], anisotropy=[1, 1, 1]),
               900003: PorPerm(type='ESF', poro=poro, perm=permg['ESF'], anisotropy=[1, 1, 0.747972992]),
               900004: PorPerm(type='E', poro=poro, perm=permg['E'], anisotropy=[1, 1, 0.9]),
               900005: PorPerm(type='D', poro=poro, perm=permg['D'], anisotropy=[1, 1, 0.8]),
               900006: PorPerm(type='C', poro=poro, perm=permg['C'], anisotropy=[1, 1, 0.7]),
               900007: PorPerm(type='G', poro=poro, perm=permg['G'], anisotropy=[1, 1, 1]),
               900008: PorPerm(type='D', poro=poro, perm=permg['D'], anisotropy=[1, 1, 0.8]),
               900009: PorPerm(type='G', poro=poro, perm=permg['G'], anisotropy=[1, 1, 1]),
               900010: PorPerm(type='E', poro=poro, perm=permg['D'], anisotropy=[1, 1, 0.9]),
               900011: PorPerm(type='G', poro=poro, perm=permg['G'], anisotropy=[1, 1, 1]),
               900012: PorPerm(type='F', poro=poro, perm=permg['F'], anisotropy=[1, 1, 1]),
               900013: PorPerm(type='ESF', poro=poro, perm=permg['ESF'], anisotropy=[1, 1, 0.747972992]),
               900014: PorPerm(type='E', poro=poro, perm=permg['E'], anisotropy=[1, 1, 0.9]),
               900015: PorPerm(type='D', poro=poro, perm=permg['D'], anisotropy=[1, 1, 0.8]),
               900016: PorPerm(type='F', poro=poro, perm=permg['F'], anisotropy=[1, 1, 1]),
               900017: PorPerm(type='E', poro=poro, perm=permg['D'], anisotropy=[1, 1, 0.9]),
               900018: PorPerm(type='D', poro=poro, perm=permg['D'], anisotropy=[1, 1, 0.8]),
               900019: PorPerm(type='C', poro=poro, perm=permg['C'], anisotropy=[1, 1, 0.7]),
               900020: PorPerm(type='F', poro=poro, perm=permg['F'], anisotropy=[1, 1, 1]),
               900021: PorPerm(type='E', poro=poro, perm=permg['D'], anisotropy=[1, 1, 0.9]),
               900022: PorPerm(type='D', poro=poro, perm=permg['D'], anisotropy=[1, 1, 0.8]),
               900023: PorPerm(type='C', poro=poro, perm=permg['C'], anisotropy=[1, 1, 0.7]),
               900024: PorPerm(type='G', poro=poro, perm=permg['G'], anisotropy=[1, 1, 1]),
               900025: PorPerm(type='F', poro=poro, perm=permg['F'], anisotropy=[1, 1, 1]),
               900026: PorPerm(type='E', poro=poro, perm=permg['D'], anisotropy=[1, 1, 0.9]),
               900027: PorPerm(type='E', poro=poro, perm=permg['E'], anisotropy=[1, 1, 0.9]),
               900028: PorPerm(type='D', poro=poro, perm=permg['D'], anisotropy=[1, 1, 0.8]),
               900029: PorPerm(type='C', poro=poro, perm=permg['C'], anisotropy=[1, 1, 0.7]),
               900030: PorPerm(type='ESF', poro=poro, perm=permg['ESF'], anisotropy=[1, 1, 0.747972992]),
               900031: PorPerm(type='W', poro=poro, perm=10000, anisotropy=[1, 1, 1]),
               }

# endregion

redirect_darts_output('run.log')

"""Define realization ID"""
# Specify here the specific model:
thermal = False
structured = False
if structured:
    output_dir = 'str_'
    if 0:
        output_dir += 'coarse'
        nx, nz = 100, 50
    elif 1:
        output_dir += 'refined'
        nx, nz = 200, 100
    else:
        output_dir += 'fine'
        nx, nz = 300, 150

    model_specs = {
        'structured': structured,
        'thickness': False,
        'curvature': False,
        'tpfa': False,
        'capillary': True,
        'nx': nx,
        'nz': nz
    }

else:
    thickness = True
    curvature = True
    tpfa = True

    output_dir = 'unstr_' + ('tpfa_' if tpfa else 'mpfa_')
    if 0:
        output_dir += 'coarse'
        lc = [0.03, 0.003]
    elif 1:
        output_dir += 'refined'
        lc = [0.015, 0.003]
    else:
        output_dir += 'fine'
        lc = [0.01, 0.003]

    model_specs = {
        'structured': structured,
        'thickness': thickness,
        'curvature': curvature,
        'tpfa': tpfa,
        'capillary': True,
        'lc': lc
    }

#m = Model(layer_props, corey, model_specs)
m = Model(layer_props, corey, model_specs, temp_in=320, pres_in=100)
m.init()

output_prop = m.output_props.vars + m.output_props.props
n_vars = m.output_props.n_vars

# Initialize CSV reporting object
csv = CSVResults(output_prop, output_dir)

regions = {0: "very-coarse", 1: "coarse", 2: "fine", 3: "very-fine"}
if structured:
    csv.centroids = m.centroids
    v = m.dx * m.dy * m.dz
    csv.volume = np.ones(m.nb) * v
    csv.poro = m.poro
    csv.op_num = m.op_num
    csv.corey = corey
    csv.seal = m.seal
    csv.conn_0 = m.reservoir.conn_0
    csv.conn_1 = m.reservoir.conn_1
    csv.regions = regions
else:
    csv.centroids = m.reservoir.discretizer.centroid_all_cells
    csv.volume = m.reservoir.discretizer.volume_all_cells
    csv.poro = m.reservoir.cell_poro
    csv.op_num = m.op_num
    csv.corey = corey
    csv.seal = m.reservoir.seal
    csv.conn_0 = m.reservoir.conn_0
    csv.conn_1 = m.reservoir.conn_1
    csv.regions = regions

csv.find_output_cells(model_specs['curvature'], cache=1)

# Write to vtk using class methods of discretizer
if not structured:
    # Properties for writing to vtk format:
    vtk_output_dir = output_dir + "/vtk_unstr"
    mesh_vtk_filename = "mesh.vtk"

    mesh_prop = ['permx', 'poro', 'op_num']
    mesh_array = np.zeros((len(m.reservoir.depth), len(mesh_prop)))
    mesh_array[:, 0] = m.reservoir.permx
    mesh_array[:, 1] = m.reservoir.cell_poro
    mesh_array[:, 2] = m.op_num[:-len(m.reservoir.wells)*2]

    m.reservoir.discretizer.write_mesh_to_vtk(vtk_output_dir, mesh_array, mesh_prop, mesh_vtk_filename)

# Time-related properties:
# Specify some other time-related properties (NOTE: all time parameters are in [days])
size_report_step = 12 * 60 / (24 * 60)  # Half Size of the reporting step (when output is writen to .vtk format)
num_report_steps = int(50. / size_report_step)

if 1:
    m.max_dt = 1e-7
    m.run_python_my(1e-6)  # pre-equilibration
    m.save_restart_data('post_equil.pkl')
else:
    m.load_restart_data('post_equil.pkl')

# Output properties and export to vtk
property_array = m.output_properties()
# if structured:
#     m.export_vtk(file_name='solution', local_cell_data={
#         'sat0': property_array[:, n_vars],
#         'xCO2': property_array[:, n_vars + 1],
#         'rho_v': property_array[:, n_vars + 2],
#         'rho_m_Aq': property_array[:, n_vars + 3]
#     })
# else:
#     # Write to vtk using class methods of unstructured discretizer (uses within meshio write to vtk function):
#     m.reservoir.discretizer.write_to_vtk(vtk_output_dir, property_array, output_prop, 0)

# Run over all reporting time-steps:
event1 = True
event2 = True
eps = 1e-6

max_dt = (size_report_step + eps) / 40 / 6
m.max_dt = max_dt
m.params.max_ts = max_dt
m.params.tolerance_newton = 1e-2
inj_comp = [zero]
inj_temp = 320
inj_rate = 5 * 24 * 60 / 1E6
m.physics.set_well_controls(well=m.reservoir.wells[0], is_control=True, is_inj=True, control_type=well_control_iface.VOLUMETRIC_RATE,
                            target=inj_rate, phase_name='V', inj_composition=inj_comp, inj_temp=inj_temp)
first_ts = 1e-8
for i in range(num_report_steps):

    print('\n---------------------------SELF-PRINT #%d---------------------------' % (i + 1))

    if m.physics.engine.t > 0.09375 - eps and event1:
        print("Injector 2 started")
        m.physics.set_well_controls(well=self.reservoir.wells[1], is_control=True, is_inj=True, control_type=well_control_iface.MOLAR_RATE,
                                    target=inj_rate, phase_name='V', inj_composition=self.inj_comp, inj_temp=self.inj_temp)
        event1 = False

    if m.physics.engine.t > 0.20833 - eps and event2:
        print("Both injectors stopped")
        m.physics.set_well_controls(well=self.reservoir.wells[0], is_control=True, is_inj=True, control_type=well_control_iface.MOLAR_RATE,
                                    target=0., phase_name='V', inj_composition=self.inj_comp, inj_temp=self.inj_temp)
        m.physics.set_well_controls(well=self.reservoir.wells[1], is_control=True, is_inj=True, control_type=well_control_iface.MOLAR_RATE,
                                    target=0., phase_name='V', inj_composition=self.inj_comp, inj_temp=self.inj_temp)
        event2 = False
        m.save_restart_data('end_of_injection.pkl')
        m.max_dt = (size_report_step + eps) / 10

    print('Run simulation from %f to %f days with max dt = %f'
          % (m.physics.engine.t, m.physics.engine.t + size_report_step, m.max_dt))
    if m.physics.engine.t < 1:
        max_res = 1e6
    else:
        max_res = 1e10

    m.run(size_report_step, restart_dt=first_ts)
    first_ts = 0

    property_array = m.output_properties()

    if i % 2 != 0.:
        csv.update_time_series(property_array)
        csv.write_time_series()
        csv.ith_step_ts += 1

        if (i + 1) % int(1 / size_report_step) == 0.:
            t = str(24 * int((i + 1) / int(1 / size_report_step))) + 'h'
            csv.write_spatial_map(property_array, t)

        if structured:
            m.export_vtk(file_name='solution', local_cell_data={
                'sat0': property_array[:, n_vars],
                'xCO2': property_array[:, n_vars + 1],
                'rho_v': property_array[:, n_vars + 2],
                'rho_m_Aq': property_array[:, n_vars + 3]
            })
        else:
            # Write to vtk using class methods of unstructured discretizer (uses within meshio write to vtk function):
            m.reservoir.discretizer.write_to_vtk(vtk_output_dir, property_array, output_prop, i + 1)

# Output spatial maps and time series
from visualization.visualize_spatial_maps import visualizeSpatialMaps
visualizeSpatialMaps(output_dir)
from visualization.visualize_time_series import visualizeTimeSeries
visualizeTimeSeries(output_dir)

# After the simulation, print some of the simulation timers and statistics,
# newton iters, etc., how much time spent where:
m.print_timers()
m.print_stat()

time_data = pd.DataFrame.from_dict(m.physics.engine.time_data)
writer = pd.ExcelWriter('time_data.xlsx')
time_data.to_excel(writer, 'Sheet1')
writer.save()
