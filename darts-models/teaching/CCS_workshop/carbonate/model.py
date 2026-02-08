from darts.engines import value_vector, sim_params
#from darts.physics.geothermal.geothermal import Geothermal
from darts.models.cicd_model import DartsModel
from darts.physics.properties.iapws.iapws_property_vec import enthalpy_to_temperature
from darts.reservoirs.unstruct_reservoir import UnstructReservoir
from darts.engines import well_control_iface

from dataclasses import dataclass
from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer
from darts.physics.properties.basic import ConstFunc
from darts.physics.properties.density import Garcia2001
from darts.physics.properties.viscosity import Fenghour1998, Islam2012
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy
from dartsflash.libflash import NegativeFlash, FlashParams, InitialGuess
from dartsflash.libflash import CubicEoS, AQEoS
from dartsflash.components import CompData

from scipy.special import erf

import matplotlib.pyplot as plt
import os
import numpy as np
import meshio
from darts.input.input_data import InputData

def fmt(x):
    return '{:.3}'.format(x)


@dataclass
class Corey:
    nw: float
    ng: float
    swc: float
    sgc: float
    krwe: float
    krge: float
    labda: float
    p_entry: float
    pcmax: float
    c2: float
    def modify(self, std, mult):
        i = 0
        for attr, value in self.__dict__.items():
            if attr != 'type':
                setattr(self, attr, value * (1 + mult[i] * float(getattr(std, attr))))
            i += 1

    def random(self, std):
        for attr, value in self.__dict__.items():
            if attr != 'type':
                std_in = value * float(getattr(std, attr))
                param = np.random.normal(value, std_in)
                if param < 0:
                    param = 0
                setattr(self, attr, param)



# Here the Model class is defined (child-class from DartsModel) in which most of the data and properties for the
# simulation are defined, e.g. for the reservoir/physics/sim_parameters/etc.
class Model(DartsModel):
    def __init__(self, idata : InputData):
        # base class constructor
        super().__init__()
        self.idata = idata
        self.zero = 1e-10

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.bound_cond = idata.geom['well_loc_type']

        # Some permeability input data for the simulation
        poro = idata.rock.porosity  # Matrix porosity [-]
        frac_aper = idata.geom['frac_aper']  # Aperture of fracture cells (but also takes a list of apertures for each segment) [m]

        self.inj_well_coords = idata.geom['inj_well_coords']
        self.prod_well_coords = idata.geom['prod_well_coords']

        fname = '_' + idata.geom['mesh_prefix'] + '_' + str(idata.geom['char_len']) + '.msh'
        mesh_file = os.path.join('meshes_' + idata.geom['case_name'], idata.geom['case_name'] + fname)

        if idata.rock.perm_file is not None: # set heterogeneous permeability from a file
            permx = self.get_perm_unstr_from_struct_grid(idata.rock.perm_file, self.input_data)
            permy = permx
            permz = permx * 0.1
        else: # set homogeneous permeability
            permx = idata.rock.permx
            permy = idata.rock.permy
            permz = idata.rock.permz

        # initialize reservoir
        self.reservoir = UnstructReservoir(timer=self.timer, mesh_file=mesh_file,
                                      permx=permx, permy=permy, permz=permz,
                                      poro=poro,
                                      rcond=idata.rock.conductivity,
                                      hcap=idata.rock.heat_capacity,
                                      frac_aper=frac_aper)

        # parameters for fracture aperture computation depending on principal stresses
        if 'Sh_max' in idata.stress:
            self.reservoir.sh_max = idata.stress['Sh_max']
            self.reservoir.sh_min = idata.stress['Sh_min']
            self.reservoir.sh_max_azimuth = idata.stress['SHmax_azimuth']
            self.reservoir.sigma_c = idata.stress['sigma_c']

        # read mesh to get the number of fractures for tags specification
        # assume mesh is extruded and fractures have a quad shape
        # fracture tags start from 90000 according to .geo file generation code
        msh = meshio.read(mesh_file)
        c = msh.cell_data_dict['gmsh:physical']
        n_fractures = (np.unique(c['quad']) >= 90000).sum()
        n_fractures = n_fractures * (1 + int(idata.geom['overburden_layers']>0) + int(idata.geom['underburden_layers']>0))

        # 9991 - rsv, 9992 - overburden, 9993 - underburden, 9994 - overburden2, 9995 - underburden2
        self.reservoir.physical_tags['matrix'] = [9991 + i for i in range(5)]
        # multiplied by 3 because physical surfaces for fracture are also in underburden and overburden
        self.reservoir.physical_tags['fracture'] = [90000 + i for i in range(n_fractures)]

        self.reservoir.physical_tags['boundary'] = [2, 1, 3, 4, 5, 6]  # order: Z- (bottom); Z+ (top) ; Y-; X+; Y+; X-

        '''     matrix_tag   surface_tag                             fracture_tag    test_case
                ----------      2     overburden2 top                                     }
                | 9994                    overburden2                                     }
                ----------      2     overburden top       ------------- 90003        }   }
                | 9992                    overburden       | FRACTURE  |              }   }case_1_burden_2
                ----------      2     reservoir top        |-----------| 90001    }   }case_1_burden
                | 9991                    RESERVOIR        | FRACTURE  | 90000    }case_1 }
                ----------      1     reservoir bottom     |-----------| 90002    }   }   }
                | 9993                    underburden      | FRACTURE  |              }   }
                ----------      1     underburden bottom   ------------- 90004        }   }
                | 9995                    underburden2                                    }
                ----------      1     underburden2 bottom                                 }
        '''

        # discretize
        print('Discretize...')
        self.reservoir.init_reservoir(verbose=True)
        print('Done')

        def find_nearest(array, value):
            distances = np.linalg.norm(array - value, axis=1)
            idx = np.argmin(distances)
            return array[idx], idx

        self.reservoir.op_num = np.zeros(self.reservoir.mesh.n_res_blocks)
        mask = []
        array = self.reservoir.discretizer.centroid_all_cells
        for target in self.reservoir.discretizer.frac_cell_info_dict:
            value = self.reservoir.discretizer.frac_cell_info_dict[target].centroid
            rvalue, idx = find_nearest(array, value)
            mask.append(idx)
            self.reservoir.op_num[idx] = 1

        np.array(self.reservoir.mesh.op_num, copy=False)[:] = self.reservoir.op_num

        if True:

            plt.figure(dpi=100)
            plt.title('Centroids')
            plt.scatter(self.reservoir.discretizer.centroid_all_cells[:, 0],
                        self.reservoir.discretizer.centroid_all_cells[:, 1],
                        s=1, label = 'matrix cells, opnum = 0')
            plt.scatter(self.reservoir.discretizer.centroid_all_cells[mask, 0],
                        self.reservoir.discretizer.centroid_all_cells[mask, 1],
                        s = 5, marker = 'x', label = 'fracture cells, opnum = 1')
            plt.legend()
            plt.xlabel('X, m'); plt.ylabel('Y, m')
            plt.savefig('opnum.png')
            plt.close()

        if False: # set boundary volume XY
            print('Set boundary volume...')
            # set boundary volume XY
            bnd_xy_tags = [3, 4, 5, 6]
            boundary_cells = self.reservoir.discretizer.find_cells(bnd_xy_tags, 'face')
            boundary_cells = np.array(boundary_cells) + self.reservoir.discretizer.frac_cells_tot
            bnd_vol = 1e+15  # [m3] - large volume for boundary cells to mimic infinite reservoir
            # for vtk output
            self.reservoir.discretizer.volume_all_cells[boundary_cells] = bnd_vol
            # for engines
            np.array(self.reservoir.mesh.volume, copy=False)[boundary_cells] = bnd_vol
            print('Done')

        # initialize physics
        self.cell_property = ['pressure', 'CO2', 'temperature']
        
        self.idata.obl.n_points = 400
        self.idata.obl.zero = 1e-13
        self.idata.obl.min_p = 0.
        self.idata.obl.max_p = 1000.
        self.idata.obl.min_t = 10.
        self.idata.obl.max_t = 100.
        self.idata.obl.min_z = self.idata.obl.zero
        self.idata.obl.max_z = 1 - self.idata.obl.zero        

        #self.physics = Geothermal(self.idata, self.timer)
        
        self.set_physics()

        # End timer for model initialization:
        self.timer.node["initialization"].stop()



    def set_physics(self):
        # corey_params = Corey(nw=1.5, ng=1.5, swc=0.32, sgc=0.10, krwe=1.0, krge=1.0, labda=2.,
        #                      p_entry=2, pcmax=300, c2=1.5)
        self.salinity = 0
        self.components = ['CO2', 'H2O']

        # Fluid components, ions and solid
        comp_data = CompData(self.components, setprops=True)
        nc, ni = comp_data.nc, comp_data.ni
        # len(components)
        flash_params = FlashParams(comp_data)
        flash_params.add_eos("PR", CubicEoS(comp_data, CubicEoS.PR))
        flash_params.add_eos("AQ", AQEoS(comp_data, {AQEoS.CompType.water: AQEoS.Jager2003,
                                                     AQEoS.CompType.solute: AQEoS.Ziabakhsh2012,
                                                     AQEoS.CompType.ion: AQEoS.Jager2003
                                                     }))
        pr = flash_params.eos_params["PR"].eos
        aq = flash_params.eos_params["AQ"].eos
        flash_params.eos_order = ["PR", "AQ"]
        phases = ["V", "Aq"]

        # Flash-related parameters
        # flash_params.split_switch_tol = 1e-3

        state_spec = Compositional.StateSpecification.P

        self.physics = Compositional(self.components, phases, timer=self.timer, n_points=self.idata.obl.n_points,
                                     min_p=self.idata.obl.min_p, max_p=self.idata.obl.max_p,
                                     min_z=self.idata.obl.min_z, max_z=self.idata.obl.max_z,
                                     state_spec=state_spec, cache=False)
        #self.physics.n_axes_points[0] = 1001  # sets OBL points for pressure

        self.physics.dispersivity = {}

        diff_w = 1e-9 * 86400
        diff_g = 2e-8 * 86400

        regions = {'matrix cells': 0, 'fractured cells' : 1}

        corey_params = {
            'matrix cells' : Corey(nw=1.5, ng=1.5, swc=0.32, sgc=0.10, krwe=1.0, krge=1.0, labda=2.,
                             p_entry=2, pcmax=300, c2=2.5),
            'fractured cells' : Corey(nw=1., ng=1., swc=0.0001, sgc=0.0001, krwe=1.0, krge=1.0, labda=2.,
                                        p_entry=2, pcmax=300, c2=1.5) # capillary pressure parameters are not actually used
        }

        for i, region in enumerate(regions):
            property_container = PropertyContainer(components_name=self.components, phases_name=phases, Mw=comp_data.Mw,
                                                   min_z=self.zero, temperature=350)

            # property_container.flash_ev = ConstantK(nc=2, ki=[0.001, 100])
            property_container.flash_ev = NegativeFlash(flash_params, ["PR", "AQ"], [InitialGuess.Henry_VA])
            property_container.density_ev = dict([('V', EoSDensity(eos=pr, Mw=comp_data.Mw)),
                                                  ('Aq', Garcia2001(self.components)), ])
            property_container.viscosity_ev = dict([('V', Fenghour1998()),
                                                    ('Aq', Islam2012(self.components)), ])
            property_container.diffusion_ev = dict([('V', ConstFunc(np.ones(nc) * diff_g)),
                                                    ('Aq', ConstFunc(np.ones(nc) * diff_w))])
            property_container.enthalpy_ev = dict([('V', EoSEnthalpy(eos=pr)),
                                                   ('Aq', EoSEnthalpy(eos=aq)), ])
            property_container.conductivity_ev = dict([('V', ConstFunc(8.4)),
                                                       ('Aq', ConstFunc(170.)), ])

            if region == 'matrix cells':
                property_container.rel_perm_ev = dict([('V', ModBrooksCorey(corey_params[region], 'V')),
                                                       ('Aq', ModBrooksCorey(corey_params[region], 'Aq'))])
                property_container.capillary_pressure_ev = ModCapillaryPressure(corey_params[region])
            else:
                # !! no capillary pressure in fractured cells !!
                property_container.rel_perm_ev = dict([('V', ModBrooksCorey(corey_params[region], 'V')),
                                                       ('Aq', ModBrooksCorey(corey_params[region], 'Aq'))])

            self.physics.add_property_region(property_container, i)

            property_container.output_props = {"satV": lambda ii=i: self.physics.property_containers[ii].sat[0],
                                               "rhoV": lambda ii=i: self.physics.property_containers[ii].dens[0],
                                               "rho_mA": lambda ii=i: self.physics.property_containers[ii].dens_m[1],
                                               "enthV": lambda ii=i: self.physics.property_containers[ii].enthalpy[0]}

            for j, phase_name in enumerate(phases):
                for c, component_name in enumerate(self.components):
                    key = f"x{component_name}" if phase_name == 'Aq' else f"y{component_name}"
                    property_container.output_props[key] = lambda ii=i, jj=j, cc=c : self.physics.property_containers[ii].x[jj, cc]

        self.physics.dispersivity[0] = np.zeros((self.physics.nph, self.physics.nc))


    def print_range(self, time, part='cells'):
        depth = np.array(self.reservoir.mesh.depth, copy=True)
        start, end = self.get_mat_frac_range(part)
        D = depth[start:end]
        P = self.get_pressure(part)
        T = self.get_temperature(part)
        suf = '(MAT)'  # matrix cells
        if part == 'full':
            suf = '(MAT+FRAC)'  # matrix+fracture cells
        elif part == 'fracs':
            suf = '(FRAC)'  # matrix+fracture cells
        print('Time', fmt(time/365.25), ' years; ', time, 'days, '
              'D_range:', fmt(D.min()), '-', fmt(D.max()), 'm; ',
              'P_range:', fmt(P.min()), '-', fmt(P.max()), 'bars; ',
              'T_range:', fmt(T.min()), '-', fmt(T.max()), 'K', suf)

    def set_initial_conditions(self):
        if self.idata.initial.type == 'gradient':
            # Specify reference depth, values and gradients to construct depth table in super().set_initial_conditions()
            input_depth = [0., np.amax(self.reservoir.mesh.depth)]
            input_distribution = {'pressure': [1., 1. + input_depth[1] * self.idata.initial.pressure_gradient / 1000],
                                  'CO2': [0.01, 0.01]
                                  }
            return self.physics.set_initial_conditions_from_depth_table(self.reservoir.mesh,
                                                                        input_distribution=input_distribution,
                                                                        input_depth=input_depth)
        elif self.idata.initial.type == 'uniform':
            input_distribution = {'pressure': self.idata.initial.initial_pressure,
                                  'CO2': self.zero}
            return self.physics.set_initial_conditions_from_array(self.reservoir.mesh,
                                                                  input_distribution=input_distribution)

    def well_is_inj(self, wname : str):  # determine well control by its name
        return "I" in wname

    def set_well_controls(self):
        wctrl = self.idata.well_data.controls
        inj_rate = wctrl.inj_rate
        prod_rate = wctrl.prod_rate
        
        inj_comp = value_vector([0.9])

        P = self.get_pressure('full')
        T = self.get_temperature('full')
        if P.size:
            self.pressure_initial_mean = P.mean()
            self.temperature_initial_mean = T.mean()
        else:
            self.pressure_initial_mean = self.idata.initial.initial_pressure
            self.temperature_initial_mean = self.idata.initial.initial_temperature

        if True:
            inj_bhp = self.pressure_initial_mean + wctrl.delta_p_inj
            inj_temp = self.temperature_initial_mean - wctrl.delta_temp
            prod_bhp = self.pressure_initial_mean - wctrl.delta_p_prod
        else:  # if engine is not initialized yet, set well control rate=0
            inj_temp = 0
            inj_rate = 0
            prod_rate = 0

        for i, w in enumerate(self.reservoir.wells):
            if self.well_is_inj(w.name):
                if inj_rate is None:
                    # w.control = self.physics.new_bhp_water_inj(inj_bhp, inj_temp)
                    self.physics.set_well_controls(wctrl=w.control,
                                                   control_type = well_control_iface.BHP,
                                                   is_inj = True,
                                                   target = inj_bhp,
                                                   inj_composition=inj_comp)
                else:
                    # w.control = self.physics.new_rate_water_inj(inj_rate, inj_temp)
                    self.physics.set_well_controls(wctrl=w.control,
                                                   control_type = well_control_iface.VOLUMETRIC_RATE,
                                                   is_inj = True,
                                                   target = inj_rate,
                                                   phase_name = 'water',
                                                   inj_composition=inj_comp)
                    # w.constraint = self.physics.new_bhp_water_inj(wctrl.inj_bhp_constraint, inj_temp)
                    self.physics.set_well_controls(wctrl=w.constraint,
                                                   control_type = well_control_iface.BHP,
                                                   is_inj = True,
                                                   target = wctrl.inj_bhp_constraint,
                                                   inj_temp = inj_temp)
            else:
                if prod_rate is None:
                    # w.control = self.physics.new_bhp_prod(prod_bhp)
                    self.physics.set_well_controls(wctrl=w.control,
                                                   control_type=well_control_iface.BHP,
                                                   is_inj=False,
                                                   target=prod_bhp)
                else:
                    # w.control = self.physics.new_rate_water_prod(prod_rate)
                    self.physics.set_well_controls(wctrl=w.control,
                                                   control_type=well_control_iface.VOLUMETRIC_RATE,
                                                   is_inj=False,
                                                   target = prod_rate)
                    # w.constraint = self.physics.new_bhp_prod(wctrl.prod_bhp_constraint)
                    self.physics.set_well_controls(wctrl=w.constraint,
                                                   control_type=well_control_iface.BHP,
                                                   is_inj = False,
                                                   target = wctrl.prod_bhp_constraint)

            # print(w.name,
            #       w.well_head_depth,
            #       w.control.target_pressure if hasattr(w.control, 'target_pressure') else '',
            #       w.control.target_temperature if hasattr(w.control, 'target_temperature') else '',
            #       w.control.target_rate if hasattr(w.control, 'target_rate') else '')
        return 0

    def get_mat_frac_range(self, part):
        start = 0
        if part == 'full':
            end = self.reservoir.discretizer.frac_cells_tot + self.reservoir.discretizer.mat_cells_tot
        elif part == 'cells':
            end = self.reservoir.discretizer.mat_cells_tot
            start = self.reservoir.discretizer.frac_cells_tot
        elif part == 'fracs':
            end = self.reservoir.discretizer.frac_cells_tot
        return [start, end]

    def get_pressure(self, part='cells'):
        nvars = 2
        start, end = self.get_mat_frac_range(part)
        Xn = np.array(self.physics.engine.X, copy=True)
        P = Xn[nvars*start:nvars*end:nvars]
        return P

    def get_temperature(self, part='cells'):
        nvars = 2
        start, end = self.get_mat_frac_range(part)
        Xn = np.array(self.physics.engine.X, copy=True)
        T = enthalpy_to_temperature(Xn[nvars*start:nvars*end])
        return T

    def calc_well_loc(self):
        #TODO use idx = self.reservoir.find_cell_index(wc)
        # Store number of control volumes (NOTE: in case of fractures, this includes both matrix and fractures):
        self.nb = self.reservoir.discretizer.volume_all_cells.size
        self.num_frac = self.reservoir.discretizer.frac_cells_tot
        self.num_mat = self.reservoir.discretizer.mat_cells_tot
        if self.bound_cond == 'wells_in_frac':
            offset = 0
            left_int = 0
            right_int = self.num_frac
        elif self.bound_cond == 'wells_in_mat':
            offset = self.num_frac
            left_int = self.num_frac
            right_int = self.num_frac + self.num_mat
        elif self.bound_cond == 'wells_in_nearest_cell':
            offset = 0
            left_int = 0
            right_int = self.num_frac + self.num_mat
        else:
            raise('error: wrong self.bound_cond')

        # Find closest control volume to dummy_well point:
        self.injection_wells = []
        dummy_well_inj = self.inj_well_coords

        self.store_dist_to_well_inj = np.zeros((len(dummy_well_inj),))
        self.store_coord_well_inj = np.zeros((len(dummy_well_inj), 3))
        ii = 0
        for ith_inj in dummy_well_inj:
            dist_to_well_point = np.linalg.norm(self.reservoir.discretizer.centroid_all_cells[left_int:right_int] - ith_inj,
                                                axis=1)
            cell_id = np.argmin(dist_to_well_point) + offset
            self.injection_wells.append(cell_id)

            self.store_coord_well_inj[ii, :] = self.reservoir.discretizer.centroid_all_cells[cell_id]
            self.store_dist_to_well_inj[ii] = np.min(dist_to_well_point)
            ii += 1

        self.production_wells = []
        dummy_well_prod = self.prod_well_coords

        self.store_dist_to_well_prod = np.zeros((len(dummy_well_prod),))
        self.store_coord_well_prod = np.zeros((len(dummy_well_prod), 3))
        ii = 0
        for ith_prod in dummy_well_prod:
            dist_to_well_point = np.linalg.norm(self.reservoir.discretizer.centroid_all_cells[left_int:right_int] - ith_prod,
                                                axis=1)
            cell_id = np.argmin(dist_to_well_point) + offset
            self.production_wells.append(cell_id)

            self.store_coord_well_prod[ii, :] = self.reservoir.discretizer.centroid_all_cells[cell_id]
            self.store_dist_to_well_prod[ii] = np.min(dist_to_well_point)
            ii += 1

        self.well_perf_loc = np.array([self.injection_wells, self.production_wells])

    def set_wells(self, well_index=100):
        """
        Class method which initializes the wells (adding wells and their perforations to the reservoir)
        :return:
        """
        self.calc_well_loc()

        for i in range(len(self.well_perf_loc[0])):
            self.reservoir.add_well(f'I{i + 1}')
            self.reservoir.add_perforation(self.reservoir.wells[-1].name, res_cell_idx=self.well_perf_loc[0][i],
                                 well_index=well_index, well_indexD=0, verbose=True)

        for i in range(len(self.well_perf_loc[1])):
            self.reservoir.add_well(f'P{i + 1}')
            self.reservoir.add_perforation(self.reservoir.wells[-1].name, res_cell_idx=self.well_perf_loc[1][i],
                                 well_index=well_index, well_indexD=0, verbose=True)

    def get_perm_unstr_from_struct_grid(self, perm_file, input_data):
        # Set non-uniform permeability
        if perm_file != None:
            [xx, yy, perm_rect_2d] = np.load(perm_file, allow_pickle=True)
            perm_rect_1d = perm_rect_2d.flatten()  # TODO: check XY-order
            cntr = self.discretizer.centroid_all_cells[self.discretizer.fracture_cell_count:]
            z_middle = input_data['z_top'] + input_data['height_res'] * 0.5  # middle depth of the reservoir
            rect_grid = np.vstack((xx.flatten(), yy.flatten(), np.zeros(xx.flatten().shape) + z_middle)).transpose()
            perm_unstr = np.zeros(cntr.size)
            c_idx = 0
            for c in cntr:
                dist_to_point = np.linalg.norm(c - rect_grid, axis=1)
                cell_id = np.argmin(dist_to_point)
                perm_from_rect = perm_rect_1d.flatten()[cell_id]
                perm_unstr[c_idx] = perm_from_rect
                c_idx += 1
            return perm_unstr

    def plot_rel_perms(self):
        """

        This function accesses the relative permeability and capillary pressure objects per region
        (fractured, matrix cells) and plots these.

        """

        # access and plot relative permeability object in the physics
        N = 100
        sg = np.linspace(0, 1, N)
        regions = {'matrix cells': 0, 'fracture cells': 1}

        plt.figure()
        plt.title('Relative permeabilities per region')
        plt.grid()

        for region_no, region_name in enumerate(regions):
            krg = np.zeros_like(sg)
            krw = np.zeros_like(sg)

            for i, s in enumerate(sg):
                krg[i] = self.physics.property_containers[region_no].rel_perm_ev['V'].evaluate(s)
                krw[i] = self.physics.property_containers[region_no].rel_perm_ev['Aq'].evaluate(1 - s)

            if region_name == 'matrix cells':
                krg_style = {'color': 'g', 'linestyle': '-'}
                krw_style = {'color': 'b', 'linestyle': '-'}
            else:
                krg_style = {'color': 'g', 'linestyle': '--'}
                krw_style = {'color': 'b', 'linestyle': '--'}
            plt.plot(sg, krg, label=region_name + ', krg', **krg_style)
            plt.plot(sg, krw, label=region_name + ', krw', **krw_style)

        plt.legend()
        plt.xlabel('sg')
        plt.ylabel('kr')
        plt.savefig('relperms.png')
        plt.close()

        # access and plot relative permeability object in the physics
        plt.figure()
        plt.title('Capillary pressure per region')
        plt.grid()

        for region_no, region_name in enumerate(regions):
            Pc = np.zeros_like(sg)

            for i, s in enumerate(sg):
                Pc[i] = self.physics.property_containers[region_no].capillary_pressure_ev.evaluate([s, 1 - s])[1]

            plt.plot(1 - sg, Pc, label=region_name)
        plt.legend()
        plt.xlabel('Sw')
        plt.ylabel('Pc')
        plt.savefig('capillary_pressure.png')
        plt.close()

        return 0
            
            
class ModBrooksCorey:
    def __init__(self, corey, phase):

        self.phase = phase

        if self.phase == "Aq":
            self.k_rw_e = corey.krwe
            self.swc = corey.swc
            self.sgc = 0
            self.nw = corey.nw
        else:
            self.k_rg_e = corey.krge
            self.sgc = corey.sgc
            self.swc = 0
            self.ng = corey.ng

    def evaluate(self, sat):
        if self.phase == "Aq":
            Se = (sat - self.swc)/(1 - self.swc - self.sgc)
            if Se > 1:
                Se = 1
            elif Se < 0:
                Se = 0
            k_r = self.k_rw_e * Se ** self.nw
        else:
            Se = (sat - self.sgc) / (1 - self.swc - self.sgc)
            if Se > 1:
                Se = 1
            elif Se < 0:
                Se = 0
            k_r = self.k_rg_e * Se ** self.ng

        return k_r

class ModCapillaryPressure:
    def __init__(self, corey):
        self.swc = corey.swc
        self.p_entry = corey.p_entry
        self.labda = corey.labda
        # self.labda = 3
        self.eps = 1e-10
        self.pcmax = corey.pcmax
        self.c2 = corey.c2

    def evaluate(self, sat):
        sat_w = sat[1]
        # sat_w = sat
        Se = (sat_w - self.swc)/(1 - self.swc)
        if Se < self.eps:
            Se = self.eps
        # pc = self.p_entry * self.eps ** (1/self.labda) * Se ** (-1/self.labda)  # for p_entry to non-wetting phase
        pc_b = self.p_entry * Se ** (-1/self.c2) # basic capillary pressure
        pc = self.pcmax * erf((pc_b * np.sqrt(np.pi)) / (self.pcmax * 2)) # smoothened capillary pressure
        # if Se > 1 - self.eps:
        #     pc = 0

        # pc = self.p_entry
        Pc = np.array([0, pc], dtype=object)  # V, Aq
        return Pc
