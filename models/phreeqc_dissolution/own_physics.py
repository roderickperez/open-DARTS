from darts.engines import *
from own_operator_evaluator import my_own_acc_flux_etor, my_own_rate_evaluator, my_own_comp_etor, my_own_results_etor
from own_properties import *

from darts.engines import *
from darts.physics.physics_base import PhysicsBase

import numpy as np
import pickle
import hashlib
import os

# Define our own operator evaluator class
class OwnPhysicsClass(PhysicsBase):
    def __init__(self, timer, components, n_points, min_p, max_p, min_z, input_data_struct, properties,
                 platform='cpu', itor_type='multilinear', itor_mode='adaptive', itor_precision='d', cache=True):
        # Obtain properties from user input during initialization:
        self.timer = timer.node["simulation"]
        self.n_points = n_points
        self.min_p = min_p
        self.max_p = max_p
        self.min_z = min_z
        self.components = components
        self.n_components = len(components)
        NE = self.n_components
        self.vars = ["p"] + components[:-1]
        self.thermal = False
        self.phases = ['vapor', 'liquid']
        self.n_phases = len(self.phases)
        self.n_vars = self.n_components
        n_axes_points = index_vector([n_points] * self.n_vars)
        self.n_axes_min = value_vector([min_p] + [min_z] * (self.n_components - 1))
        self.n_axes_max = value_vector([max_p] + [1 - min_z] * (self.n_components - 1))

        # Engine initialization
        # engine_name = eval("engine_nc_kin_dif_cpu%d" % self.nr_components)
        # engine_name = eval("engine_nc_kin_cpu%d" % self.n_components)
        self.n_ops = NE + self.n_phases * NE + self.n_phases + self.n_phases * NE + NE + 3 + 2 * self.n_phases + 1 + self.n_phases

        super().__init__(variables=self.vars,
                         nc=self.n_components,
                         phases=self.phases,
                         n_ops=self.n_ops,
                         axes_min=self.n_axes_min,
                         axes_max=self.n_axes_max,
                         n_axes_points=n_axes_points,
                         timer=timer,
                         cache=cache)

        # acc_flux_itor_name = eval("multilinear_adaptive_cpu_interpolator_i_d_%d_%d" % (self.n_components, self.nr_ops))
        # rate_interpolator_name = eval("multilinear_adaptive_cpu_interpolator_i_d_%d_%d" % (self.n_components, self.n_phases))
        #
        # acc_flux_itor_name_long = eval("multilinear_adaptive_cpu_interpolator_l_d_%d_%d" % (self.n_components, self.nr_ops))
        # rate_interpolator_name_long = eval("multilinear_adaptive_cpu_interpolator_l_d_%d_%d" % (self.n_components, self.n_phases))
        #
        # # Additional itor's
        # comp_itor_name_long = eval("multilinear_adaptive_cpu_interpolator_l_d_%d_%d" % (self.n_components, 2))
        # results_itor_name_long = eval("multilinear_adaptive_cpu_interpolator_l_d_%d_%d" % (self.n_components, self.n_phases))

        # Initialize main evaluator
        self.acc_flux_etor = my_own_acc_flux_etor(input_data_struct, properties)

        # Initialize table entries (nr of points, axis min, and axis max):
        # nr_of_points for [pres, comp1, ..., compN-1]:
        # self.acc_flux_etor.axis_points = index_vector([
        #     self.n_points, self.n_points, self.n_points, self.n_points, self.n_points])
        #
        # # axis_min for [pres, comp1, ..., compN-1]:
        # self.acc_flux_etor.axis_min = value_vector([self.min_p, min_z, min_z, min_z, min_z])
        # # axis_max for [pres, comp1, ..., compN-1]:
        # self.acc_flux_etor.axis_max = value_vector([self.max_p, max_z, max_z, max_z, max_z])

        # Create actual accumulation and flux interpolator:

        self.acc_flux_itor = self.create_interpolator(evaluator=self.acc_flux_etor,
                                                      timer_name='reservoir interpolation',
                                                      n_ops=self.n_ops, platform=platform,
                                                      algorithm=itor_type, mode=itor_mode,
                                                      precision=itor_precision)

        # ==============================================================================================================

        # Create initialization evaluator
        self.comp_etor = my_own_comp_etor(input_data_struct.pressure_init, properties.init_flash_ev)

        self.n_axes_points2 = index_vector([2 * n_points] * self.n_vars)
        self.n_axes_min2 = value_vector([0] + [min_z] * (self.n_components - 1))
        self.n_axes_max2 = value_vector([1] + [1 - min_z] * (self.n_components - 1))

        self.comp_itor = self.create_interpolator_old(self.comp_etor, self.n_vars, 2, self.n_axes_points2,
                                                  self.n_axes_min2, self.n_axes_max2, platform=platform,
                                                  algorithm=itor_type, mode=itor_mode,
                                                  precision=itor_precision)

        # ==============================================================================================================

        # Create results evaluator
        self.results_etor = my_own_results_etor(input_data_struct, properties)

        # Initialize results interpolator
        self.results_itor = self.create_interpolator_old(self.results_etor, self.n_vars, self.n_phases, self.n_axes_points2,
                                                     self.n_axes_min, self.n_axes_max, platform=platform,
                                                     algorithm=itor_type, mode=itor_mode,
                                                     precision=itor_precision)

        # ==============================================================================================================

        # Create rate evaluator and interpolator:
        self.rate_etor = my_own_rate_evaluator(properties, input_data_struct.temperature, input_data_struct.c_r)

        self.rate_itor = self.create_interpolator_old(self.rate_etor, self.n_vars, self.n_phases, self.n_axes_points,
                                                  self.n_axes_min, self.n_axes_max, platform=platform,
                                                  algorithm=itor_type, mode=itor_mode,
                                                  precision=itor_precision)

        self.create_itor_timers(self.acc_flux_itor, 'reservoir interpolation')
        self.create_itor_timers(self.comp_itor, 'comp interpolation')
        self.create_itor_timers(self.results_itor, 'results interpolation')
        self.create_itor_timers(self.rate_itor, 'rate interpolation')

        # define well control factories
        # Injection wells (upwind method requires both bhp and inj_stream for bhp controlled injection wells):
        self.new_bhp_inj = lambda bhp, inj_stream: bhp_inj_well_control(bhp, value_vector(inj_stream))
        self.new_rate_gas_inj = lambda rate, inj_stream: rate_inj_well_control(self.phases, 0, self.n_components,
                                                                               self.n_components, rate,
                                                                               value_vector(inj_stream), self.rate_itor)
        self.new_rate_oil_inj = lambda rate, inj_stream: rate_inj_well_control(self.phases, 1, self.n_components,
                                                                               self.n_components, rate,
                                                                               value_vector(inj_stream), self.rate_itor)
        # Production wells:
        self.new_bhp_prod = lambda bhp: bhp_prod_well_control(bhp)
        self.new_rate_gas_prod = lambda rate: rate_prod_well_control(self.phases, 0, self.n_components,
                                                                     self.n_components,
                                                                     rate, self.rate_itor)
        self.new_rate_oil_prod = lambda rate: rate_prod_well_control(self.phases, 1, self.n_components,
                                                                     self.n_components,
                                                                     rate, self.rate_itor)

        self.new_acc_flux_itor = lambda new_acc_flux_etor: \
            acc_flux_itor_name(new_acc_flux_etor, self.acc_flux_etor.axis_points,
                               self.acc_flux_etor.axis_min, self.acc_flux_etor.axis_max)

    # Define some class methods:
    def init_wells(self, wells):
        for w in wells:
            assert isinstance(w, ms_well)
            w.init_rate_parameters(self.n_vars, self.n_ops, self.phases, self.rate_itor)

    def set_engine(self, discr_type: str = 'tpfa', platform: str = 'cpu'):
        """
        Function to set :class:`engine_super` object.

        :param discr_type: Type of discretization, 'tpfa' (default) or 'mpfa'
        :type discr_type: str
        :param platform: Switch for CPU/GPU engine, 'cpu' (default) or 'gpu'
        :type platform: str
        """
        if discr_type == 'mpfa':
            if self.thermal:
                return eval("engine_super_mp_%s%d_%d_t" % (platform, self.nc, self.nph))()
            else:
                return eval("engine_super_mp_%s%d_%d" % (platform, self.nc, self.nph))()
        else:
            if self.thermal:
                return eval("engine_super_%s%d_%d_t" % (platform, self.nc, self.nph))()
            else:
                return eval("engine_super_%s%d_%d" % (platform, self.nc, self.nph))()

    def set_boundary_conditions(self, mesh, uniform_pressure, uniform_composition):
        assert isinstance(mesh, conn_mesh)

        # Class methods which can create constant pressure and composition boundary condition:
        pressure = np.array(mesh.pressure, copy=False)
        pressure.fill(uniform_pressure)

        mesh.composition.resize(mesh.n_blocks * (self.n_components - 1))
        composition = np.array(mesh.composition, copy=False)
        for c in range(self.n_components - 1):
            composition[c::(self.n_components - 1)] = uniform_composition[c]

    def create_interpolator_old(self, evaluator: operator_set_evaluator_iface, n_dims: int, n_ops: int,
                            axes_n_points: index_vector, axes_min: value_vector, axes_max: value_vector,
                            algorithm: str = 'multilinear', mode: str = 'adaptive',
                            platform: str = 'cpu', precision: str = 'd'):
        # verify then inputs are valid
        assert len(axes_n_points) == n_dims
        assert len(axes_min) == n_dims
        assert len(axes_max) == n_dims
        for n_p in axes_n_points:
            assert n_p > 1

        # calculate object name using 32 bit index type (i)
        itor_name = "%s_%s_%s_interpolator_i_%s_%d_%d" % (algorithm,
                                                          mode,
                                                          platform,
                                                          precision,
                                                          n_dims,
                                                          n_ops)
        itor = None
        general = False
        cache_loaded = 0
        # try to create itor with 32-bit index type first (kinda a bit faster)
        try:
            itor = eval(itor_name)(evaluator, axes_n_points, axes_min, axes_max)
        except (ValueError, NameError):
            # 32-bit index type did not succeed: either total amount of points is out of range or has not been compiled
            # try 64 bit now raising exception this time if goes wrong:
            itor_name = itor_name.replace('interpolator_i', 'interpolator_l')
            try:
                itor = eval(itor_name)(evaluator, axes_n_points, axes_min, axes_max)
            except (ValueError, NameError):
                # if 64-bit index also failed, probably the combination of required n_ops and n_dims
                # was not instantiated/exposed. In this case substitute general implementation of interpolator
                itor = eval("multilinear_adaptive_cpu_interpolator_general")(evaluator, axes_n_points, axes_min, axes_max,
                                                                             n_dims, n_ops)
                general = True

        if self.cache:
            # create unique signature for interpolator
            itor_cache_signature = "%s_%s_%s_%d_%d" % (type(evaluator).__name__, mode, precision, n_dims, n_ops)
            # geenral itor has a different point_data format
            if general:
                itor_cache_signature += "_general_"
            for dim in range(n_dims):
                itor_cache_signature += "_%d_%e_%e" % (axes_n_points[dim], axes_min[dim], axes_max[dim])
            # compute signature hash to uniquely identify itor parameters and load correct cache
            itor_cache_signature_hash = str(hashlib.md5(itor_cache_signature.encode()).hexdigest())
            itor_cache_filename = 'obl_point_data_' + itor_cache_signature_hash + '.pkl'

            # if cache file exists, read it
            if os.path.exists(itor_cache_filename):
                with open(itor_cache_filename, "rb") as fp:
                    print("Reading cached point data for ", type(itor).__name__)
                    itor.point_data = pickle.load(fp)
                    cache_loaded = 1
            if mode == 'adaptive':
                # for adaptive itors, delay obl data save moment, because
                # during simulations new points will be evaluated.
                # on model destruction (or interpreter exit), itor point data will be written to disk
                self.created_itors.append((itor, itor_cache_filename))

        itor.init()
        # for static itors, save the cache immediately after init, if it has not been already loaded
        # otherwise, there is no point to save the same data over and over
        if self.cache and mode == 'static' and not cache_loaded:
            with open(itor_cache_filename, "wb") as fp:
                print("Writing point data for ", type(itor).__name__)
                pickle.dump(itor.point_data, fp, protocol=4)
        return itor

    def set_interpolators(self, platform='cpu', itor_type='multilinear', itor_mode='adaptive', itor_precision='d',
                          is_barycentric: bool=False):
        pass
