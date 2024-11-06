from darts.engines import *
from own_operator_evaluator import my_own_acc_flux_etor, my_own_comp_etor, my_own_rate_evaluator

from darts.engines import *
from darts.physics.super.physics import Compositional

import numpy as np
import pickle
import hashlib
import os

# Define our own operator evaluator class
class OwnPhysicsClass(Compositional):
    def __init__(self, timer, elements, n_points, min_p, max_p, min_z, input_data_struct, properties,
                 platform='cpu', itor_type='multilinear', itor_mode='adaptive', itor_precision='d', cache=True):
        # Obtain properties from user input during initialization:
        self.input_data_struct = input_data_struct
        nc = len(elements)
        NE = nc
        vars = ["p"] + elements[:-1]
        phases = ['vapor', 'liquid']
        n_phases = len(phases)
        n_vars = len(vars)
        n_axes_points = index_vector([n_points] * n_vars)
        n_axes_min = value_vector([min_p] + [min_z] * (nc - 1))
        n_axes_max = value_vector([max_p] + [1 - min_z] * (nc - 1))

        super().__init__(components=elements, phases=phases, n_points=n_points, thermal=False,
                         min_p=min_p, max_p=max_p, min_z=min_z, max_z=1-min_z,
                         axes_min=n_axes_min, axes_max=n_axes_max, n_axes_points=n_axes_points,
                         timer=timer, cache=cache)
        self.vars = vars

    def set_operators(self):
        for region in self.regions:
            self.reservoir_operators[region] = my_own_acc_flux_etor(self.input_data_struct, self.property_containers[region])
            self.property_operators[region] = my_own_comp_etor(self.input_data_struct, self.property_containers[region])
        self.rate_operators = my_own_rate_evaluator(self.property_containers[0], self.input_data_struct.temperature, self.input_data_struct.c_r)

    def set_interpolators(self, platform='cpu', itor_type='multilinear', itor_mode='adaptive',
                          itor_precision='d', is_barycentric: bool = False):
        region = 0

        # Create actual accumulation and flux interpolator:
        self.acc_flux_itor = self.create_interpolator(evaluator=self.reservoir_operators[region],
                                                      timer_name='reservoir interpolation',
                                                      n_ops=self.n_ops, platform=platform,
                                                      algorithm=itor_type, mode=itor_mode,
                                                      precision=itor_precision, is_barycentric=is_barycentric)

        # ==============================================================================================================
        # Create initialization & porosity evaluator
        self.n_axes_points2 = index_vector([2 * self.n_axes_points[0]] * (self.n_vars + 1))
        self.axes_min2 = value_vector([self.axes_min[0]] + [self.axes_min[1]] * self.n_vars)
        self.axes_max2 = value_vector([self.axes_max[0]] + [self.axes_max[1]] * self.n_vars)
        self.comp_itor = self.create_interpolator_old(evaluator=self.property_operators[region],
                                                      n_dims=self.n_vars + 1,
                                                      n_ops=2,
                                                      axes_n_points=self.n_axes_points2,
                                                      axes_min=self.axes_min2,
                                                      axes_max=self.axes_max2,
                                                      platform=platform,
                                                      algorithm=itor_type,
                                                      mode=itor_mode,
                                                      precision=itor_precision)
        self.create_itor_timers(self.comp_itor, 'comp interpolation')

        # ==============================================================================================================
        # Create rate interpolator:
        self.rate_itor = self.create_interpolator_old(evaluator=self.rate_operators,
                                                      n_dims=self.n_vars,
                                                      n_ops=self.nph,
                                                      axes_n_points=self.n_axes_points,
                                                      axes_min=self.axes_min,
                                                      axes_max=self.axes_max,
                                                      platform=platform,
                                                      algorithm=itor_type,
                                                      mode=itor_mode,
                                                      precision=itor_precision)
        self.create_itor_timers(self.rate_itor, 'rate interpolation')

    def define_well_controls(self):
        # define well control factories
        # Injection wells (upwind method requires both bhp and inj_stream for bhp controlled injection wells):
        self.new_bhp_inj = lambda bhp, inj_stream: bhp_inj_well_control(bhp, value_vector(inj_stream))
        self.new_rate_gas_inj = lambda rate, inj_stream: rate_inj_well_control(self.phases, 0, self.nc,
                                                                               self.nc, rate,
                                                                               value_vector(inj_stream), self.rate_itor)
        self.new_rate_oil_inj = lambda rate, inj_stream: rate_inj_well_control(self.phases, 1, self.nc,
                                                                               self.nc, rate,
                                                                               value_vector(inj_stream), self.rate_itor)
        # Production wells:
        self.new_bhp_prod = lambda bhp: bhp_prod_well_control(bhp)
        self.new_rate_gas_prod = lambda rate: rate_prod_well_control(self.phases, 0, self.nc,
                                                                     self.nc,
                                                                     rate, self.rate_itor)
        self.new_rate_oil_prod = lambda rate: rate_prod_well_control(self.phases, 1, self.nc,
                                                                     self.nc,
                                                                     rate, self.rate_itor)

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
