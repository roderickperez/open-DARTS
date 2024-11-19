from darts.engines import *
from phreeqc_dissolution.operator_evaluator import my_own_acc_flux_etor, my_own_comp_etor, my_own_rate_evaluator

from darts.engines import *
from darts.physics.super.physics import Compositional

import numpy as np
import pickle
import hashlib
import os

# Define our own operator evaluator class
class PhreeqcDissolution(Compositional):
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
        self.rate_operators = my_own_rate_evaluator(self.property_containers[0], self.input_data_struct.temperature)

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
        self.n_axes_points2 = index_vector([self.n_axes_points[0]] * (self.n_vars + 1))
        self.axes_min2 = value_vector([self.axes_min[0]] + [self.axes_min[1]] * self.n_vars)
        self.axes_max2 = value_vector([self.axes_max[0]] + [self.axes_max[1]] * self.n_vars)
        self.comp_itor = self.create_interpolator(evaluator=self.property_operators[region],
                                                  timer_name='comp %d interpolation' % region,
                                                  n_vars=self.n_vars + 1,
                                                  n_ops=2,
                                                  n_axes_points=self.n_axes_points2,
                                                  axes_min=self.axes_min2,
                                                  axes_max=self.axes_max2,
                                                  platform=platform,
                                                  algorithm=itor_type,
                                                  mode=itor_mode,
                                                  precision=itor_precision)

        # ==============================================================================================================
        # Create rate interpolator:
        self.rate_itor = self.create_interpolator(evaluator=self.rate_operators,
                                                  timer_name='rate %d interpolation' % region,
                                                  n_vars=self.n_vars,
                                                  n_ops=self.nph,
                                                  n_axes_points=self.n_axes_points,
                                                  axes_min=self.axes_min,
                                                  axes_max=self.axes_max,
                                                  platform=platform,
                                                  algorithm=itor_type,
                                                  mode=itor_mode,
                                                  precision=itor_precision)

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