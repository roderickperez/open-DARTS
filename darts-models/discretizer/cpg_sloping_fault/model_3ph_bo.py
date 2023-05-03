from darts.models.physics_sup.physics_comp_sup import Compositional
from darts.models.darts_model import DartsModel
from darts.engines import sim_params
import numpy as np
from darts.models.physics_sup.properties_black_oil import *
from darts.models.physics_sup.property_container import *

import os

# Model class creation here!
class Model(DartsModel):
    def __init__(self, pvt):
        # Call base class constructor
        super().__init__()

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.zero = 1e-12
        self.thermal = 0
        """Physical properties"""
        # Create property containers:
        self.pvt = pvt
        self.property_container = model_properties(phases_name=['gas', 'oil', 'wat'],
                                                   components_name=['g', 'o', 'w'],
                                                   pvt=self.pvt, min_z=self.zero / 10)

        self.phases = self.property_container.phases_name

        """ properties correlations """
        self.property_container.flash_ev = flash_black_oil(self.pvt)
        self.property_container.density_ev = dict([('gas', DensityGas(self.pvt)),
                                                   ('oil', DensityOil(self.pvt)),
                                                   ('wat', DensityWat(self.pvt))])
        self.property_container.viscosity_ev = dict([('gas', ViscGas(self.pvt)),
                                                     ('oil', ViscOil(self.pvt)),
                                                     ('wat', ViscWat(self.pvt))])
        self.property_container.rel_perm_ev = dict([('gas', GasRelPerm(self.pvt)),
                                                    ('oil', OilRelPerm(self.pvt)),
                                                    ('wat', WatRelPerm(self.pvt))])
        self.property_container.capillary_pressure_ev = dict([('pcow', CapillaryPressurePcow(self.pvt)),
                                                              ('pcgo', CapillaryPressurePcgo(self.pvt))])

        self.property_container.rock_compress_ev = RockCompactionEvaluator(self.pvt)

        """ Activate physics """
        self.physics = Compositional(self.property_container, self.timer, n_points=500, min_p=1, max_p=200,
                                min_z=self.zero / 10, max_z=1 - self.zero / 10)

        self.inj_stream = [1 - 2e-8, 1e-8]
        # initial composition should be backtracked from saturations
        self.ini_stream = [0.001225901537, 0.7711341309]

        # Some newton parameters for non-linear solution:
        self.params.first_ts = 0.0001
        self.params.mult_ts = 2
        self.params.max_ts = 2
        self.params.tolerance_newton = 1e-3
        self.params.tolerance_linear = 1e-3

        self.params.max_i_newton = 10
        self.params.max_i_linear = 50

        self.runtime = 2000

        self.timer.node["initialization"].stop()

    # Initialize reservoir and set boundary conditions:
    def set_initial_conditions(self):
        # initial composition should be backtracked from saturations
        self.ini_stream = [0.001225901537, 0.7711341309]
        self.physics.set_uniform_initial_conditions(self.reservoir.mesh, 200, self.ini_stream)

    def properties(self, state):
        (sat, x, rho, rho_m, mu, kr, ph) = self.property_container.evaluate(state)
        return sat[0]
class model_properties(property_container):
    def __init__(self, phases_name, components_name, pvt, min_z=1e-11):
        # Call base class constructor
        self.nph = len(phases_name)
        Mw = np.ones(self.nph)
        self.pvt = pvt
        super().__init__(phases_name, components_name, Mw, min_z)
        self.pvt = pvt
        self.surf_dens = get_table_keyword(self.pvt, 'DENSITY')[0]
        self.surf_oil_dens = self.surf_dens[0]
        self.surf_wat_dens = self.surf_dens[1]
        self.surf_gas_dens = self.surf_dens[2]

        self.x = np.zeros((self.nph, self.nc))

    def evaluate(self, state):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]

        zc = np.append(vec_state_as_np[1:], 1 - np.sum(vec_state_as_np[1:]))

        if zc[-1] < 0:
            # print(zc)
            zc = self.comp_out_of_bounds(zc)

        self.clean_arrays()
        # two-phase flash - assume water phase is always present and water component last
        (xgo, V, pbub) = self.flash_ev.evaluate(pressure, zc)
        for i in range(self.nph):
            self.x[i, i] = 1

        if V < 0:
            ph = [1, 2]
        else:  # assume oil and water are always exists
            self.x[1][0] = xgo
            self.x[1][1] = 1 - xgo
            ph = [0, 1, 2]

        for j in ph:
            M = 0
            # molar weight of mixture
            for i in range(self.nc):
                M += self.Mw[i] * self.x[j][i]
            self.dens[j] = self.density_ev[self.phases_name[j]].evaluate(pressure, pbub, xgo)  # output in [kg/m3]
            self.dens_m[j] = self.dens[j] / M
            self.mu[j] = self.viscosity_ev[self.phases_name[j]].evaluate(pressure, pbub)  # output in [cp]

        self.nu[2] = zc[2]
        # two phase undersaturated condition
        if pressure > pbub:
            self.nu[0] = 0
            self.nu[1] = zc[1]
        else:
            self.nu[1] = zc[1] / (1 - xgo)
            self.nu[0] = 1 - self.nu[1] - self.nu[2]

        self.compute_saturation(ph)

        for j in ph:
            self.kr[j] = self.rel_perm_ev[self.phases_name[j]].evaluate(self.sat[0], self.sat[2])

        pcow = self.capillary_pressure_ev['pcow'].evaluate(self.sat[2])
        pcgo = self.capillary_pressure_ev['pcgo'].evaluate(self.sat[0])

        self.pc = np.array([-pcgo, 0, pcow])

        return self.sat, self.x, self.dens, self.dens_m, self.mu, self.kr, self.pc, ph

    def evaluate_at_cond(self, pressure, zc):

        self.sat[:] = 0

        if zc[-1] < 0:
            # print(zc)
            zc = self.comp_out_of_bounds(zc)

        ph = []
        for j in range(self.nph):
            if zc[j] > self.min_z:
                ph.append(j)
            self.dens_m[j] = self.density_ev[self.phases_name[j]].dens_sc

        self.nu = zc
        self.compute_saturation(ph)


        return self.sat, self.dens_m
