import numpy as np
from darts.engines import *
from property_container import PropertyContainer


class AccFluxGravityEvaluator(operator_set_evaluator_iface):
    def __init__(self, property_container: PropertyContainer):
        super().__init__()  # Initialize base-class

        self.property = property_container

    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]

        nc = self.property.nc
        nph = self.property.nph
        ne = nc

        sat, x, rho, rho_m, mu, kr, FM = self.property.evaluate(state)

        compr = self.property.rock_compr_ev.evaluate(pressure)

        density_tot = np.sum(sat * rho_m)
        zc = np.append(vec_state_as_np[1:nc], 1 - np.sum(vec_state_as_np[1:nc]))

        #       al + bt        + gm + dlt + chi     + rock_temp por    + gr/cap  + por
        total = ne + ne * nph + nph + ne + ne * nph + 3 + 2 * nph + 1

        for i in range(total):
            values[i] = 0

        """ CONSTRUCT OPERATORS HERE """
        """ Alpha operator represents accumulation term: """
        for i in range(nc):
            values[i] = compr * density_tot * zc[i]

        """ Beta operator represents flux term: """
        for j in range(nph):
            shift = ne + ne * j
            for i in range(nc):
                values[shift + i] = x[j][i] * rho_m[j] * kr[j] * FM[j] / mu[j]

        """ Gamma operator for diffusion (same for thermal and isothermal) """
        shift = ne + ne * nph
        for j in range(nph):
            values[shift + j] = compr * sat[j]

        """ Chi operator for diffusion """
        shift += nph
        for i in range(nc):
            for j in range(nph):
                values[shift + i * nph + j] = 0.0001728 * rho_m[j] * x[j][i]

        """ Delta operator for reaction """
        shift += nph * ne
        for i in range(ne):
            values[shift + i] = 0

        """ Gravity and Capillarity operators """
        shift += ne
        # E3-> gravity
        for i in range(nph):
            values[shift + 3 + i] = rho[i]

        # E4-> capillarity
        for i in range(nph):
            values[shift + 3 + nph + i] = 0

        # E5_> porosity
        values[shift + 3 + 2 * nph] = 0

        return 0


class AccFluxGravityWellEvaluator(operator_set_evaluator_iface):
    def __init__(self, property_container):
        super().__init__()  # Initialize base-class

        self.property = property_container

    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]

        nc = self.property.nc
        nph = self.property.nph
        ne = nc

        sat, x, rho, rho_m, mu, kr, FM = self.property.evaluate(state)

        compr = self.property.rock_compr_ev.evaluate(pressure)

        density_tot = np.sum(sat * rho_m)
        zc = np.append(vec_state_as_np[1:nc], 1 - np.sum(vec_state_as_np[1:nc]))

        #       al + bt        + gm + dlt + chi     + rock_temp por    + gr/cap  + por
        total = ne + ne * nph + nph + ne + ne * nph + 3 + 2 * nph + 1

        for i in range(total):
            values[i] = 0

        """ CONSTRUCT OPERATORS HERE """
        """ Alpha operator represents accumulation term: """
        for i in range(nc):
            values[i] = compr * density_tot * zc[i]

        """ Beta operator represents flux term: """
        for j in range(nph):
            shift = ne + ne * j
            for i in range(nc):
                values[shift + i] = x[j][i] * rho_m[j] * kr[j] / mu[j]

        """ Gamma operator for diffusion (same for thermal and isothermal) """
        shift = ne + ne * nph
        for j in range(nph):
            values[shift + j] = compr * sat[j]

        """ Chi operator for diffusion """
        shift += nph
        for j in range(nph):
            for i in range(nc):
                values[shift + ne * j + i] = 0.0001728 * rho_m[j] * x[j][i]

        """ Delta operator for reaction """
        shift += nph * ne
        for i in range(ne):
            values[shift + i] = 0

        """ Gravity and Capillarity operators """
        shift += ne
        # E3-> gravity
        for i in range(nph):
            values[shift + 3 + i] = 0

        # E4-> capillarity
        for i in range(nph):
            values[shift + 3 + nph + i] = 0
        # E5_> porosity
        values[shift + 3 + 2 * nph] = 0

        return 0


class RateEvaluator(operator_set_evaluator_iface):
    def __init__(self, property_container: PropertyContainer):
        super().__init__()  # Initialize base-class

        self.property = property_container

    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        sat, xx, rho, rho_m, mu, kr, FM = self.property.evaluate(state)

        """ CONSTRUCT RATE OPERATORS HERE """
        num_rate_op = self.property.nc  # two p two c just for this case, otherwise need to import phases number

        # step-1
        flux = np.zeros((num_rate_op, 1))
        for i in range(num_rate_op):
            flux[i] = rho_m[0] * kr[0] * xx[0][i] / mu[0] + rho_m[1] * kr[1] * xx[1][i] / mu[1]
        # step-2
        flux_sum = np.sum(flux)
        # step-3
        total_density = sum(sat*rho_m)
        # step-4
        values[0] = sat[0] * flux_sum / total_density
        values[1] = sat[1] * flux_sum / total_density

        return 0


class PropertyEvaluator(operator_set_evaluator_iface):
    def __init__(self, variables, property_container):
        super().__init__()  # Initialize base-class
        self.property = property_container

        self.vars = variables
        self.n_vars = len(self.vars)
        self.props = ['sg', 'x0', 'rhoA', 'rhoV', 'muA']
        self.n_props = len(self.props)

    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        sat, x, rho, rho_m, mu, kr, FM = self.property.evaluate(state)

        values[0] = sat[0]
        values[1] = x[1, 0]
        values[2] = rho[1]
        values[3] = rho[0]
        values[4] = mu[1]

        return 0
