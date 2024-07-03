import numpy as np
from darts.engines import operator_set_evaluator_iface, value_vector
from darts.physics.operators_base import OperatorsBase
from darts.physics.super.property_container import PropertyContainer


class OperatorsSuper(OperatorsBase):
    def __init__(self, property_container: PropertyContainer, thermal: bool):
        super().__init__(property_container, thermal)  # Initialize base-class

        self.min_z = property_container.min_z

        # Operator order
        self.ACC_OP = 0  # accumulation operator - ne
        self.FLUX_OP = self.ACC_OP + self.ne  # flux operator - ne * nph
        self.UPSAT_OP = self.FLUX_OP + self.ne * self.nph  # saturation operator (diffusion/conduction term) - nph
        self.GRAD_OP = self.UPSAT_OP + self.nph  # gradient operator (diffusion/conduction term) - ne * nph
        self.KIN_OP = self.GRAD_OP + self.ne * self.nph  # kinetic operator - ne

        # extra operators
        self.RE_INTER_OP = self.KIN_OP + self.ne  # rock internal energy operator - 1
        self.RE_TEMP_OP = self.RE_INTER_OP + 1  # rock temperature operator - 1
        self.ROCK_COND = self.RE_INTER_OP + 2  # rock conduction operator - 1
        self.GRAV_OP = self.RE_INTER_OP + 3  # gravity operator - nph
        self.PC_OP = self.RE_INTER_OP + 3 + self.nph  # capillary operator - nph
        self.PORO_OP = self.RE_INTER_OP + 3 + 2 * self.nph  # porosity operator - 1
        self.n_ops = self.PORO_OP + 1

    def print_operators(self, state: value_vector, values: value_vector):
        """Method for printing operators, grouped"""
        print("================================================")
        print("STATE", state)
        print("ALPHA (accumulation)", values[self.ACC_OP:self.FLUX_OP])
        for j in range(self.nph):
            idx0, idx1 = self.FLUX_OP + j * self.ne, self.FLUX_OP + (j+1) * self.ne
            print("BETA (flux) {}".format(j), values[idx0:idx1])
        print("GAMMA (diffusion)", values[self.UPSAT_OP:self.GRAD_OP])
        for j in range(self.nph):
            idx0, idx1 = self.GRAD_OP + j * self.ne, self.GRAD_OP + (j+1) * self.ne
            print("CHI (diffusion) {}".format(j), values[idx0:idx1])
        print("DELTA (reaction)", values[self.KIN_OP:self.RE_INTER_OP])
        print("GRAVITY", values[self.GRAV_OP:self.PC_OP])
        print("CAPILLARITY", values[self.PC_OP:self.PORO_OP])
        print("POROSITY", values[self.PORO_OP])
        if self.thermal:
            print("ROCK ENERGY, TEMP, COND", values[self.RE_INTER_OP:self.GRAV_OP])
        return


class ReservoirOperators(OperatorsSuper):
    def evaluate(self, state: value_vector, values: value_vector):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        vec_values_as_np = np.asarray(values)
        pressure = vec_state_as_np[0]

        nm = self.property.nm
        nc_fl = self.nc - nm

        vec_values_as_np[:] = 0

        #  some arrays will be reused in thermal
        self.ph, self.sat, self.x, rho, self.rho_m, self.mu, self.kr, pc, mass_source = self.property.evaluate(state)
        self.ph = np.array(self.ph, dtype=np.intp)

        self.compr = self.property.rock_compr_ev.evaluate(pressure)

        density_tot = np.sum(self.sat * self.rho_m)
        zc = np.append(vec_state_as_np[1:self.nc], 1 - np.sum(vec_state_as_np[1:self.nc]))
        phi = 1 - np.sum(zc[nc_fl:self.nc])

        """ CONSTRUCT OPERATORS HERE """

        """ Alpha operator represents accumulation term """
        vec_values_as_np[self.ACC_OP:self.ACC_OP + nc_fl] = self.compr * density_tot * zc[:nc_fl]

        """ and alpha for mineral components """
        vec_values_as_np[self.ACC_OP + nc_fl:self.ACC_OP + self.nc] = self.property.solid_dens * zc[nc_fl:self.nc]

        """ Beta operator represents flux term: """
        for j in self.ph:
            vec_values_as_np[self.FLUX_OP + j * self.ne:self.FLUX_OP + j * self.ne + nc_fl] = self.x[j] * self.rho_m[j] * self.kr[j] / self.mu[j]

        """ Gamma operator for diffusion (same for thermal and isothermal) """
        vec_values_as_np[self.UPSAT_OP + self.ph] = self.compr * self.sat[self.ph]

        """ Chi operator for diffusion """
        for j in self.ph:
            D = self.property.diffusion_ev[self.property.phases_name[j]].evaluate()
            vec_values_as_np[self.GRAD_OP + j * self.ne:self.GRAD_OP + j * self.ne + nc_fl] = D * self.x[j] * self.rho_m[j]

        """ Delta operator for reaction """
        vec_values_as_np[self.KIN_OP:self.KIN_OP + self.nc] = mass_source

        """ Gravity and Capillarity operators """
        # E3-> gravity
        vec_values_as_np[self.GRAV_OP + self.ph] = rho[self.ph]

        # E4-> capillarity
        vec_values_as_np[self.PC_OP + self.ph] = np.array(pc)[self.ph]

        # E5_> porosity
        vec_values_as_np[self.PORO_OP] = phi

        if self.thermal:
            self.evaluate_thermal(state, values)

        # self.print_operators(state, values)

        return 0

    def evaluate_thermal(self, state: value_vector, values: value_vector):
        vec_state_as_np = np.asarray(state)
        vec_values_as_np = np.asarray(values)
        pressure = state[0]
        temperature = vec_state_as_np[-1]

        rock_energy = self.property.rock_energy_ev.evaluate(temperature=temperature)
        enthalpy, cond, energy_source = self.property.evaluate_thermal(state)

        """ Alpha operator represents accumulation term: """
        vec_values_as_np[self.ACC_OP + self.nc] += self.compr * np.sum(self.sat[self.ph] * self.rho_m[self.ph] * enthalpy[self.ph])  # fluid enthalpy (kJ/m3)
        vec_values_as_np[self.ACC_OP + self.nc] -= self.compr * 100 * pressure

        """ Beta operator represents flux term: """
        vec_values_as_np[self.FLUX_OP + self.ph * self.ne + self.nc] = enthalpy[self.ph] * self.rho_m[self.ph] * self.kr[self.ph] / self.mu[self.ph]

        """ Chi operator for temperature in conduction """
        vec_values_as_np[self.GRAD_OP + self.ph * self.ne + self.nc] = temperature * cond[self.ph]

        """ Delta operator for reaction """
        vec_values_as_np[self.KIN_OP + self.nc] = energy_source

        """ Additional energy operators """
        # E1-> rock internal energy
        vec_values_as_np[self.RE_INTER_OP] = rock_energy / self.compr  # (T-T_0), multiplied by rock hcap inside engine
        # E2-> rock temperature
        vec_values_as_np[self.RE_TEMP_OP] = temperature
        # E3-> rock conduction
        vec_values_as_np[self.ROCK_COND] = 1 / self.compr  # multiplied by rock cond inside engine

        return 0

class MassFluxOperators(OperatorsBase):
    def __init__(self, property_container: PropertyContainer, thermal: bool):
        super().__init__(property_container, thermal)  # Initialize base-class

        nm = self.property.nm
        self.nc_fl = self.nc - nm

        self.n_ops = self.nph * self.nc_fl

    def evaluate(self, state: value_vector, values: value_vector):
        for i in range(self.n_ops):
            values[i] = 0

        _, _, x, rho, _, mu, kr, _, _ = self.property.evaluate(state)

        """ Beta operator here represents mass flux term: """
        for j in range(self.nph):
            for i in range(self.nc_fl):
                values[self.nc_fl * j + i] = x[j][i] * rho[j] * kr[j] / mu[j]

class GeomechanicsReservoirOperators(ReservoirOperators):
    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Reservoir operators
        super().evaluate(state, values)

        # Rock density operator
        # TODO: function of matrix pressure = I1 / 3 = (s_xx + s_yy + s_zz) / 3
        values[self.PORO_OP + 1] = self.property.rock_density_ev.evaluate()

        return 0

    def print_operators(self, state, values):
        """Method for printing operators, grouped"""
        super().print_operators(state, values)
        print("ROCK DENSITY", values[self.PORO_OP + 1])
        return


class WellOperators(OperatorsSuper):
    def evaluate(self, state: value_vector, values: value_vector):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]

        nm = self.property.nm
        nc_fl = self.nc - nm

        for i in range(self.n_ops):
            values[i] = 0

        ph, sat, x, rho, rho_m, mu, kr, pc, mass_source = self.property.evaluate(state)

        self.compr = self.property.rock_compr_ev.evaluate(pressure)

        density_tot = np.sum(sat * rho_m)
        zc = np.append(vec_state_as_np[1:self.nc], 1 - np.sum(vec_state_as_np[1:self.nc]))
        phi = 1

        """ CONSTRUCT OPERATORS HERE """

        """ Alpha operator represents accumulation term """
        for i in range(nc_fl):
            values[self.ACC_OP + i] = self.compr * density_tot * zc[i]

        """ and alpha for mineral components """
        for i in range(nm):
            values[self.ACC_OP + nc_fl + i] = self.property.solid_dens[i] * zc[nc_fl + i]

        """ Beta operator represents flux term: """
        for j in ph:
            for i in range(self.nc):
                values[self.FLUX_OP + j * self.ne + i] = x[j][i] * rho_m[j] * kr[j] / mu[j]

        """ Gamma operator for diffusion (same for thermal and isothermal) """

        """ Chi operator for diffusion """

        """ Delta operator for reaction """
        for i in range(self.nc):
            values[self.KIN_OP + i] = mass_source[i]

        """ Gravity and Capillarity operators """
        # E3-> gravity
        for j in range(self.nph):
            values[self.GRAV_OP + j] = rho[j]

        # E5_> porosity
        values[self.PORO_OP] = phi

        if self.thermal:
            self.evaluate_thermal(state, values)

        # self.print_operators(state, values)

        return 0

    def evaluate_thermal(self, state: value_vector, values: value_vector):
        return


class SinglePhaseGeomechanicsOperators(OperatorsBase):
    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """

        _, _, _, rho, _, self.mu, _, _, _ = self.property.evaluate(state)
        values[0] = rho[0]
        values[1] = rho[0] / self.mu[0]

        return 0


class RateOperators(operator_set_evaluator_iface):
    def __init__(self, property_container):
        super().__init__()

        self.nc = property_container.nc
        self.nph = property_container.nph
        self.n_ops = property_container.nph

        self.property = property_container

    def evaluate(self, state: value_vector, values: value_vector):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        for i in range(self.n_ops):
            values[i] = 0

        ph, sat, x, rho, rho_m, mu, kr, pc, mass_source = self.property.evaluate(state)

        flux = np.zeros(self.nc)
        # step-1
        for j in ph:
            for i in range(self.nc):
                flux[i] += rho_m[j] * kr[j] * x[j][i] / mu[j]
        # step-2
        flux_sum = np.sum(flux)

        #(sat_sc, rho_m_sc) = self.property.evaluate_at_cond(1, self.flux/flux_sum)
        sat_sc = sat
        rho_m_sc = rho_m

        # step-3
        total_density = np.sum(sat_sc * rho_m_sc)
        # step-4
        for j in ph:
            values[j] = rho_m[j] * kr[j] / mu[j]
            #sat_sc[j] * flux_sum / total_density

        # print(state, values)
        return 0
