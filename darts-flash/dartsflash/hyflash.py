import numpy as np
import xarray as xr

from dartsflash.dartsflash import DARTSFlash, R
from dartsflash.components import CompData, ConcentrationUnits as cu
from dartsflash.libflash import VdWP
from dartsflash.libflash import RootFinding


class HyFlash(DARTSFlash):
    hydrate_eos: dict = {}

    def add_hydrate_eos(self, name: str, eos: VdWP):
        """
        Method to add hydrate EoS to map
        """
        self.hydrate_eos[name] = eos

    def calc_df(self, pressure, temperature, composition, phase: str = "sI"):
        """
        Method to calculate fugacity difference between fluid mixture and hydrate phase

        :param pressure: Pressure [bar]
        :param temperature: Temperature [K]
        :param composition: Feed mole fractions [-]
        :param phase: Hydrate phase type
        """
        self.f.evaluate(pressure, temperature, composition)
        flash_results = self.f.get_flash_results()
        V = np.array(flash_results.nu)
        x = np.array(flash_results.X).reshape(len(V), self.ns)
        f0 = self.flash_params.eos_params[self.flash_params.eos_order[0]].eos.fugacity(pressure, temperature, x[0, :])

        fwH = self.hydrate_eos[phase].fw(pressure, temperature, f0)
        df = fwH - f0[self.comp_data.H2O_idx]
        return df

    def calc_xH(self, pressure, temperature, x, phase: str = "sI"):
        """
        Method to calculate hydrate composition

        :param pressure: Pressure [bar]
        :param temperature: Temperature [K]
        :param x: Fluid phase compositions [-]
        :param phase: Hydrate phase type
        """
        f0 = self.flash_params.eos_params[self.flash_params.eos_order[0]].eos.fugacity(pressure, temperature, x[0, :])
        fwH = self.hydrate_eos[phase].fw(pressure, temperature, f0)
        return self.hydrate_eos[phase].xH()

    def calc_equilibrium_pressure(self, temperature: float, composition: list, p_init: float = None, phase: str = "sI",
                                  dp: float = 10., min_p: float = 1., max_p: float = 500., tol_f: float = 1e-15, tol_x: float = 1e-15,
                                  verbose: bool = False):
        """
        Method to calculate equilibrium pressure between fluid phases and hydrate phase at given T, z

        :param temperature: Temperature [K]
        :param composition: Feed mole fractions [-]
        :param p_init: Initial guess for equilibrium pressure [bar]
        :param phase: Hydrate phase type
        :param dp: Step size to find pressure bounds
        :param min_p: Minimum pressure [bar]
        :param tol_f: Tolerance for objective function
        :param tol_x: Tolerance for variable
        :param verbose: Switch for verbose output
        """
        # Find bounds for pressure
        p_init = min_p if p_init is None else p_init
        p_min, p_max = p_init, p_init
        if self.calc_df(p_init, temperature, composition, phase) > 0:
            # Hydrate fugacity larger than fluid fugacity
            while True:
                p_max = min(max_p, p_max+dp)
                if self.calc_df(p_max, temperature, composition, phase) < 0:
                    break
                p_min = min(max_p, p_min+dp)
                if p_min == max_p:
                    if verbose:
                        print("Equilibrium pressure above p_max", temperature)
                    return None
        else:
            # Hydrate fugacity smaller than fluid fugacity
            while True:
                p_min = max(min_p, p_min-dp)
                if self.calc_df(p_min, temperature, composition, phase) > 0:
                    break
                p_max = max(min_p, p_max-dp)
                if p_max == min_p:
                    if verbose:
                        print("Equilibrium pressure below p_min", temperature)
                    return None

        pres = (p_min + p_max) / 2

        # Define objective function for Brent's method
        def obj_fun(pres):
            df = self.calc_df(pres, temperature, composition, phase)
            return -df

        rf = RootFinding()
        error = rf.brent(obj_fun, pres, p_min, p_max, tol_f, tol_x)

        if not error == 1:
            return rf.getx()
        else:
            if verbose:
                print("Not converged", temperature)
            return None

    def calc_equilibrium_temperature(self, pressure: float, composition: list, t_init: float = None, phase: str = "sI",
                                     dT: float = 10., min_T: float = 250., tol_f: float = 1e-15, tol_x: float = 1e-15,
                                     verbose: bool = False):
        """
        Method to calculate equilibrium temperature between fluid phases and hydrate phase at given P, z

        :param pressure: Pressure [bar]
        :param composition: Feed mole fractions [-]
        :param t_init: Initial guess for equilibrium temperature [K]
        :param phase: Hydrate phase type
        :param dT: Step size to find pressure bounds
        :param min_T: Minimum temperature [K]
        :param tol_f: Tolerance for objective function
        :param tol_x: Tolerance for variable
        :param verbose: Switch for verbose output
        """
        # Find bounds for temperature
        t_init = min_T if t_init is not None else t_init
        T_min, T_max = t_init, t_init
        if self.calc_df(pressure, t_init, composition, phase) < 0:
            while True:
                T_max += dT
                if self.calc_df(pressure, T_max, composition) > 0:
                    break
                T_min += dT
        else:
            while True:
                T_min = max(min_T, T_min - dT)
                if self.calc_df(pressure, T_min, composition, phase) < 0:
                    break
                T_max = max(min_T, T_max - dT)
                if T_max == min_T:
                    if verbose:
                        print("Equilibrium temperature below T_min", pressure)
                    return None

        temp = (T_min + T_max) / 2

        # Define objective function for Brent's method
        def obj_fun(temp):
            df = self.calc_df(pressure, temp, composition, phase)
            return df

        rf = RootFinding()
        error = rf.brent(obj_fun, temp, T_min, T_max, tol_f, tol_x)

        if not error == 1:
            return rf.getx()
        else:
            if verbose:
                print("Not converged", pressure)
            return None

    def evaluate_equilibrium(self, state_spec: dict, compositions: dict, mole_fractions: bool,
                             concentrations: dict = None, concentration_unit: cu = cu.MOLALITY, print_state: str = None):
        """
        Method to calculate equilibrium pressure/temperature between fluid phases and hydrate phase at given P/T, z

        :param state_spec: Dictionary containing state specification. Either pressure or temperature should be None
        :param compositions: Dictionary containing compositions
        :param mole_fractions: Switch for mole fractions in state
        :param concentrations: Dictionary of concentrations
        :param concentration_unit: Unit for concentration. 0/MOLALITY) molality (mol/kg H2O), 1/WEIGHT) Weight fraction (-)
        :param print_state: Switch for printing state and progress
        """
        assert state_spec["pressure"] is None or state_spec["temperature"] is None, \
            "Please only provide range of pressures OR temperatures, the other one will be computed"
        calc_pressure = state_spec["pressure"] is None
        output_arrays = {"pres": 1} if calc_pressure else {"temp": 1}
        output_type = {"pres": float} if calc_pressure else {"temp": float}

        self.prev = 1. if calc_pressure else 273.15
        def evaluate(state):
            if calc_pressure:
                result = self.calc_equilibrium_pressure(state[1], state[2:], self.prev, phase="sI")
                output_data = {"pres": lambda pressure=result: pressure}
            else:
                result = self.calc_equilibrium_temperature(state[0], state[2:], self.prev, phase="sI")
                output_data = {"temp": lambda temperature=result: temperature}
            self.prev = result if result is not None else self.prev

            return output_data

        return self.evaluate_full_space(state_spec=state_spec, compositions=compositions, mole_fractions=mole_fractions,
                                        evaluate=evaluate, output_arrays=output_arrays, output_type=output_type,
                                        concentrations=concentrations, concentration_unit=concentration_unit, print_state=print_state)

    def calc_properties(self, state_spec: dict, compositions: dict, state_variables: list, flash_results: xr.Dataset,
                        guest_idx: list, aq_eos: str, nonaq_eos: str, hydrate_phase: str = "sI", x_H: float = None,
                        concentrations: dict = None, concentration_unit: cu = cu.MOLALITY, print_state: str = None):
        """
        Method to calculate hydrate phase properties at given P,T,z:
        - Hydration number nH [-]
        - Density rhoH [kg/m3]
        - Enthalpy of hydrate formation/dissociation dH [kJ/kmol]

        :param state_spec: Dictionary containing state specification
        :param compositions: Dictionary containing variable dimensions
        :param state_variables: List of state variable names to find index in flash results
        :param flash_results: Dataset of flash results with equilibrium curves
        :param guest_idx: Index of guest molecule(s)
        :param aq_eos: Name of aqueous EoS
        :param nonaq_eos: Name of non-aqueous EoS
        :param hydrate_phase: Hydrate phase type
        :param x_H: Constant hydrate composition, optional
        :param concentrations: Dictionary of concentrations
        :param concentration_unit: Unit for concentration. 0/MOLALITY) molality (mol/kg H2O), 1/WEIGHT) Weight fraction (-)
        :param print_state: Switch for printing state and progress
        """
        state_spec["pressure"] = np.array([state_spec["pressure"]]) if not hasattr(state_spec["pressure"], "__len__") else state_spec["pressure"]
        state_spec["temperature"] = np.array([state_spec["temperature"]]) if not hasattr(state_spec["temperature"], "__len__") else state_spec["temperature"]
        assert state_spec["pressure"][0] is None or state_spec["temperature"][0] is None, \
            "Please only provide range of pressures OR temperatures, the other one will be computed"
        calc_pressure = state_spec["pressure"][0] is None
        output_arrays = {"nH": 1, "rhoH": 1, "dH": 1}

        def evaluate(state):
            output_data = {}

            # Retrieve flash results at current state: p, T, X, eos_idxs, roots
            flash_result = flash_results.loc[{state_var: state[i] for i, state_var in enumerate(state_variables[:-1])}]
            eos_idxs = flash_result.eos_idx.values
            X = flash_result.X.values
            x = np.array(X).reshape(self.np_max, self.ns)
            # roots = flash_result.roots.values
            roots = np.zeros(self.np_max)

            # Calculate hydrate composition
            if x_H is None:
                if calc_pressure:
                    xH = self.calc_xH(flash_result.pres.values[0], state[1], x)
                else:
                    xH = self.calc_xH(state[0], flash_result.temp.values[0], x)
            else:
                xH = x_H

            # Calculate hydration number nH
            nH = 1. / xH[guest_idx] - 1.
            output_data["nH"] = lambda: nH

            # Density rhoH
            Mw = np.sum(xH * np.array(self.flash_params.comp_data.Mw)) * 1e-3
            output_data["rhoH"] = lambda: Mw / self.hydrate_eos[hydrate_phase].V(state[0], state[1], xH)

            # Enthalpy of hydrate formation/dissociation
            Hv = self.eos[nonaq_eos].H(state[0], state[1], x[1, :], 0, True) * R
            Ha = nH * self.eos[aq_eos].H(state[0], state[1], x[0, :], 0, True) * R
            Hh = self.hydrate_eos[hydrate_phase].H(state[0], state[1], xH, 0, True) * (nH + 1) * R
            output_data["dH"] = lambda: (Hv + Ha - Hh) * 1e-3  # H_hyd < H_fluids -> enthalpy release upon formation

            return output_data

        return self.evaluate_full_space(state_spec=state_spec, compositions=compositions, mole_fractions=True,
                                        evaluate=evaluate, output_arrays=output_arrays, concentrations=concentrations,
                                        concentration_unit=concentration_unit, print_state=print_state)
