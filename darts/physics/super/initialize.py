import numpy as np
from darts.engines import value_vector, index_vector
from darts.physics.base.operators_base import PropertyOperators


class Initialize:
    def __init__(self, physics, depth_bottom: float, depth_top: float, boundary_state: dict, nb: int = 100):
        self.physics = physics
        self.vars = physics.vars
        self.var_idxs = {var: i for i, var in enumerate(physics.vars)}
        self.nv = len(self.vars)
        self.thermal = ('temperature' in self.vars)

        # check input and create depths
        assert(depth_bottom > depth_top)
        assert(boundary_state['depth'] <= depth_bottom and boundary_state['depth'] >= depth_top)
        self.nb = nb
        self.depths = np.linspace(start=depth_top, stop=depth_bottom, num=nb)
        self.boundary_state = boundary_state
        self.bc_idx = (np.fabs(self.depths - boundary_state['depth'])).argmin()
        self.depths[self.bc_idx] = boundary_state['depth']

        # Add evaluators of phase saturations, rhoT and dX (if kinetic reactions are defined)
        property_container = physics.property_containers[0]
        self.props = {}
        self.props.update({'s' + ph: lambda j=j: property_container.sat[j] for j, ph in enumerate(physics.phases)})
        self.props.update({'rhoT': lambda: np.sum(property_container.sat * property_container.dens)})
        self.props.update({'dX' + str(k): lambda k=k: property_container.dX[k]
                           for k, kr in enumerate(property_container.kinetic_rate_ev)})

        self.props_idxs = {prop: i for i, prop in enumerate(self.props.keys())}
        self.primary_specs = {}
        self.secondary_specs = {}

        self.etor = PropertyOperators(physics.property_containers[0], physics.thermal, self.props)
        self.itor = physics.create_interpolator(self.etor, n_ops=physics.n_ops, timer_name='initialization itor')

    def evaluate(self, Xi: int):
        """
        Function to return array of properties.
        Primary variables (vars) are obtained from engine, secondary variables (props) are interpolated by property_itor.

        :returns: property_array
        :rtype: np.ndarray
        """
        # Interpolate values and derivatives in property_itor
        state_idxs = index_vector([0])
        values = value_vector(np.zeros(self.physics.n_ops))
        derivs = value_vector(np.zeros(self.physics.n_ops * self.nv))

        self.itor.evaluate_with_derivatives(value_vector(Xi), state_idxs, values, derivs)

        return values, derivs

    def solve(self, dTdh: float = 0.03):
        if self.thermal:
            self.T = lambda i: self.boundary_state['temperature'] + (self.depths[i] - self.depths[self.bc_idx]) * dTdh

        # Set state in known cell
        X = np.zeros((self.nb, self.nv))
        X[self.bc_idx] = np.array([self.boundary_state[v] for v in self.vars])

        # Solve cells from specified cell upwards
        for i in range(self.bc_idx, 0, -1):
            self.solve_cell(X, cell_idx=i-1, downward=False)

        # Solve cells from specified cell downwards
        for i in range(self.bc_idx, self.nb-1):
            self.solve_cell(X, cell_idx=i+1, downward=True)

        return X.flatten()

    def solve_cell(self, X: np.ndarray, cell_idx: int, downward: bool = True, max_iter: int = 100):
        # Find neighbouring cell for which state is known
        if downward:
            known_idx = cell_idx - 1
        else:
            known_idx = cell_idx + 1

        n_vars = self.nv - self.thermal
        values0, _ = self.evaluate(X[known_idx])
        gh0 = 9.81 * self.depths[known_idx] * 1e-5
        gh1 = 9.81 * self.depths[cell_idx] * 1e-5

        # Solve nonlinear unknowns
        # Initialize using same composition, recalculate pressure and evaluate temperature gradient
        X[cell_idx, 0] = X[known_idx, 0] + values0[self.props_idxs['rhoT']] * (gh1 - gh0)
        X[cell_idx, 1:] = X[known_idx, 1:]
        if self.thermal:
            X[cell_idx, -1] = self.T(cell_idx)

        rhoT_idx = self.props_idxs['rhoT']
        for it in range(max_iter):
            # nc variables for pressure and nc-1 compositions, temperature is calculated from gradient
            res = np.zeros(n_vars)
            Jac = np.zeros((n_vars, n_vars))

            # Evaluate operators and derivatives at current state Xi
            values1, derivs1 = self.evaluate(X[cell_idx])

            # Pressure equation
            mgh = (values1[rhoT_idx] + values0[rhoT_idx]) * (gh1 - gh0) / 2
            res[0] = X[known_idx, 0] + mgh - X[cell_idx, 0]
            Jac[0, 0] -= 1.
            for j in range(n_vars):
                Jac[0, j] += derivs1[rhoT_idx * self.nv + j] * (gh1 - gh0) / 2

            # Specification equation
            for j1, (var, spec) in enumerate(self.primary_specs.items()):
                var_idx = self.var_idxs[var]
                res_idx = j1 + 1
                res[res_idx] = X[cell_idx, var_idx] - spec[cell_idx]
                Jac[res_idx, :] = 0.
                Jac[res_idx, var_idx] = 1.
            j1 = len(self.primary_specs)

            # Specification of secondary variables
            for j2, (var, spec) in enumerate(self.secondary_specs.items()):
                prop_idx = self.props_idxs[var]
                res_idx = j1 + j2 + 1
                res[res_idx] = values1[prop_idx] - spec[cell_idx]

                for jj in range(n_vars):
                    Jac[res_idx, jj] = derivs1[prop_idx * self.nv + jj]
            j2 = len(self.secondary_specs)

            dX = np.linalg.solve(Jac, res)
            X[cell_idx, :n_vars] -= dX

            if np.linalg.norm(res) < 1e-10:
                return X

        print("MAX ITER REACHED FOR INITIALIZATION", X[cell_idx, :])
        return X

    # def set_bc(self, boundary_state: dict, dTdh: float = 0.03, bc_idx: int = 0):
    #     self.top_bc = True if bc_idx == 0 else False
    #     self.boundary_state = boundary_state
    #     if self.thermal:
    #         self.T = lambda i: boundary_state['temperature'] + (self.depths[i] - self.depths[bc_idx]) * dTdh

    def set_specs(self, primary_specs: dict = None, secondary_specs: dict = None):
        if primary_specs is not None:
            for spec, values in primary_specs.items():
                self.primary_specs[spec] = values if isinstance(values, (list, np.ndarray)) else np.ones(self.nb) * values
                assert len(self.primary_specs[spec]) == self.nb, "Length of " + spec + " not compatible"
        if secondary_specs is not None:
            for spec, values in secondary_specs.items():
                self.secondary_specs[spec] = values if isinstance(values, (list, np.ndarray)) else np.ones(self.nb) * values
                assert len(self.secondary_specs[spec]) == self.nb, "Length of " + spec + " not compatible"
        assert len(self.primary_specs) + len(self.secondary_specs) >= self.nv - 2, \
            "Not enough variables specified for well-defined system of equations"

    # def set_initial_guess(self, state0: dict):
    #     for j, z0 in state0.items():
    #         state0[j] = z0 if isinstance(z0, (list, np.ndarray)) else [z0] * self.nb
    #     X0 = np.zeros(self.ne)
    #
    #     # X of boundary cell
    #     idx = 0 if self.top_bc else self.nb - 1
    #     X0[idx * self.nv] = self.boundary_state['pressure']
    #     if self.thermal:
    #         X0[idx * self.nv + self.nv - 1] = self.T(idx)
    #
    #     for j, var in enumerate(self.vars[1:-1]):
    #         if var in self.boundary_state.keys():
    #             X0[idx * self.nv + j + 1] = self.boundary_state[var]
    #         else:
    #             X0[idx * self.nv + j + 1] = state0[var][idx]
    #
    #     # X of other blocks
    #     for i in range(1, self.nb):
    #         # Set pressure of block i
    #         if self.top_bc:
    #             idx = i
    #             values, _ = self.evaluate(X0, idx - 1)
    #             gh = 9.81 * (self.depths[idx] - self.depths[idx - 1]) * 1e-5
    #             X0[idx * self.nv] = X0[(idx - 1) * self.nv] + values[self.props_idxs['rhoT']] * gh
    #         else:
    #             idx = self.nb - i - 1
    #             values, _ = self.evaluate(X0, idx + 1)
    #             gh = 9.81 * (self.depths[idx - 1] - self.depths[idx]) * 1e-5
    #             X0[idx * self.nv] = X0[(idx + 1) * self.nv] - values[self.props_idxs['rhoT']] * gh
    #
    #         if self.thermal:
    #             # Set temperature of block i
    #             X0[idx * self.nv + self.nv - 1] = self.T(idx)
    #
    #         # Set zj of block i
    #         for j, var in enumerate(self.vars[1:-1]):
    #             X0[idx * self.nv + j + 1] = state0[var][idx]
    #
    #     return X0
    #
    # def pressure_equation(self, X, res, Jac, i: int, top: bool):
    #     # Evaluate values and derivatives in blocks i and i+1
    #     if top:
    #         idx0, idx1 = i, i + 1
    #         dh = self.depths[idx1] - self.depths[idx0]
    #     else:
    #         idx0, idx1 = self.nb - i - 1, self.nb - i - 2
    #         dh = self.depths[idx1] - self.depths[idx0]
    #     values0, derivs0 = self.evaluate(X, idx0)
    #     values1, derivs1 = self.evaluate(X, idx1)
    #
    #     # P[i] + 1/2 * (rho[i+1] + rho[i]) * gh - P[i+1] = 0
    #     rhoT_idx = self.props_idxs['rhoT']
    #     gh = 9.81 * dh * 1e-5
    #     mgh = (values0[rhoT_idx] + values1[rhoT_idx]) * 0.5 * gh
    #
    #     # If top BC: P[0] = P_top; otherwise P[-1] = P_bottom
    #     res[idx1 * self.nv] = X[idx0 * self.nv] + mgh - X[idx1 * self.nv]
    #     Jac[idx1 * self.nv, idx0 * self.nv] += 1.
    #     Jac[idx1 * self.nv, idx1 * self.nv] -= 1.
    #     for j in range(self.nv):
    #         Jac[idx1 * self.nv, idx0 * self.nv + j] += derivs0[rhoT_idx * self.nv + j] * 0.5 * gh
    #         Jac[idx1 * self.nv, idx1 * self.nv + j] += derivs1[rhoT_idx * self.nv + j] * 0.5 * gh
    #
    #     return res, Jac
    #
    # def spec_equation(self, X, res, Jac, i: int):
    #     values, derivs = self.evaluate(X, i)
    #
    #     # Specification of primary variables (P,T are not specified here)
    #     for j1, (var, spec) in enumerate(self.primary_specs.items()):
    #         var_idx = self.var_idxs[var]
    #         res_idx = i * self.nv + j1 + 1
    #         res[res_idx] = X[i * self.nv + var_idx] - spec[i]
    #         Jac[res_idx, :] = 0.
    #         Jac[res_idx, i * self.nv + var_idx] = 1.
    #     j1 = len(self.primary_specs)
    #
    #     # Specification of secondary variables
    #     for j2, (var, spec) in enumerate(self.secondary_specs.items()):
    #         prop_idx = self.props_idxs[var]
    #         res_idx = i * self.nv + j1 + j2 + 1
    #         res[res_idx] = values[prop_idx] - spec[i]
    #
    #         for jj in range(self.nv):
    #             Jac[res_idx, i * self.nv + jj] = derivs[prop_idx * self.nv + jj]
    #     j2 = len(self.secondary_specs)
    #
    #     # assert j1 + j2 + 2 == self.nv
    #     return res, Jac
    #
    # def assemble(self, X):
    #     res = np.zeros(self.ne)
    #     Jac = np.zeros((self.ne, self.ne))
    #
    #     # NB-1 pressure equations
    #     for i in range(self.nb - 1):
    #         res, Jac = self.pressure_equation(X, res, Jac, i, self.top_bc)
    #
    #     # NB specification equations
    #     for i in range(self.nb):
    #         res, Jac = self.spec_equation(X, res, Jac, i)
    #
    #         if self.thermal:
    #             # Temperature equation
    #             idx = i * self.nv + self.nv - 1
    #             res[idx] = 0
    #             Jac[idx, :] = 0
    #             Jac[idx, idx] = 1
    #
    #     # Pressure BC
    #     idx = 0 if self.top_bc else (self.nb - 1) * self.nv
    #
    #     res[idx] = 0
    #     Jac[idx, :] = 0
    #     Jac[idx, idx] = 1.
    #
    #     return res, Jac
    #
    # def solve(self, X0, tol: float = 1e-10, max_iter: int = 100):
    #     """
    #     :param X0: Initial guess of solution
    #     :return: Set of primary variables at solution
    #     :param tol: Convergence criterion
    #     :type tol: float
    #     :param max_iter: Maximum number of iterations
    #     :type max_iter: int
    #     """
    #     # Start from initial guess
    #     X = X0
    #     it = 0
    #     while 1:
    #         it += 1
    #         res, Jac = self.assemble(X)
    #
    #         # Solve Newton step
    #         dX = np.linalg.solve(Jac, res)
    #         X -= dX
    #         norm = np.linalg.norm(res)
    #
    #         if np.linalg.norm(norm) < tol or it == max_iter:
    #             if it == 100:
    #                 raise ValueError('Initialization procedure did not converge')
    #             break
    #
    #     return X
