import numpy as np
from darts.engines import value_vector, index_vector
from darts.physics.super.physics import Compositional
from darts.physics.operators_base import PropertyOperators


class Initialize:
    props: dict = {}

    def __init__(self, physics: Compositional, depths: list):
        self.physics = physics
        self.vars = physics.vars
        self.var_idxs = {var: i for i, var in enumerate(physics.vars)}
        self.nv = len(self.vars)

        self.depths = depths
        self.nb = len(depths)
        self.ne = self.nv * self.nb

        self.props = {'s' + ph: lambda j=j: physics.property_containers[0].sat[j] for j, ph in enumerate(physics.phases)}
        self.props['rhoT'] = lambda: np.sum(physics.property_containers[0].sat * physics.property_containers[0].dens)
        self.props_idxs = {prop: i for i, prop in enumerate(self.props.keys())}
        self.primary_specs = {}
        self.secondary_specs = {}

        self.etor = PropertyOperators(physics.property_containers[0], physics.thermal, self.props)
        self.itor = physics.create_interpolator(self.etor, n_ops=physics.n_ops, timer_name='initialization itor')

    def evaluate(self, X, i: int):
        """
        Function to return array of properties.
        Primary variables (vars) are obtained from engine, secondary variables (props) are interpolated by property_itor.

        :returns: property_array
        :rtype: np.ndarray
        """
        # Interpolate values and derivatives in property_itor
        Xi = value_vector([X[i * self.nv + j] for j in range(self.nv)])
        state_idxs = index_vector([0])
        values = value_vector(np.zeros(self.physics.n_ops))
        derivs = value_vector(np.zeros(self.physics.n_ops * self.nv))

        self.itor.evaluate_with_derivatives(Xi, state_idxs, values, derivs)

        return values, derivs

    def set_bc(self, boundary_state: dict, dTdh: float = 0.03, bc_idx: int = 0):
        self.top_bc = True if bc_idx == 0 else False
        self.T = lambda i: boundary_state['temperature'] + (self.depths[i] - self.depths[bc_idx]) * dTdh

        self.P0 = boundary_state['pressure']
        self.Z0 = {var: boundary_state[var] for var in self.vars[1:-1]}

    def set_specs(self, primary_specs: dict = None, secondary_specs: dict = None):
        for spec, values in primary_specs.items():
            self.primary_specs[spec] = values if isinstance(values, (list, np.ndarray)) else np.ones(self.nb) * values
            assert len(self.primary_specs[spec]) == self.nb, "Length of " + spec + " not compatible"
        for spec, values in secondary_specs.items():
            self.secondary_specs[spec] = values if isinstance(values, (list, np.ndarray)) else np.ones(self.nb) * values
            assert len(self.secondary_specs[spec]) == self.nb, "Length of " + spec + " not compatible"
        assert len(self.primary_specs) + len(self.secondary_specs) >= self.nv - 2, \
            "Not enough variables specified for well-defined system of equations"

    def set_initial_guess(self, Z0: dict):
        for j, z0 in Z0.items():
            Z0[j] = z0 if isinstance(z0, (list, np.ndarray)) else [z0] * self.nb
        X0 = np.zeros(self.ne)

        # X of boundary cell
        idx = 0 if self.top_bc else self.nb - 1
        X0[idx * self.nv] = self.P0
        X0[idx * self.nv + self.nv - 1] = self.T(idx)
        for j in range(self.nv - 2):
            if j in self.Z0.keys():
                X0[idx * self.nv + j + 1] = self.Z0[j]
            else:
                X0[idx * self.nv + j + 1] = Z0[self.vars[j + 1]][idx]

        # X of other blocks
        for i in range(1, self.nb):
            # Set pressure of block i
            if self.top_bc:
                idx = i
                values, _ = self.evaluate(X0, idx - 1)
                gh = 9.81 * (self.depths[idx] - self.depths[idx - 1]) * 1e-5
                X0[idx * self.nv] = X0[(idx - 1) * self.nv] + values[self.props_idxs['rhoT']] * gh
            else:
                idx = self.nb - i - 1
                values, _ = self.evaluate(X0, idx + 1)
                gh = 9.81 * (self.depths[idx - 1] - self.depths[idx]) * 1e-5
                X0[idx * self.nv] = X0[(idx + 1) * self.nv] - values[self.props_idxs['rhoT']] * gh

            # Set temperature of block i
            X0[idx * self.nv + self.nv - 1] = self.T(idx)

            # Set zj of block i
            for j, var in self.vars[1:-1]:
                X0[idx * self.nv + j + 1] = Z0[var][idx]

        return X0

    def pressure_equation(self, X, res, Jac, i: int, top: bool):
        # Evaluate values and derivatives in blocks i and i+1
        if top:
            idx0, idx1 = i, i + 1
            dh = self.depths[idx1] - self.depths[idx0]
        else:
            idx0, idx1 = self.nb - i - 1, self.nb - i
            dh = self.depths[idx0] - self.depths[idx1]
        values0, derivs0 = self.evaluate(X, idx0)
        values1, derivs1 = self.evaluate(X, idx1)

        # P[i] + 1/2 * (rho[i+1] + rho[i]) * gh - P[i+1] = 0
        rhoT_idx = self.props_idxs['rhoT']
        gh = 9.81 * dh * 1e-5
        mgh = (values0[rhoT_idx] + values1[rhoT_idx]) * 0.5 * gh

        # If top BC: P[0] = P_top; otherwise P[-1] = P_bottom
        res[idx1 * self.nv] = X[idx0 * self.nv] + mgh - X[idx1 * self.nv]
        Jac[idx1 * self.nv, idx0 * self.nv] += 1.
        Jac[idx1 * self.nv, idx1 * self.nv] -= 1.
        for j in range(self.nv):
            Jac[idx1 * self.nv, idx0 * self.nv + j] += derivs0[rhoT_idx * self.nv + j] * 0.5 * gh
            Jac[idx1 * self.nv, idx1 * self.nv + j] += derivs1[rhoT_idx * self.nv + j] * 0.5 * gh

        return res, Jac

    def spec_equation(self, X, res, Jac, i: int):
        values, derivs = self.evaluate(X, i)

        # Specification of primary variables (P,T are not specified here)
        for j1, (var, spec) in enumerate(self.primary_specs.items()):
            var_idx = self.var_idxs[var]
            res_idx = i * self.nv + j1 + 1
            res[res_idx] = X[i * self.nv + var_idx] - spec[i]
            Jac[res_idx, :] = 0.
            Jac[res_idx, i * self.nv + var_idx] = 1.
        j1 = len(self.primary_specs)

        # Specification of secondary variables
        for j2, (var, spec) in enumerate(self.secondary_specs.items()):
            prop_idx = self.props_idxs[var]
            res_idx = i * self.nv + j1 + j2 + 1
            res[res_idx] = values[prop_idx] - spec[i]

            for jj in range(self.nv):
                Jac[res_idx, i * self.nv + jj] = derivs[prop_idx * self.nv + jj]
        j2 = len(self.secondary_specs)

        assert j1 + j2 + 2 == self.nv
        return res, Jac

    def assemble(self, X):
        res = np.zeros(self.ne)
        Jac = np.zeros((self.ne, self.ne))

        # NB-1 pressure equations
        for i in range(self.nb - 1):
            res, Jac = self.pressure_equation(X, res, Jac, i, self.top_bc)

        # NB specification equations
        for i in range(self.nb):
            res, Jac = self.spec_equation(X, res, Jac, i)

            # Temperature equation
            idx = i * self.nv + self.nv - 1
            res[idx] = 0
            Jac[idx, :] = 0
            Jac[idx, idx] = 1

        # Pressure BC
        idx = 0 if self.top_bc else (self.nb - 1) * self.nv

        res[idx] = 0
        Jac[idx, :] = 0
        Jac[idx, idx] = 1.

        return res, Jac

    def solve(self, X0, tol: float = 1e-10, max_iter: int = 100):
        """
        :param X0: Initial guess of solution
        :return: Set of primary variables at solution
        :param tol: Convergence criterion
        :type tol: float
        :param max_iter: Maximum number of iterations
        :type max_iter: int
        """
        # Start from initial guess
        X = X0
        it = 0
        while 1:
            it += 1
            res, Jac = self.assemble(X)

            # Solve Newton step
            dX = np.linalg.solve(Jac, res)
            X -= dX
            norm = np.linalg.norm(res)

            if np.linalg.norm(norm) < tol or it == max_iter:
                if it == 100:
                    raise ValueError('Initialization procedure did not converge')
                break

        return X
