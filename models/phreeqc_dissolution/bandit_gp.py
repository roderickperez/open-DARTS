import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel


class GPBandit:
    def __init__(self, context_dim, dt_bounds, kappa=2.0, noise=1e-4):
        """
        Initialize the GP bandit model.
        :param context_dim: Dimension of the context vector.
        :param dt_bounds: Tuple (min_dt, max_dt) for allowed dt values.
        :param kappa: Exploration parameter for the UCB acquisition function.
        :param noise: Noise level for GP regression.
        """
        self.context_dim = context_dim
        self.dt_bounds = dt_bounds
        self.kappa = kappa
        self.noise = noise

        # The input to the GP is a concatenation of context and dt.
        self.input_dim = context_dim + 1

        # Define a kernel: constant * RBF + WhiteKernel to model noise.
        kernel = C(1.0, (1e-2, 1e2)) * \
                 RBF(length_scale=np.ones(self.input_dim), length_scale_bounds=(1e-2, 1e2)) + \
                 WhiteKernel(noise_level=noise)

        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=noise,
                                           n_restarts_optimizer=2, normalize_y=True)

        # Initialize training data storage.
        self.X = None  # Will store feature vectors of shape (n_samples, input_dim)
        self.y = None  # Corresponding rewards, shape (n_samples,)

    def _feature(self, context, dt):
        """
        Create the combined feature vector by stacking context and dt.
        :param context: numpy array of shape (context_dim,)
        :param dt: scalar timestep value.
        :return: numpy array of shape (input_dim,)
        """
        return np.hstack((context, [dt]))

    def update(self, context, dt, reward):
        """
        Update the GP model with a new observation.
        :param context: numpy array of shape (context_dim,)
        :param dt: chosen timestep (scalar)
        :param reward: observed reward (scalar)
        """
        # Form the feature vector from context and dt.
        x_new = self._feature(context, dt).reshape(1, -1)
        y_new = np.array([reward])

        # Append new data to our training set.
        if self.X is None:
            self.X = x_new
            self.y = y_new
        else:
            self.X = np.vstack((self.X, x_new))
            self.y = np.hstack((self.y, y_new))

        # Re-fit the GP model with the updated data (feasible if the number of samples is small).
        self.gp.fit(self.X, self.y)

    def acquisition(self, context, dt):
        """
        Compute the Upper Confidence Bound (UCB) acquisition function for a given context and dt.
        :param context: numpy array of shape (context_dim,)
        :param dt: scalar timestep value.
        :return: Acquisition value (scalar).
        """
        x = self._feature(context, dt).reshape(1, -1)
        # Predict mean and standard deviation using the GP.
        mu, sigma = self.gp.predict(x, return_std=True)
        # Ensure mu and sigma are at least 1D.
        mu = np.atleast_1d(mu)
        sigma = np.atleast_1d(sigma)
        # UCB acquisition: mean plus kappa times the standard deviation.
        return mu[0] + self.kappa * sigma[0]

    def select_action(self, context, num_candidates=50):
        """
        Given a context, select the next dt by maximizing the acquisition function.
        The function evaluates a grid of candidate dt values.
        :param context: numpy array of shape (context_dim,)
        :param num_candidates: Number of candidate dt values to sample in the interval.
        :return: chosen dt (scalar)
        """
        min_dt, max_dt = self.dt_bounds
        candidate_dts = np.linspace(min_dt, max_dt, num_candidates)
        # Evaluate the acquisition function for each candidate dt.
        acq_values = [self.acquisition(context, dt) for dt in candidate_dts]
        best_index = np.argmax(acq_values)
        chosen_dt = candidate_dts[best_index]
        return chosen_dt


# ================= Example Usage ====================
if __name__ == "__main__":
    # Example parameters.
    context_dim = 5  # e.g., [max_residual, nonlinear_iters, linear_iters, max_CFL, max_change]
    dt_bounds = (1e-5, 1e-3)  # Allowable dt range.
    kappa = 2.0  # Exploration-exploitation trade-off parameter.

    # Instantiate the GP bandit model.
    bandit = GPBandit(context_dim=context_dim, dt_bounds=dt_bounds, kappa=kappa)


    # Suppose this function computes the reward for a given dt in your simulation.
    def simulate_reward(context, dt):
        # For demonstration, assume reward is higher for larger dt up to a point,
        # but if dt is too high, divergence causes a penalty.
        # Here we use a simple function: reward = dt - penalty*(dt - ideal_dt)^2.
        ideal_dt = 5e-4  # Hypothetical ideal dt.
        penalty = 1e4
        reward = dt - penalty * (dt - ideal_dt) ** 2
        return reward


    # Simulate a sequence of decisions.
    for t in range(20):
        # Example: generate a random context vector.
        context = np.random.rand(context_dim)  # In practice, use your actual simulation context.

        # Select dt using the GP bandit.
        chosen_dt = bandit.select_action(context)
        print(f"Step {t + 1}: chosen dt = {chosen_dt:.2e}")

        # Simulate obtaining a reward.
        reward = simulate_reward(context, chosen_dt)
        print(f"  Observed reward: {reward:.2e}")

        # Update the GP bandit with the new observation.
        bandit.update(context, chosen_dt, reward)
