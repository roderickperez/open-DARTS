import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from sklearn.tree import DecisionTreeRegressor

# Define the neural network model to approximate the Q-values.
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        """
        Initialize the neural network.
        :param state_size: The number of input features (dimensions of the state).
        :param action_size: The number of possible actions.
        """
        super(DQN, self).__init__()
        # First fully connected layer with 24 neurons
        self.fc1 = nn.Linear(state_size, 24)
        # Second fully connected layer with 24 neurons
        self.fc2 = nn.Linear(24, 24)
        # Output layer: one output per action
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        """
        Define the forward pass through the network.
        :param x: Input tensor (state vector).
        :return: Output tensor (Q-values for each action).
        """
        x = torch.relu(self.fc1(x))  # Apply ReLU activation after first layer
        x = torch.relu(self.fc2(x))  # Apply ReLU activation after second layer
        x = self.fc3(x)  # No activation on output layer (linear output for Q-values)
        return x

class TreeQ:
    def __init__(self, state_size, action_size, max_depth=5):
        """
        Initialize a separate decision tree for each action.
        :param state_size: Dimensionality of the state.
        :param action_size: Number of actions.
        :param max_depth: Maximum depth of each tree.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.models = [DecisionTreeRegressor(max_depth=max_depth) for _ in range(action_size)]
        # Track whether a model has been fitted at least once.
        self.initialized = [False] * action_size

    def predict(self, state):
        """
        Predict Q-values for each action given a single state.
        :param state: A numpy array of shape (state_size,).
        :return: A numpy array of Q-values for each action.
        """
        q_values = []
        for a in range(self.action_size):
            if self.initialized[a]:
                q_val = self.models[a].predict(state.reshape(1, -1))[0]
            else:
                q_val = 0.0  # Default value if the tree hasn't been trained.
            q_values.append(q_val)
        return np.array(q_values)

    def update(self, states, targets, actions):
        """
        Update the tree models using the provided replay batch.
        :param states: np.array of shape (n_samples, state_size)
        :param targets: np.array of shape (n_samples,) - target Q-values.
        :param actions: np.array of shape (n_samples,) - the action taken.
        """
        for a in range(self.action_size):
            idx = np.where(actions == a)[0]
            if len(idx) > 0:
                X_a = states[idx]
                y_a = targets[idx]
                # Fit a new tree (or refit) for this action.
                self.models[a].fit(X_a, y_a)
                self.initialized[a] = True

class LinearQ(nn.Module):
    def __init__(self, state_size, action_size):
        super(LinearQ, self).__init__()
        # A single linear layer mapping state to Q-values for each action.
        self.fc = nn.Linear(state_size, action_size)

    def forward(self, x):
        return self.fc(x)


# Define the DQN Agent class
class DQNAgent:
    def __init__(self, state_size, action_size, model_type='nn'):
        """
        Initialize the agent.
        :param state_size: Dimensionality of the state space.
        :param action_size: Number of available actions.
        :param model_type: 'nn' (default), 'linear', or 'tree'
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95           # Discount factor
        self.epsilon = 1.0          # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model_type = model_type

        if self.model_type in ['nn', 'linear']:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self.model_type == 'nn':
                self.model = DQN(state_size, action_size).to(self.device)
            elif self.model_type == 'linear':
                self.model = LinearQ(state_size, action_size).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.criterion = nn.MSELoss()
        elif self.model_type == 'tree':
            # For the tree model, we do not use PyTorch; use scikit-learn.
            self.tree_model = TreeQ(state_size, action_size)
        else:
            raise ValueError("Unknown model type: choose 'nn', 'linear', or 'tree'.")
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Choose an action based on the current state using an epsilon-greedy strategy.
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        if self.model_type in ['nn', 'linear']:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()
        elif self.model_type == 'tree':
            q_values = self.tree_model.predict(state)
            return np.argmax(q_values)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        if self.model_type in ['nn', 'linear']:
            # Prepare tensors for training.
            states = torch.FloatTensor([exp[0] for exp in minibatch]).to(self.device)
            actions = torch.LongTensor([exp[1] for exp in minibatch]).unsqueeze(1).to(self.device)
            rewards = torch.FloatTensor([exp[2] for exp in minibatch]).to(self.device)
            next_states = torch.FloatTensor([exp[3] for exp in minibatch]).to(self.device)
            dones = torch.FloatTensor([float(exp[4]) for exp in minibatch]).to(self.device)

            current_q_values = self.model(states).gather(1, actions)
            with torch.no_grad():
                next_q_values = self.model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            target_q_values = target_q_values.unsqueeze(1)

            loss = self.criterion(current_q_values, target_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        elif self.model_type == 'tree':
            # For tree-based approximator, use the replay data to refit the trees.
            states = np.array([exp[0] for exp in minibatch])
            actions = np.array([exp[1] for exp in minibatch])
            rewards = np.array([exp[2] for exp in minibatch])
            next_states = np.array([exp[3] for exp in minibatch])
            dones = np.array([float(exp[4]) for exp in minibatch])
            # Get predicted Q-values for next states.
            next_q_values = np.array([self.tree_model.predict(s) for s in next_states])
            max_next_q = np.max(next_q_values, axis=1)
            targets = rewards + (1 - dones) * self.gamma * max_next_q
            self.tree_model.update(states, targets, actions)

        # Decay epsilon regardless of the model.
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay