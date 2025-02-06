import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

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


# Define the DQN Agent class
class DQNAgent:
    def __init__(self, state_size, action_size):
        """
        Initialize the DQN agent.
        :param state_size: Dimensionality of the state space.
        :param action_size: Number of available actions.
        """
        self.state_size = state_size  # Number of state features
        self.action_size = action_size  # Number of possible actions
        self.memory = deque(maxlen=2000)  # Experience replay buffer with a maximum size of 2000
        self.gamma = 0.95  # Discount factor for future rewards
        self.epsilon = 1.0  # Initial exploration rate (epsilon-greedy policy)
        self.epsilon_min = 0.01  # Minimum exploration rate after decay
        self.epsilon_decay = 0.995  # Decay rate for the exploration probability
        self.learning_rate = 0.001  # Learning rate for the optimizer

        # Check if CUDA (GPU) is available and use it if possible; otherwise, use CPU.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Instantiate the Q-network and move it to the chosen device.
        self.model = DQN(state_size, action_size).to(self.device)
        # Define the optimizer (Adam) for the neural network parameters.
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # Define the loss function as Mean Squared Error (MSE).
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        """
        Store an experience tuple in the replay memory.
        :param state: The state at time t.
        :param action: The action taken at time t.
        :param reward: The reward received after taking the action.
        :param next_state: The state at time t+1.
        :param done: Boolean flag indicating if the episode ended.
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Choose an action for the current state using an epsilon-greedy strategy.
        :param state: The current state (numpy array).
        :return: The chosen action (integer).
        """
        # With probability epsilon, take a random action (exploration).
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        # Otherwise, use the network to predict Q-values and choose the best action (exploitation).
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(
            self.device)  # Convert state to tensor and add batch dimension.
        with torch.no_grad():  # Disable gradient computation for inference.
            q_values = self.model(state_tensor)
        # Return the index of the action with the highest Q-value.
        return torch.argmax(q_values).item()

    def replay(self, batch_size):
        """
        Sample a random batch of experiences from the replay memory and train the network.
        :param batch_size: Number of experiences to sample for training.
        """
        if len(self.memory) < batch_size:
            return  # Not enough samples to form a batch.

        # Randomly sample a batch of experiences from memory.
        minibatch = random.sample(self.memory, batch_size)

        # Prepare tensors for states, actions, rewards, next_states, and done flags.
        states = torch.FloatTensor([exp[0] for exp in minibatch]).to(self.device)
        actions = torch.LongTensor([exp[1] for exp in minibatch]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([exp[2] for exp in minibatch]).to(self.device)
        next_states = torch.FloatTensor([exp[3] for exp in minibatch]).to(self.device)
        dones = torch.FloatTensor([float(exp[4]) for exp in minibatch]).to(self.device)

        # Compute the current Q-values for the actions taken in each state.
        current_q_values = self.model(states).gather(1, actions)

        # Compute the Q-values for the next states.
        with torch.no_grad():
            # Get the maximum predicted Q-value for each next state.
            next_q_values = self.model(next_states).max(1)[0]
        # Compute the target Q-values using the Bellman equation.
        # If the state is terminal (done), then the target is just the reward.
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        # Reshape target Q-values to match the shape of current_q_values.
        target_q_values = target_q_values.unsqueeze(1)

        # Calculate the loss between the current Q-values and the target Q-values.
        loss = self.criterion(current_q_values, target_q_values)

        # Zero the gradients, perform backpropagation, and update the network parameters.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon to reduce exploration over time.
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay