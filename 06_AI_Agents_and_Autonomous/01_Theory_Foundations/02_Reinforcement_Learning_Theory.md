# Reinforcement Learning Theory and Mathematical Foundations

## Overview
Reinforcement Learning (RL) is a subfield of machine learning where agents learn to make optimal decisions through trial and error interactions with an environment. This section covers the mathematical foundations, algorithms, and theoretical principles behind RL.

## 1. Mathematical Foundations

### 1.1 Markov Decision Processes (MDPs)

An MDP is a tuple $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$ where:

- $\mathcal{S}$: State space (finite or continuous)
- $\mathcal{A}$: Action space (finite or continuous)
- $\mathcal{P}$: Transition probability function
- $\mathcal{R}$: Reward function
- $\gamma$: Discount factor ($0 \leq \gamma \leq 1$)

**Transition Probability:**
$$\mathcal{P}(s'|s,a) = \mathbb{P}[S_{t+1} = s' | S_t = s, A_t = a]$$

**Reward Function:**
$$\mathcal{R}(s,a,s') = \mathbb{E}[R_{t+1} | S_t = s, A_t = a, S_{t+1} = s']$$

### 1.2 Value Functions

**State Value Function:**
$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t R_{t+1} | S_0 = s\right]$$

**Action Value Function:**
$$Q^\pi(s,a) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t R_{t+1} | S_0 = s, A_0 = a\right]$$

**Bellman Equations:**
$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} \mathcal{P}(s'|s,a) [\mathcal{R}(s,a,s') + \gamma V^\pi(s')]$$

$$Q^\pi(s,a) = \sum_{s'} \mathcal{P}(s'|s,a) [\mathcal{R}(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s',a')]$$

### 1.3 Optimal Value Functions

**Optimal State Value:**
$$V^*(s) = \max_\pi V^\pi(s)$$

**Optimal Action Value:**
$$Q^*(s,a) = \max_\pi Q^\pi(s,a)$$

**Bellman Optimality Equations:**
$$V^*(s) = \max_a \sum_{s'} \mathcal{P}(s'|s,a) [\mathcal{R}(s,a,s') + \gamma V^*(s')]$$

$$Q^*(s,a) = \sum_{s'} \mathcal{P}(s'|s,a) [\mathcal{R}(s,a,s') + \gamma \max_{a'} Q^*(s',a')]$$

```python
import numpy as np
from typing import List, Dict, Tuple, Callable, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import defaultdict, deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

@dataclass
class State:
    """State representation"""
    id: int
    features: np.ndarray

@dataclass
class Action:
    """Action representation"""
    id: int
    name: str
    parameters: Dict[str, float]

class MDP:
    """Markov Decision Process"""

    def __init__(self, states: List[State], actions: List[Action],
                 transition_matrix: np.ndarray, reward_matrix: np.ndarray,
                 discount_factor: float = 0.99):
        self.states = states
        self.actions = actions
        self.transition_matrix = transition_matrix  # shape: [n_states, n_actions, n_states]
        self.reward_matrix = reward_matrix  # shape: [n_states, n_actions, n_states]
        self.discount_factor = discount_factor

        self.n_states = len(states)
        self.n_actions = len(actions)

    def get_transition_prob(self, state: State, action: Action, next_state: State) -> float:
        """Get transition probability P(s'|s,a)"""
        return self.transition_matrix[state.id, action.id, next_state.id]

    def get_reward(self, state: State, action: Action, next_state: State) -> float:
        """Get reward R(s,a,s')"""
        return self.reward_matrix[state.id, action.id, next_state.id]

    def get_expected_reward(self, state: State, action: Action) -> float:
        """Get expected reward for state-action pair"""
        expected_reward = 0.0
        for next_state in self.states:
            prob = self.get_transition_prob(state, action, next_state)
            reward = self.get_reward(state, action, next_state)
            expected_reward += prob * reward
        return expected_reward

    def sample_next_state(self, state: State, action: Action) -> Tuple[State, float]:
        """Sample next state and reward given current state and action"""
        probs = self.transition_matrix[state.id, action.id, :]
        next_state_id = np.random.choice(self.n_states, p=probs)
        next_state = self.states[next_state_id]

        reward = self.get_reward(state, action, next_state)

        return next_state, reward

class Policy(ABC):
    """Abstract base class for policies"""

    @abstractmethod
    def get_action(self, state: State) -> Action:
        """Get action for given state"""
        pass

    @abstractmethod
    def get_action_prob(self, state: State, action: Action) -> float:
        """Get probability of action in given state"""
        pass

class DeterministicPolicy(Policy):
    """Deterministic policy"""

    def __init__(self, policy_mapping: Dict[int, int]):
        self.policy_mapping = policy_mapping  # state_id -> action_id

    def get_action(self, state: State) -> Action:
        action_id = self.policy_mapping.get(state.id, 0)
        return self.actions[action_id] if hasattr(self, 'actions') else None

    def get_action_prob(self, state: State, action: Action) -> float:
        action_id = self.policy_mapping.get(state.id, 0)
        return 1.0 if action.id == action_id else 0.0

class StochasticPolicy(Policy):
    """Stochastic policy"""

    def __init__(self, policy_matrix: np.ndarray, actions: List[Action]):
        self.policy_matrix = policy_matrix  # shape: [n_states, n_actions]
        self.actions = actions

    def get_action(self, state: State) -> Action:
        probs = self.policy_matrix[state.id, :]
        action_id = np.random.choice(len(self.actions), p=probs)
        return self.actions[action_id]

    def get_action_prob(self, state: State, action: Action) -> float:
        return self.policy_matrix[state.id, action.id]
```

## 2. Dynamic Programming Methods

### 2.1 Value Iteration

```python
class ValueIteration:
    """Value Iteration algorithm"""

    def __init__(self, mdp: MDP, theta: float = 1e-6):
        self.mdp = mdp
        self.theta = theta

        self.V = np.zeros(mdp.n_states)
        self.policy = np.zeros(mdp.n_states, dtype=int)

    def run(self, max_iterations: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Run value iteration"""
        for iteration in range(max_iterations):
            delta = 0.0

            for state in self.mdp.states:
                v = self.V[state.id]

                # Bellman optimality update
                action_values = []
                for action in self.mdp.actions:
                    action_value = 0.0
                    for next_state in self.mdp.states:
                        prob = self.mdp.get_transition_prob(state, action, next_state)
                        reward = self.mdp.get_reward(state, action, next_state)
                        action_value += prob * (reward + self.mdp.discount_factor * self.V[next_state.id])

                    action_values.append(action_value)

                self.V[state.id] = max(action_values)
                self.policy[state.id] = np.argmax(action_values)

                delta = max(delta, abs(v - self.V[state.id]))

            if delta < self.theta:
                break

        return self.V, self.policy

    def extract_policy(self) -> StochasticPolicy:
        """Extract deterministic policy from value function"""
        policy_matrix = np.zeros((self.mdp.n_states, self.mdp.n_actions))
        for state in self.mdp.states:
            policy_matrix[state.id, self.policy[state.id]] = 1.0

        return StochasticPolicy(policy_matrix, self.mdp.actions)
```

### 2.2 Policy Iteration

```python
class PolicyIteration:
    """Policy Iteration algorithm"""

    def __init__(self, mdp: MDP, theta: float = 1e-6):
        self.mdp = mdp
        self.theta = theta

        self.V = np.zeros(mdp.n_states)
        self.policy_matrix = np.ones((mdp.n_states, mdp.n_actions)) / mdp.n_actions

    def policy_evaluation(self):
        """Evaluate current policy"""
        while True:
            delta = 0.0

            for state in self.mdp.states:
                v = self.V[state.id]

                # Bellman update for current policy
                new_v = 0.0
                for action in self.mdp.actions:
                    action_prob = self.policy_matrix[state.id, action.id]

                    action_value = 0.0
                    for next_state in self.mdp.states:
                        prob = self.mdp.get_transition_prob(state, action, next_state)
                        reward = self.mdp.get_reward(state, action, next_state)
                        action_value += prob * (reward + self.mdp.discount_factor * self.V[next_state.id])

                    new_v += action_prob * action_value

                self.V[state.id] = new_v
                delta = max(delta, abs(v - new_v))

            if delta < self.theta:
                break

    def policy_improvement(self):
        """Improve policy based on current value function"""
        policy_stable = True

        for state in self.mdp.states:
            old_action = np.argmax(self.policy_matrix[state.id, :])

            # Find best action
            action_values = []
            for action in self.mdp.actions:
                action_value = 0.0
                for next_state in self.mdp.states:
                    prob = self.mdp.get_transition_prob(state, action, next_state)
                    reward = self.mdp.get_reward(state, action, next_state)
                    action_value += prob * (reward + self.mdp.discount_factor * self.V[next_state.id])

                action_values.append(action_value)

            best_action = np.argmax(action_values)

            if old_action != best_action:
                policy_stable = False

            # Update policy
            self.policy_matrix[state.id, :] = 0.0
            self.policy_matrix[state.id, best_action] = 1.0

        return policy_stable

    def run(self, max_iterations: int = 1000) -> StochasticPolicy:
        """Run policy iteration"""
        for iteration in range(max_iterations):
            self.policy_evaluation()
            policy_stable = self.policy_improvement()

            if policy_stable:
                break

        return StochasticPolicy(self.policy_matrix, self.mdp.actions)
```

## 3. Temporal Difference Learning

### 3.1 Q-Learning

```python
class QLearning:
    """Q-Learning algorithm"""

    def __init__(self, mdp: MDP, learning_rate: float = 0.1, epsilon: float = 0.1):
        self.mdp = mdp
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        self.Q = np.zeros((mdp.n_states, mdp.n_actions))

    def get_action(self, state: State) -> Action:
        """ε-greedy action selection"""
        if random.random() < self.epsilon:
            # Explore: random action
            action_id = random.randint(0, self.mdp.n_actions - 1)
        else:
            # Exploit: best action
            action_id = np.argmax(self.Q[state.id, :])

        return self.mdp.actions[action_id]

    def update(self, state: State, action: Action, reward: float, next_state: State, done: bool):
        """Q-learning update"""
        current_q = self.Q[state.id, action.id]

        if done:
            target = reward
        else:
            max_next_q = np.max(self.Q[next_state.id, :])
            target = reward + self.mdp.discount_factor * max_next_q

        # Q-learning update
        self.Q[state.id, action.id] += self.learning_rate * (target - current_q)

    def train(self, num_episodes: int, max_steps: int = 1000) -> np.ndarray:
        """Train Q-learning agent"""
        episode_returns = []

        for episode in range(num_episodes):
            state = self.mdp.states[0]  # Start from initial state
            episode_return = 0.0

            for step in range(max_steps):
                action = self.get_action(state)
                next_state, reward = self.mdp.sample_next_state(state, action)

                done = (step == max_steps - 1) or (next_state.id == len(self.mdp.states) - 1)

                self.update(state, action, reward, next_state, done)

                episode_return += reward
                state = next_state

                if done:
                    break

            episode_returns.append(episode_return)

            # Decay epsilon
            self.epsilon *= 0.9995
            self.epsilon = max(0.01, self.epsilon)

        return np.array(episode_returns)

    def extract_policy(self) -> DeterministicPolicy:
        """Extract greedy policy from Q-values"""
        policy_mapping = {}
        for state in self.mdp.states:
            policy_mapping[state.id] = np.argmax(self.Q[state.id, :])

        return DeterministicPolicy(policy_mapping)
```

### 3.2 SARSA

```python
class SARSA:
    """State-Action-Reward-State-Action algorithm"""

    def __init__(self, mdp: MDP, learning_rate: float = 0.1, epsilon: float = 0.1):
        self.mdp = mdp
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        self.Q = np.zeros((mdp.n_states, mdp.n_actions))

    def get_action(self, state: State) -> Action:
        """ε-greedy action selection"""
        if random.random() < self.epsilon:
            action_id = random.randint(0, self.mdp.n_actions - 1)
        else:
            action_id = np.argmax(self.Q[state.id, :])

        return self.mdp.actions[action_id]

    def update(self, state: State, action: Action, reward: float,
               next_state: State, next_action: Action, done: bool):
        """SARSA update"""
        current_q = self.Q[state.id, action.id]

        if done:
            target = reward
        else:
            next_q = self.Q[next_state.id, next_action.id]
            target = reward + self.mdp.discount_factor * next_q

        # SARSA update
        self.Q[state.id, action.id] += self.learning_rate * (target - current_q)

    def train(self, num_episodes: int, max_steps: int = 1000) -> np.ndarray:
        """Train SARSA agent"""
        episode_returns = []

        for episode in range(num_episodes):
            state = self.mdp.states[0]  # Start from initial state
            action = self.get_action(state)
            episode_return = 0.0

            for step in range(max_steps):
                next_state, reward = self.mdp.sample_next_state(state, action)
                done = (step == max_steps - 1) or (next_state.id == len(self.mdp.states) - 1)

                next_action = self.get_action(next_state)

                self.update(state, action, reward, next_state, next_action, done)

                episode_return += reward
                state = next_state
                action = next_action

                if done:
                    break

            episode_returns.append(episode_return)

            # Decay epsilon
            self.epsilon *= 0.9995
            self.epsilon = max(0.01, self.epsilon)

        return np.array(episode_returns)
```

## 4. Deep Reinforcement Learning

### 4.1 Deep Q-Networks (DQN)

```python
class DQNNetwork(nn.Module):
    """Deep Q-Network"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 128]):
        super(DQNNetwork, self).__init__()

        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            ])
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

class ReplayBuffer:
    """Experience replay buffer"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """Add experience to buffer"""
        experience = (state, action, reward, next_state, done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample batch of experiences"""
        batch = random.sample(self.buffer, batch_size)

        states = torch.FloatTensor([exp[0] for exp in batch])
        actions = torch.LongTensor([exp[1] for exp in batch])
        rewards = torch.FloatTensor([exp[2] for exp in batch])
        next_states = torch.FloatTensor([exp[3] for exp in batch])
        dones = torch.BoolTensor([exp[4] for exp in batch])

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)

class DQNAgent:
    """Deep Q-Network agent"""

    def __init__(self, state_dim: int, action_dim: int,
                 learning_rate: float = 1e-3, epsilon: float = 0.1,
                 epsilon_decay: float = 0.995, epsilon_min: float = 0.01,
                 gamma: float = 0.99, buffer_capacity: int = 10000,
                 batch_size: int = 32, target_update_freq: int = 100):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Networks
        self.q_network = DQNNetwork(state_dim, action_dim)
        self.target_network = DQNNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Exploration
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # Training metrics
        self.training_step = 0

    def get_action(self, state: np.ndarray) -> int:
        """ε-greedy action selection"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()

    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool):
        """Update DQN"""
        # Store experience
        self.replay_buffer.push(state, action, reward, next_state, done)

        # Sample batch if buffer is full enough
        if len(self.replay_buffer) > self.batch_size:
            self._train_step()

        # Update exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _train_step(self):
        """Perform one training step"""
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Target Q-values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones.float()) * self.gamma * next_q

        # Compute loss
        loss = F.mse_loss(current_q.squeeze(), target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def train(self, env, num_episodes: int, max_steps: int = 1000) -> List[float]:
        """Train DQN agent"""
        episode_returns = []

        for episode in range(num_episodes):
            state = env.reset()
            episode_return = 0.0

            for step in range(max_steps):
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)

                self.update(state, action, reward, next_state, done)

                episode_return += reward
                state = next_state

                if done:
                    break

            episode_returns.append(episode_return)

        return episode_returns
```

### 4.2 Policy Gradient Methods

```python
class PolicyNetwork(nn.Module):
    """Policy network for policy gradient methods"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 128]):
        super(PolicyNetwork, self).__init__()

        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            ])
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

class REINFORCEAgent:
    """REINFORCE policy gradient agent"""

    def __init__(self, state_dim: int, action_dim: int,
                 learning_rate: float = 1e-3, gamma: float = 0.99):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        self.policy_network = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

    def get_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor]:
        """Sample action from policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = F.softmax(self.policy_network(state_tensor), dim=-1)

        action_dist = Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)

        return action.item(), log_prob

    def update(self, episode_data: List[Tuple[np.ndarray, int, float, torch.Tensor]]):
        """Update policy using REINFORCE"""
        states, actions, rewards, log_probs = zip(*episode_data)

        # Calculate discounted returns
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Calculate policy gradient loss
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)

        policy_loss = torch.stack(policy_loss).sum()

        # Optimize
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

    def train(self, env, num_episodes: int, max_steps: int = 1000) -> List[float]:
        """Train REINFORCE agent"""
        episode_returns = []

        for episode in range(num_episodes):
            state = env.reset()
            episode_data = []
            episode_return = 0.0

            for step in range(max_steps):
                action, log_prob = self.get_action(state)
                next_state, reward, done, _ = env.step(action)

                episode_data.append((state, action, reward, log_prob))
                episode_return += reward
                state = next_state

                if done:
                    break

            self.update(episode_data)
            episode_returns.append(episode_return)

        return episode_returns
```

### 4.3 Actor-Critic Methods

```python
class ActorNetwork(nn.Module):
    """Actor network for policy"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 128]):
        super(ActorNetwork, self).__init__()

        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            ])
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

class CriticNetwork(nn.Module):
    """Critic network for value function"""

    def __init__(self, state_dim: int, hidden_dims: List[int] = [128, 128]):
        super(CriticNetwork, self).__init__()

        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            ])
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

class A2CAgent:
    """Advantage Actor-Critic agent"""

    def __init__(self, state_dim: int, action_dim: int,
                 actor_lr: float = 1e-3, critic_lr: float = 1e-3,
                 gamma: float = 0.99, entropy_coef: float = 0.01):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.entropy_coef = entropy_coef

        # Networks
        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def get_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Sample action from policy and get value estimate"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # Get action probabilities
        action_probs = F.softmax(self.actor(state_tensor), dim=-1)

        # Get value estimate
        value = self.critic(state_tensor)

        # Sample action
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        return action.item(), log_prob.squeeze(), value.squeeze(), entropy.squeeze()

    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool, log_prob: torch.Tensor,
               value: torch.Tensor, entropy: torch.Tensor):

        # Calculate advantage
        with torch.no_grad():
            next_value = self.critic(torch.FloatTensor(next_state).unsqueeze(0))
            if done:
                advantage = reward - value
            else:
                advantage = reward + self.gamma * next_value - value

        # Actor loss
        actor_loss = -(log_prob * advantage + self.entropy_coef * entropy)

        # Critic loss
        if done:
            target = torch.tensor(reward, dtype=torch.float32)
        else:
            target = reward + self.gamma * next_value

        critic_loss = F.mse_loss(value, target)

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def train(self, env, num_episodes: int, max_steps: int = 1000) -> List[float]:
        """Train A2C agent"""
        episode_returns = []

        for episode in range(num_episodes):
            state = env.reset()
            episode_return = 0.0

            for step in range(max_steps):
                action, log_prob, value, entropy = self.get_action(state)
                next_state, reward, done, _ = env.step(action)

                self.update(state, action, reward, next_state, done,
                           log_prob, value, entropy)

                episode_return += reward
                state = next_state

                if done:
                    break

            episode_returns.append(episode_return)

        return episode_returns
```

## 5. Advanced RL Algorithms

### 5.1 Proximal Policy Optimization (PPO)

```python
class PPONetwork(nn.Module):
    """Combined Actor-Critic network for PPO"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [64, 64]):
        super(PPONetwork, self).__init__()

        # Shared layers
        shared_layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            shared_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh()
            ])
            input_dim = hidden_dim

        self.shared = nn.Sequential(*shared_layers)

        # Actor head
        self.actor = nn.Linear(hidden_dims[-1], action_dim)

        # Critic head
        self.critic = nn.Linear(hidden_dims[-1], 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(state)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value

class PPOBuffer:
    """Buffer for PPO experience collection"""

    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.advantages = []
        self.returns = []

    def clear(self):
        """Clear buffer"""
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
        self.advantages.clear()
        self.returns.clear()

    def push(self, state, action, log_prob, value, reward, done):
        """Add experience to buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_advantages_and_returns(self, last_value, gamma=0.99, gae_lambda=0.95):
        """Compute advantages and returns using GAE"""
        advantages = []
        returns = []

        gae = 0
        for i in reversed(range(len(self.rewards))):
            if i == len(self.rewards) - 1:
                next_value = last_value
                next_done = 0
            else:
                next_value = self.values[i + 1]
                next_done = self.dones[i + 1]

            delta = self.rewards[i] + gamma * next_value * (1 - next_done) - self.values[i]
            gae = delta + gamma * gae_lambda * (1 - next_done) * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + torch.tensor(self.values, dtype=torch.float32)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.advantages = advantages
        self.returns = returns

class PPOAgent:
    """Proximal Policy Optimization agent"""

    def __init__(self, state_dim: int, action_dim: int,
                 lr: float = 3e-4, gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2, entropy_coef: float = 0.01,
                 value_coef: float = 0.5, max_grad_norm: float = 0.5,
                 ppo_epochs: int = 10, batch_size: int = 64):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size

        # Network
        self.network = PPONetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # Buffer
        self.buffer = PPOBuffer()

    def get_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Get action from policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            action_probs, value = self.network(state_tensor)

        action_dist = Categorical(F.softmax(action_probs, dim=-1))
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    def update(self):
        """Update policy using PPO"""
        # Convert buffer to tensors
        states = torch.FloatTensor(self.buffer.states)
        actions = torch.LongTensor(self.buffer.actions)
        old_log_probs = torch.FloatTensor(self.buffer.log_probs)
        returns = torch.FloatTensor(self.buffer.returns)
        advantages = torch.FloatTensor(self.buffer.advantages)

        # PPO update for multiple epochs
        for _ in range(self.ppo_epochs):
            # Create mini-batches
            indices = torch.randperm(len(states))
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # Forward pass
                action_probs, values = self.network(batch_states)
                action_dist = Categorical(F.softmax(action_probs, dim=-1))
                new_log_probs = action_dist.log_prob(batch_actions)
                entropy = action_dist.entropy()

                # Calculate ratios
                ratios = torch.exp(new_log_probs - batch_old_log_probs)

                # PPO loss
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # Optimize
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

        self.buffer.clear()

    def train(self, env, num_episodes: int, max_steps: int = 1000) -> List[float]:
        """Train PPO agent"""
        episode_returns = []

        for episode in range(num_episodes):
            state = env.reset()
            episode_return = 0.0

            for step in range(max_steps):
                action, log_prob, value = self.get_action(state)
                next_state, reward, done, _ = env.step(action)

                self.buffer.push(state, action, log_prob, value, reward, done)

                episode_return += reward
                state = next_state

                if done:
                    # Compute advantages and returns
                    last_value = self.network.critic(self.network.shared(torch.FloatTensor(state).unsqueeze(0))).item()
                    self.buffer.compute_advantages_and_returns(last_value, self.gamma, self.gae_lambda)

                    # Update policy
                    self.update()
                    break

            episode_returns.append(episode_return)

        return episode_returns
```

This comprehensive theoretical foundation covers the essential mathematical principles and algorithmic implementations of reinforcement learning, from basic MDPs to advanced deep RL algorithms like PPO, providing the necessary foundation for understanding and implementing autonomous learning agents.