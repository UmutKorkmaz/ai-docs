---
title: "Ai Agents And Autonomous - AI Agents and Autonomous Systems"
description: "## Overview. Comprehensive guide covering reinforcement learning, algorithm. Part of AI documentation system with 1500+ topics."
keywords: "reinforcement learning, algorithm, reinforcement learning, algorithm, artificial intelligence, machine learning, AI documentation"
author: "AI Documentation Team"
---

# AI Agents and Autonomous Systems Foundations

## Overview
AI agents are autonomous systems that perceive their environment, make decisions, and take actions to achieve specific goals. This section covers the theoretical foundations, architectures, and principles behind intelligent agents and autonomous systems.

## 1. Agent Theory and Fundamentals

### 1.1 What is an AI Agent?

An AI agent is a system that:
- **Perceives**: Gathers information from its environment through sensors
- **Reasons**: Processes information and makes decisions based on goals
- **Acts**: Takes actions to affect the environment
- **Learns**: Improves performance through experience

**Formal Definition:**
An agent is a tuple $(A, E, P, A, G)$ where:
- $A$: Agent architecture and capabilities
- $E$: Environment the agent operates in
- $P$: Perception function mapping environment states to observations
- $A$: Action selection function mapping observations to actions
- $G$: Goals the agent tries to achieve

### 1.2 Agent Properties

**Autonomy**: Agents operate without direct human intervention
**Reactivity**: Respond to timely changes in the environment
**Proactivity**: Take initiative to achieve goals
**Social Ability**: Interact with other agents or humans
**Learning**: Improve performance over time
**Adaptability**: Adjust to changing environments

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import random
from dataclasses import dataclass
from enum import Enum

class AgentType(Enum):
    REACTIVE = "reactive"
    DELIBERATIVE = "deliberative"
    HYBRID = "hybrid"
    LEARNING = "learning"
    MULTI_AGENT = "multi_agent"

@dataclass
class Percept:
    """Perceptual information received by agent"""
    content: Any
    timestamp: float
    source: str
    confidence: float = 1.0

@dataclass
class Action:
    """Action that agent can perform"""
    name: str
    parameters: Dict[str, Any]
    cost: float = 0.0
    duration: float = 0.0

@dataclass
class Goal:
    """Goal that agent tries to achieve"""
    description: str
    priority: float
    deadline: Optional[float] = None
    completion_criteria: Optional[callable] = None

class Environment:
    """Environment where agents operate"""
    def __init__(self, name: str):
        self.name = name
        self.state: Dict[str, Any] = {}
        self.agents: List['Agent'] = []
        self.time: float = 0.0

    def add_agent(self, agent: 'Agent'):
        """Add agent to environment"""
        self.agents.append(agent)
        agent.environment = self

    def remove_agent(self, agent: 'Agent'):
        """Remove agent from environment"""
        if agent in self.agents:
            self.agents.remove(agent)
            agent.environment = None

    def get_state(self) -> Dict[str, Any]:
        """Get current environment state"""
        return self.state.copy()

    def update_state(self, changes: Dict[str, Any]):
        """Update environment state"""
        self.state.update(changes)
        self.time += 1.0

    def is_valid_action(self, action: Action) -> bool:
        """Check if action is valid in current state"""
        return True  # Override in subclasses

class Agent(ABC):
    """Abstract base class for AI agents"""

    def __init__(self, name: str, agent_type: AgentType):
        self.name = name
        self.agent_type = agent_type
        self.environment: Optional[Environment] = None
        self.goals: List[Goal] = []
        self.memory: List[Percept] = []
        self.actions_history: List[Tuple[Percept, Action]] = []

        # Agent attributes
        self.autonomy_level: float = 1.0  # 0 = human-controlled, 1 = fully autonomous
        self.learning_enabled: bool = True
        self.adaptation_rate: float = 0.1

    @abstractmethod
    def perceive(self) -> List[Percept]:
        """Perceive environment and gather information"""
        pass

    @abstractmethod
    def deliberate(self, percepts: List[Percept]) -> Action:
        """Deliberate and decide on action"""
        pass

    @abstractmethod
    def act(self, action: Action) -> bool:
        """Execute action in environment"""
        pass

    def add_goal(self, goal: Goal):
        """Add goal to agent's goal list"""
        self.goals.append(goal)
        self.goals.sort(key=lambda g: g.priority, reverse=True)

    def remove_goal(self, goal: Goal):
        """Remove goal from agent's goal list"""
        if goal in self.goals:
            self.goals.remove(goal)

    def update_memory(self, percepts: List[Percept]):
        """Update agent's memory with new percepts"""
        self.memory.extend(percepts)

        # Limit memory size (forget old information)
        max_memory = 1000
        if len(self.memory) > max_memory:
            self.memory = self.memory[-max_memory:]

    def run_cycle(self):
        """Run one complete agent cycle"""
        if not self.environment:
            return

        # Perceive environment
        percepts = self.perceive()

        # Update memory
        self.update_memory(percepts)

        # Deliberate and decide
        action = self.deliberate(percepts)

        # Execute action
        success = self.act(action)

        # Record action
        if success:
            self.actions_history.append((percepts[-1] if percepts else None, action))

        return success
```

## 2. Agent Architectures

### 2.1 Reactive Agents

**Simple Reflex Agents**: React directly to current percepts without internal state

```python
class ReactiveAgent(Agent):
    """Simple reactive agent that responds to current percepts"""

    def __init__(self, name: str, action_rules: Dict[str, Action]):
        super().__init__(name, AgentType.REACTIVE)
        self.action_rules = action_rules

    def perceive(self) -> List[Percept]:
        """Simple perception - get environment state"""
        if not self.environment:
            return []

        # Create percepts from environment state
        percepts = []
        for key, value in self.environment.get_state().items():
            percept = Percept(
                content={key: value},
                timestamp=self.environment.time,
                source="environment"
            )
            percepts.append(percept)

        return percepts

    def deliberate(self, percepts: List[Percept]) -> Action:
        """Simple rule-based decision making"""
        if not percepts:
            return Action("wait", {})

        # Find matching rule
        for percept in percepts:
            percept_key = list(percept.content.keys())[0]
            percept_value = percept.content[percept_key]

            # Simple rule matching
            if f"{percept_key}_{percept_value}" in self.action_rules:
                return self.action_rules[f"{percept_key}_{percept_value}"]

        # Default action
        return Action("wait", {})

    def act(self, action: Action) -> bool:
        """Execute action"""
        if not self.environment:
            return False

        # Check if action is valid
        if not self.environment.is_valid_action(action):
            return False

        # Apply action effect (simplified)
        action_effects = {
            "move": {"position": "new_position"},
            "turn": {"orientation": "new_orientation"},
            "wait": {}
        }

        if action.name in action_effects:
            self.environment.update_state(action_effects[action.name])
            return True

        return False
```

### 2.2 Deliberative Agents

**Goal-Based Agents**: Maintain internal state and plan actions to achieve goals

```python
class DeliberativeAgent(Agent):
    """Deliberative agent with planning capabilities"""

    def __init__(self, name: str, planner=None):
        super().__init__(name, AgentType.DELIBERATIVE)
        self.planner = planner or SimplePlanner()
        self.internal_state: Dict[str, Any] = {}
        self.current_plan: List[Action] = []

    def perceive(self) -> List[Percept]:
        """Enhanced perception with internal state integration"""
        percepts = super().perceive()

        # Add internal state percepts
        if self.internal_state:
            internal_percept = Percept(
                content=self.internal_state.copy(),
                timestamp=self.environment.time,
                source="internal"
            )
            percepts.append(internal_percept)

        return percepts

    def deliberate(self, percepts: List[Percept]) -> Action:
        """Plan actions to achieve goals"""
        if not self.goals:
            return Action("wait", {})

        # Update internal state from percepts
        self.update_internal_state(percepts)

        # If no current plan, create one
        if not self.current_plan:
            self.current_plan = self.planner.plan(
                self.internal_state,
                self.goals[0],  # Focus on highest priority goal
                self.get_available_actions()
            )

        # Execute next action from plan
        if self.current_plan:
            action = self.current_plan.pop(0)
            return action
        else:
            return Action("wait", {})

    def update_internal_state(self, percepts: List[Percept]):
        """Update internal state based on percepts"""
        for percept in percepts:
            if percept.source == "environment":
                self.internal_state.update(percept.content)

    def get_available_actions(self) -> List[Action]:
        """Get list of available actions"""
        return [
            Action("move", {"direction": "north"}),
            Action("move", {"direction": "south"}),
            Action("move", {"direction": "east"}),
            Action("move", {"direction": "west"}),
            Action("turn", {"direction": "left"}),
            Action("turn", {"direction": "right"}),
            Action("wait", {})
        ]

class SimplePlanner:
    """Simple forward-chaining planner"""

    def __init__(self, max_depth: int = 10):
        self.max_depth = max_depth

    def plan(self, current_state: Dict[str, Any], goal: Goal,
             available_actions: List[Action]) -> List[Action]:
        """Plan actions to achieve goal"""

        # Simple breadth-first search for demonstration
        from collections import deque

        queue = deque([(current_state.copy(), [])])
        visited = {str(current_state)}

        while queue and len(queue[0][1]) < self.max_depth:
            state, plan = queue.popleft()

            # Check if goal is achieved
            if self.goal_achieved(state, goal):
                return plan

            # Generate successor states
            for action in available_actions:
                new_state = self.apply_action(state, action)
                state_str = str(new_state)

                if state_str not in visited:
                    visited.add(state_str)
                    new_plan = plan + [action]
                    queue.append((new_state, new_plan))

        return []  # No plan found

    def apply_action(self, state: Dict[str, Any], action: Action) -> Dict[str, Any]:
        """Apply action to state (simplified)"""
        new_state = state.copy()

        # Simple action effects
        if action.name == "move":
            direction = action.parameters.get("direction", "north")
            if "position" in new_state:
                x, y = new_state["position"]
                if direction == "north":
                    new_state["position"] = (x, y + 1)
                elif direction == "south":
                    new_state["position"] = (x, y - 1)
                elif direction == "east":
                    new_state["position"] = (x + 1, y)
                elif direction == "west":
                    new_state["position"] = (x - 1, y)

        return new_state

    def goal_achieved(self, state: Dict[str, Any], goal: Goal) -> bool:
        """Check if goal is achieved in state"""
        # Simplified goal checking
        if goal.completion_criteria:
            return goal.completion_criteria(state)
        else:
            # Default: assume goal is not achieved
            return False
```

### 2.3 Learning Agents

**Adaptive Agents**: Learn from experience to improve performance

```python
class LearningAgent(Agent):
    """Agent that learns from experience"""

    def __init__(self, name: str, learning_algorithm="q_learning"):
        super().__init__(name, AgentType.LEARNING)
        self.learning_algorithm = learning_algorithm
        self.q_table: Dict[str, Dict[str, float]] = {}  # Q-table for Q-learning
        self.learning_rate: float = 0.1
        self.discount_factor: float = 0.9
        self.exploration_rate: float = 0.1

    def perceive(self) -> List[Percept]:
        """Perceive environment and learning context"""
        percepts = super().perceive()

        # Add learning-related percepts
        if self.environment:
            learning_percept = Percept(
                content={
                    "exploration_rate": self.exploration_rate,
                    "learning_rate": self.learning_rate,
                    "q_table_size": len(self.q_table)
                },
                timestamp=self.environment.time,
                source="learning_system"
            )
            percepts.append(learning_percept)

        return percepts

    def deliberate(self, percepts: List[Percept]) -> Action:
        """Use learning to select actions"""
        if not self.environment:
            return Action("wait", {})

        # Get current state
        current_state = self.get_state_representation(percepts)

        # Initialize Q-table for new states
        if current_state not in self.q_table:
            self.q_table[current_state] = {}

        # Get available actions
        available_actions = self.get_available_actions()

        # Select action using ε-greedy strategy
        if random.random() < self.exploration_rate:
            # Explore: choose random action
            action = random.choice(available_actions)
        else:
            # Exploit: choose best action
            action = self.get_best_action(current_state, available_actions)

        return action

    def get_state_representation(self, percepts: List[Percept]) -> str:
        """Convert percepts to state representation"""
        # Simple state representation
        state_parts = []
        for percept in percepts:
            if percept.source == "environment":
                state_parts.extend([f"{k}={v}" for k, v in percept.content.items()])
        return "|".join(sorted(state_parts))

    def get_best_action(self, state: str, available_actions: List[Action]) -> Action:
        """Get best action for state using Q-values"""
        # Initialize Q-values for new actions
        for action in available_actions:
            action_key = f"{action.name}_{action.parameters}"
            if action_key not in self.q_table[state]:
                self.q_table[state][action_key] = 0.0

        # Find best action
        best_action_key = max(self.q_table[state].items(), key=lambda x: x[1])[0]

        # Convert key back to action
        for action in available_actions:
            action_key = f"{action.name}_{action.parameters}"
            if action_key == best_action_key:
                return action

        return random.choice(available_actions)

    def learn(self, old_state: str, action: Action, reward: float,
              new_state: str, done: bool = False):
        """Update Q-values based on experience"""
        if not self.learning_enabled:
            return

        action_key = f"{action.name}_{action.parameters}"

        # Initialize Q-values if needed
        if old_state not in self.q_table:
            self.q_table[old_state] = {}
        if new_state not in self.q_table:
            self.q_table[new_state] = {}
        if action_key not in self.q_table[old_state]:
            self.q_table[old_state][action_key] = 0.0

        # Get current Q-value
        current_q = self.q_table[old_state][action_key]

        # Get maximum Q-value for new state
        if self.q_table[new_state]:
            max_future_q = max(self.q_table[new_state].values())
        else:
            max_future_q = 0.0

        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_future_q - current_q
        )

        self.q_table[old_state][action_key] = new_q

        # Decay exploration rate
        self.exploration_rate *= 0.9995
        self.exploration_rate = max(0.01, self.exploration_rate)
```

## 3. Multi-Agent Systems

### 3.1 Multi-Agent Coordination

```python
class MultiAgentEnvironment(Environment):
    """Environment that supports multiple agents"""

    def __init__(self, name: str):
        super().__init__(name)
        self.communication_channel: Dict[str, List[Any]] = {}
        self.resource_allocation: Dict[str, float] = {}

    def broadcast_message(self, sender: str, message: Any, recipients: List[str] = None):
        """Broadcast message to agents"""
        if recipients is None:
            recipients = [agent.name for agent in self.agents]

        for recipient in recipients:
            if recipient not in self.communication_channel:
                self.communication_channel[recipient] = []
            self.communication_channel[recipient].append({
                "sender": sender,
                "message": message,
                "timestamp": self.time
            })

    def get_messages(self, agent_name: str) -> List[Dict[str, Any]]:
        """Get messages for specific agent"""
        messages = self.communication_channel.get(agent_name, [])
        self.communication_channel[agent_name] = []  # Clear messages
        return messages

    def allocate_resource(self, resource_name: str, amount: float, agent_name: str) -> bool:
        """Allocate resource to agent"""
        if resource_name not in self.resource_allocation:
            return False

        if self.resource_allocation[resource_name] >= amount:
            self.resource_allocation[resource_name] -= amount
            return True

        return False

class CooperativeAgent(Agent):
    """Agent that can cooperate with other agents"""

    def __init__(self, name: str):
        super().__init__(name, AgentType.MULTI_AGENT)
        self.cooperation_strategy: str = "negotiation"  # or "contract_net"
        self.known_agents: List[str] = []
        self.trust_levels: Dict[str, float] = {}
        self.shared_goals: List[Goal] = []

    def perceive(self) -> List[Percept]:
        """Enhanced perception with multi-agent information"""
        percepts = super().perceive()

        # Add communication percepts
        if self.environment and isinstance(self.environment, MultiAgentEnvironment):
            messages = self.environment.get_messages(self.name)
            for msg in messages:
                percept = Percept(
                    content=msg,
                    timestamp=self.environment.time,
                    source="communication"
                )
                percepts.append(percept)

        return percepts

    def negotiate_cooperation(self, task: Dict[str, Any], potential_partners: List[str]) -> Optional[str]:
        """Negotiate cooperation with other agents"""
        if not self.environment or not isinstance(self.environment, MultiAgentEnvironment):
            return None

        best_partner = None
        best_offer = None

        for partner in potential_partners:
            if partner in self.trust_levels and self.trust_levels[partner] > 0.5:
                # Send cooperation request
                request = {
                    "type": "cooperation_request",
                    "task": task,
                    "sender": self.name,
                    "deadline": self.environment.time + 10.0
                }

                self.environment.broadcast_message(self.name, request, [partner])

                # Wait for response (simplified)
                # In practice, would implement proper asynchronous negotiation
                return partner

        return best_partner

    def update_trust(self, agent_name: str, outcome: float):
        """Update trust level for another agent"""
        if agent_name not in self.trust_levels:
            self.trust_levels[agent_name] = 0.5

        # Update trust based on outcome
        if outcome > 0.7:  # Positive outcome
            self.trust_levels[agent_name] = min(1.0, self.trust_levels[agent_name] + 0.1)
        elif outcome < 0.3:  # Negative outcome
            self.trust_levels[agent_name] = max(0.0, self.trust_levels[agent_name] - 0.1)
```

### 3.2 Agent Communication Languages

```python
class Message:
    """Communication message between agents"""

    def __init__(self, sender: str, receiver: str, performative: str,
                 content: Any, conversation_id: str = None):
        self.sender = sender
        self.receiver = receiver
        self.performative = performative  # e.g., "request", "inform", "propose"
        self.content = content
        self.conversation_id = conversation_id or f"{sender}_{receiver}_{random.randint(1000, 9999)}"
        self.timestamp: float = 0.0

class CommunicationProtocol:
    """Handles communication between agents"""

    def __init__(self):
        self.conversations: Dict[str, List[Message]] = {}

    def send_message(self, message: Message, environment: MultiAgentEnvironment):
        """Send message through environment"""
        message.timestamp = environment.time
        environment.broadcast_message(
            message.sender,
            {"type": "agent_message", "message": message},
            [message.receiver]
        )

        # Track conversation
        if message.conversation_id not in self.conversations:
            self.conversations[message.conversation_id] = []
        self.conversations[message.conversation_id].append(message)

    def handle_message(self, agent: 'Agent', message: Message):
        """Handle incoming message"""
        message_handlers = {
            "request": self._handle_request,
            "inform": self._handle_inform,
            "propose": self._handle_propose,
            "accept": self._handle_accept,
            "reject": self._handle_reject
        }

        handler = message_handlers.get(message.performative, self._handle_unknown)
        handler(agent, message)

    def _handle_request(self, agent: 'Agent', message: Message):
        """Handle request message"""
        # Simple request handling
        if isinstance(message.content, dict) and "action" in message.content:
            # Generate response
            response = Message(
                sender=agent.name,
                receiver=message.sender,
                performative="inform",
                content={"status": "completed", "result": "success"},
                conversation_id=message.conversation_id
            )
            self.send_message(response, agent.environment)

    def _handle_inform(self, agent: 'Agent', message: Message):
        """Handle inform message"""
        # Process information
        pass

    def _handle_propose(self, agent: 'Agent', message: Message):
        """Handle propose message"""
        # Consider proposal and respond
        accept_prob = 0.7  # Simplified acceptance logic
        if random.random() < accept_prob:
            response = Message(
                sender=agent.name,
                receiver=message.sender,
                performative="accept",
                content={"agreement": True},
                conversation_id=message.conversation_id
            )
        else:
            response = Message(
                sender=agent.name,
                receiver=message.sender,
                performative="reject",
                content={"agreement": False},
                conversation_id=message.conversation_id
            )
        self.send_message(response, agent.environment)

    def _handle_accept(self, agent: 'Agent', message: Message):
        """Handle accept message"""
        # Proposal was accepted
        pass

    def _handle_reject(self, agent: 'Agent', message: Message):
        """Handle reject message"""
        # Proposal was rejected
        pass

    def _handle_unknown(self, agent: 'Agent', message: Message):
        """Handle unknown message type"""
        # Log or handle unknown message
        pass
```

## 4. Agent Decision Making

### 4.1 Utility Theory and Decision Making

```python
class UtilityBasedAgent(Agent):
    """Agent that makes decisions based on utility theory"""

    def __init__(self, name: str):
        super().__init__(name, AgentType.DELIBERATIVE)
        self.utility_functions: Dict[str, callable] = {}
        self.beliefs: Dict[str, float] = {}  # Probabilistic beliefs

    def add_utility_function(self, goal_name: str, utility_func: callable):
        """Add utility function for goal"""
        self.utility_functions[goal_name] = utility_func

    def calculate_expected_utility(self, action: Action, goal: Goal) -> float:
        """Calculate expected utility of action for goal"""
        if goal.description not in self.utility_functions:
            return 0.0

        utility_func = self.utility_functions[goal.description]

        # Expected utility = Σ(outcome_probability × utility(outcome))
        expected_utility = 0.0

        # Get possible outcomes (simplified)
        outcomes = self.get_possible_outcomes(action)

        for outcome, probability in outcomes.items():
            utility = utility_func(outcome, goal)
            expected_utility += probability * utility

        return expected_utility

    def get_possible_outcomes(self, action: Action) -> Dict[Any, float]:
        """Get possible outcomes and their probabilities"""
        # Simplified outcome prediction
        # In practice, would use more sophisticated models

        if action.name == "move":
            return {
                "success": 0.8,
                "failure": 0.2
            }
        elif action.name == "wait":
            return {
                "no_change": 1.0
            }
        else:
            return {
                "unknown": 1.0
            }

    def deliberate(self, percepts: List[Percept]) -> Action:
        """Make decision based on maximum expected utility"""
        if not self.goals:
            return Action("wait", {})

        # Get available actions
        available_actions = self.get_available_actions()

        # Calculate expected utility for each action-goal pair
        best_action = None
        best_utility = -float('inf')

        for action in available_actions:
            total_utility = 0.0

            for goal in self.goals:
                # Weight by goal priority
                utility = self.calculate_expected_utility(action, goal)
                weighted_utility = utility * goal.priority
                total_utility += weighted_utility

            if total_utility > best_utility:
                best_utility = total_utility
                best_action = action

        return best_action or Action("wait", {})
```

### 4.2 BDI Architecture (Belief-Desire-Intention)

```python
class BDIAgent(Agent):
    """Belief-Desire-Intention agent architecture"""

    def __init__(self, name: str):
        super().__init__(name, AgentType.DELIBERATIVE)
        self.beliefs: Dict[str, Any] = {}  # Beliefs about the world
        self.desires: List[Goal] = []      # Desired states
        self.intentions: List[Goal] = []   # Committed goals
        self.plans: Dict[str, List[Action]] = {}  # Plans for achieving intentions

    def update_beliefs(self, percepts: List[Percept]):
        """Update beliefs based on percepts"""
        for percept in percepts:
            if percept.source == "environment":
                # Simple belief update (could be probabilistic)
                for key, value in percept.content.items():
                    self.beliefs[key] = value

    def filter_desires(self) -> List[Goal]:
        """Filter desires to select achievable ones"""
        achievable_desires = []

        for desire in self.desires:
            if self.is_desire_achievable(desire):
                achievable_desires.append(desire)

        return achievable_desires

    def is_desire_achievable(self, desire: Goal) -> bool:
        """Check if desire is achievable given current beliefs"""
        # Simplified achievability check
        if desire.completion_criteria:
            return not desire.completion_criteria(self.beliefs)
        return True

    def commit_to_intentions(self, achievable_desires: List[Goal]):
        """Commit to intentions based on deliberation"""
        # Simple commitment: select top priority desires
        self.intentions = sorted(achievable_desires, key=lambda g: g.priority)[:3]

    def deliberate(self, percepts: List[Percept]) -> Action:
        """BDI deliberation process"""
        # Update beliefs
        self.update_beliefs(percepts)

        # Filter desires
        achievable_desires = self.filter_desires()

        # Commit to intentions
        self.commit_to_intentions(achievable_desires)

        # Select action for top intention
        if self.intentions:
            top_intention = self.intentions[0]
            return self.select_action_for_intention(top_intention)
        else:
            return Action("wait", {})

    def select_action_for_intention(self, intention: Goal) -> Action:
        """Select action to achieve intention"""
        # Check if we already have a plan
        if intention.description in self.plans and self.plans[intention.description]:
            return self.plans[intention.description].pop(0)
        else:
            # Create new plan (simplified)
            plan = self.create_simple_plan(intention)
            self.plans[intention.description] = plan
            return plan[0] if plan else Action("wait", {})

    def create_simple_plan(self, intention: Goal) -> List[Action]:
        """Create simple plan for intention"""
        # Very simple planning - just return basic actions
        return [Action("move", {"direction": "north"})]
```

## 5. Agent Memory and Learning

### 5.1 Episodic Memory

```python
class EpisodicMemory:
    """Episodic memory for agents"""

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.episodes: List[Dict[str, Any]] = []
        self.current_episode: Optional[Dict[str, Any]] = None

    def start_episode(self, context: Dict[str, Any]):
        """Start new episode"""
        self.current_episode = {
            "context": context.copy(),
            "events": [],
            "start_time": time.time(),
            "end_time": None
        }

    def add_event(self, event: Dict[str, Any]):
        """Add event to current episode"""
        if self.current_episode:
            event["timestamp"] = time.time()
            self.current_episode["events"].append(event)

    def end_episode(self, outcome: Dict[str, Any]):
        """End current episode"""
        if self.current_episode:
            self.current_episode["end_time"] = time.time()
            self.current_episode["outcome"] = outcome

            # Add to memory
            self.episodes.append(self.current_episode.copy())

            # Maintain capacity
            if len(self.episodes) > self.capacity:
                self.episodes = self.episodes[-self.capacity:]

            self.current_episode = None

    def retrieve_similar_episodes(self, query_context: Dict[str, Any],
                                 k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve episodes similar to query context"""
        similarities = []

        for episode in self.episodes:
            similarity = self.calculate_similarity(query_context, episode["context"])
            similarities.append((similarity, episode))

        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [episode for _, episode in similarities[:k]]

    def calculate_similarity(self, context1: Dict[str, Any],
                           context2: Dict[str, Any]) -> float:
        """Calculate similarity between contexts"""
        # Simple Jaccard similarity
        keys1 = set(context1.keys())
        keys2 = set(context2.keys())
        intersection = keys1.intersection(keys2)
        union = keys1.union(keys2)

        if len(union) == 0:
            return 0.0

        return len(intersection) / len(union)
```

### 5.2 Working Memory

```python
class WorkingMemory:
    """Working memory for agent's current thoughts"""

    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self.items: List[Dict[str, Any]] = []
        self.attention_focus: Optional[int] = None

    def add_item(self, item: Dict[str, Any]):
        """Add item to working memory"""
        # Add to beginning of list
        self.items.insert(0, item)

        # Maintain capacity
        if len(self.items) > self.capacity:
            self.items = self.items[:self.capacity]

    def remove_item(self, item: Dict[str, Any]):
        """Remove item from working memory"""
        if item in self.items:
            self.items.remove(item)

    def focus_on(self, item_index: int):
        """Focus attention on specific item"""
        if 0 <= item_index < len(self.items):
            self.attention_focus = item_index

    def get_focused_item(self) -> Optional[Dict[str, Any]]:
        """Get currently focused item"""
        if self.attention_focus is not None:
            return self.items[self.attention_focus]
        return None

    def search_items(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search for items matching query"""
        results = []

        for item in self.items:
            match = True
            for key, value in query.items():
                if key not in item or item[key] != value:
                    match = False
                    break

            if match:
                results.append(item)

        return results
```

This comprehensive foundation provides the theoretical and practical basis for understanding AI agents, from basic reactive agents to sophisticated multi-agent systems with communication, learning, and memory capabilities. These concepts form the foundation for more advanced topics in reinforcement learning and autonomous systems.