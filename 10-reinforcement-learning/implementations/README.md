# RL Implementations

Reusable code for RL experiments.

## Structure

```
implementations/
├── agents/          # Agent implementations
├── environments/    # Custom environments
├── utils/           # Utilities and helpers
└── requirements.txt # Dependencies
```

## Usage

```python
from implementations.agents import DQNAgent
from implementations.environments import GridWorld
from implementations.utils import ReplayBuffer

# Create environment and agent
env = GridWorld()
agent = DQNAgent(state_dim=16, action_dim=4)

# Training loop
for episode in range(1000):
    state = env.reset()
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
```

## Dependencies

See `requirements.txt` for full list.
