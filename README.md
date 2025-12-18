# Pacman AI: Search, Multi-Agent, and Reinforcement Learning

A comprehensive implementation of AI agents for the classic Pacman game, featuring search algorithms, adversarial agents, probabilistic inference, and reinforcement learning. Based on UC Berkeley's CS188 Intro to AI course projects.

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![AI](https://img.shields.io/badge/AI-Search%20%7C%20RL%20%7C%20Inference-green.svg)

## Project Overview

This project implements intelligent Pacman agents across four major AI domains:
1. **Search Algorithms** - Navigate mazes efficiently using uninformed and informed search
2. **Multi-Agent Systems** - Handle adversarial ghosts using game theory
3. **Probabilistic Inference** - Track hidden ghosts using Bayesian reasoning
4. **Reinforcement Learning** - Learn optimal policies through experience

## Project Components

### Part 1: Search Algorithms (`search.py`)

Implements fundamental search algorithms for pathfinding in Pacman mazes.

#### Algorithms Implemented

**Depth-First Search (DFS)**
- Uses stack (LIFO) for frontier management
- Graph search with visited set to prevent cycles
- Explores deepest nodes first
- Complete but not optimal

**Breadth-First Search (BFS)**
- Uses queue (FIFO) for frontier management
- Explores shallowest nodes first
- Guarantees shortest path for unweighted graphs
- Optimal for uniform step costs

**Uniform Cost Search (UCS)**
- Uses priority queue ordered by path cost
- Expands least-cost nodes first
- Optimal for graphs with varying step costs
- Dijkstra's algorithm for pathfinding

**A* Search**
- Uses priority queue ordered by f(n) = g(n) + h(n)
- g(n): cost from start to n
- h(n): heuristic estimate from n to goal
- Optimal with admissible heuristics
- Most efficient informed search

**Key Features:**
- Graph search implementation (avoids revisiting states)
- Consistent action path reconstruction
- Efficient data structure usage (Stack, Queue, PriorityQueue)
- Support for arbitrary search problems

### Part 2: Multi-Agent Search (`multiAgents.py`)

Implements adversarial search agents for playing against ghost opponents.

#### Reflex Agent
**Evaluation Function Design:**
- Considers distance to nearest food
- Avoids active ghosts (penalty for proximity)
- Pursues scared ghosts (reward for proximity)
- Penalizes stopping actions
- Rewards eating food (reduces food count)
- Balances immediate rewards vs strategic positioning

#### Minimax Agent
**Algorithm:**
- Recursive tree search alternating between max (Pacman) and min (ghosts) layers
- Depth-limited search to specified depth
- Assumes optimal adversarial play
- Terminal states: win, lose, or depth limit reached

**Properties:**
- Guaranteed optimal against optimal opponents
- Complete game tree exploration to depth limit
- Time complexity: O(b^m) where b=branching factor, m=depth

#### Alpha-Beta Pruning Agent
**Optimizations:**
- Prunes branches that cannot affect final decision
- Maintains alpha (best max value) and beta (best min value)
- Cuts off search when beta ≤ alpha
- Same result as minimax but faster

**Performance:**
- Reduces nodes examined from O(b^m) to O(b^(m/2)) in best case
- Enables deeper search in same time
- Move ordering affects pruning effectiveness

#### Expectimax Agent
**Probabilistic Modeling:**
- Models suboptimal opponents with random behavior
- Max nodes: choose best action (Pacman)
- Expectation nodes: weighted average (ghosts)
- Useful when opponents don't play optimally

**Evaluation Function (`betterEvaluationFunction`):**
- Multi-factor state evaluation:
  - Game score baseline
  - Inverse distance to nearest food (1.5 weight)
  - Penalty for remaining food count (-2.0 per food)
  - Penalty for remaining capsules (-3.0 per capsule)
  - Inverse distance to nearest capsule (0.5 weight)
  - Ghost proximity handling:
    - Scared ghosts: attraction (4.0 weight)
    - Active ghosts: strong repulsion when close (-6.0 weight)
    - Weak repulsion at distance (-0.2 weight)

### Part 3: Ghostbusters - Probabilistic Inference (`inference.py`, `bustersAgents.py`)

Tracks hidden ghosts using probabilistic reasoning and noisy distance observations.

#### Exact Inference (`ExactInference`)

**Belief Distribution:**
- Maintains probability distribution over all possible ghost positions
- Discrete distribution over legal positions

**Observation Update:**
- Applies Bayes' rule: P(position | observation) ∝ P(observation | position) × P(position)
- Uses noisy Manhattan distance observations
- Likelihood based on distance observation model
- Handles special case: ghost in jail (observation = None)

**Time Elapse:**
- Predicts belief after one time step
- Uses ghost transition model (position distribution helper)
- Accounts for Pacman's movement affecting capture probability
- Marginalizes over all possible ghost actions

**Algorithm:**
```python
# Observation update
for each position:
    belief[position] *= P(observation | position)
normalize beliefs

# Time elapse
newBeliefs = {}
for each old_position with non-zero belief:
    positionDist = getPositionDistribution(old_position)
    for new_position, prob in positionDist:
        newBeliefs[new_position] += belief[old_position] * prob
beliefs = newBeliefs
```

#### Greedy Busters Agent

**Strategy:**
- Uses most likely ghost position from belief distribution (argMax)
- Chooses action minimizing maze distance to closest ghost
- Uses pre-computed Distancer for efficient pathfinding
- Tie-breaking by lexicographic action ordering

### Part 4: Reinforcement Learning (`qlearningAgents.py`, `valueIterationAgents.py`)

Learns optimal policies through trial and error.

#### Value Iteration (`ValueIterationAgent`)

**Algorithm:**
- Computes optimal value function V*(s)
- Iterative updates: V_{k+1}(s) = max_a Σ P(s'|s,a)[R(s,a,s') + γV_k(s')]
- Runs for fixed number of iterations
- Derives policy from value function: π*(s) = argmax_a Q*(s,a)

**Q-Value Computation:**
```python
Q(s,a) = Σ P(s'|s,a)[R(s,a,s') + γV(s')]
```

**Asynchronous Value Iteration:**
- Updates one state per iteration (cyclic)
- More efficient for large state spaces
- Same convergence guarantees

**Prioritized Sweeping:**
- Maintains priority queue of states by Bellman error
- Updates states in order of importance
- Propagates changes backward through state space
- Computes predecessor relationships for efficient updates

#### Q-Learning (`QLearningAgent`)

**Algorithm:**
- Model-free temporal difference learning
- Updates Q-values based on experience: 
  - Q(s,a) ← (1-α)Q(s,a) + α[r + γ max_a' Q(s',a')]
- ε-greedy exploration: random action with probability ε
- Converges to optimal Q* with sufficient exploration

**Parameters:**
- α (alpha): learning rate (0.2)
- γ (gamma): discount factor (0.8)
- ε (epsilon): exploration rate (0.05)

**Approximate Q-Learning:**
- Uses feature-based representation: Q(s,a) = Σ w_i f_i(s,a)
- Weight updates: w_i ← w_i + α[r + γV(s') - Q(s,a)]f_i(s,a)
- Generalizes across similar states
- Scales to large state spaces
- Feature extractors: closest food, ghost distances, etc.

## Technical Implementation

### Data Structures

**Search:**
- Stack (DFS): LIFO frontier
- Queue (BFS): FIFO frontier  
- PriorityQueue (UCS, A*): Min-heap ordered by priority
- Sets for visited state tracking
- Cost dictionaries for optimal path tracking

**Inference:**
- DiscreteDistribution: Probability distributions (extends dict)
- Belief state: Distribution over positions
- Observation models: Conditional probability tables

**Reinforcement Learning:**
- util.Counter: Q-value tables (default 0.0)
- Weight vectors for approximate Q-learning
- Feature extractors for state representation

### Algorithm Complexities

| Algorithm | Time | Space | Optimality |
|-----------|------|-------|------------|
| DFS | O(b^m) | O(bm) | No |
| BFS | O(b^d) | O(b^d) | Yes (uniform cost) |
| UCS | O(b^d) | O(b^d) | Yes |
| A* | O(b^d) | O(b^d) | Yes (admissible h) |
| Minimax | O(b^m) | O(bm) | Yes (vs optimal) |
| Alpha-Beta | O(b^(m/2)) | O(bm) | Yes (vs optimal) |
| Expectimax | O(b^m) | O(bm) | Expected value |

Where: b=branching factor, m=max depth, d=solution depth

## Project Structure
```
pacman-ai/
├── search.py                  # Search algorithms (DFS, BFS, UCS, A*)
├── multiAgents.py             # Adversarial agents (Minimax, Alpha-Beta, Expectimax)
├── inference.py               # Probabilistic inference (Exact, Particle Filter)
├── bustersAgents.py           # Ghost hunting agents
├── qlearningAgents.py         # Q-Learning and Approximate Q-Learning
├── valueIterationAgents.py    # Value Iteration and variants
├── analysis.py                # MDP parameter tuning
├── game.py                    # Pacman game logic (provided)
├── util.py                    # Data structures (provided)
└── README.md                  # Documentation
```

## Running the Agents

### Search Agents
```bash
# DFS
python pacman.py -l mediumMaze -p SearchAgent -a fn=dfs

# BFS
python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs

# UCS
python pacman.py -l mediumMaze -p SearchAgent -a fn=ucs

# A* with Manhattan heuristic
python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
```

### Multi-Agent Games
```bash
# Minimax
python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=4

# Alpha-Beta
python pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic

# Expectimax
python pacman.py -p ExpectimaxAgent -l minimaxClassic -a depth=3

# Better evaluation
python pacman.py -p AlphaBetaAgent -l smallClassic -a evalFn=better
```

### Ghostbusters (Inference)
```bash
# Exact inference
python busters.py -p GreedyBustersAgent -l trickyClassic

# Watch beliefs update
python busters.py -p GreedyBustersAgent -l trickyClassic --frameTime=0.1
```

### Reinforcement Learning
```bash
# Value Iteration
python gridworld.py -a value -i 100 -g BridgeGrid

# Q-Learning
python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid

# Approximate Q-Learning
python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumGrid
```

## Key Learning Outcomes

### Algorithms & Theory
- Search algorithm implementation and analysis
- Adversarial search and game theory
- Probabilistic reasoning and Bayesian inference
- Markov Decision Processes (MDPs)
- Reinforcement learning fundamentals
- Value iteration and policy derivation
- Temporal difference learning

### Programming Skills
- Python algorithm implementation
- Recursive problem solving
- Data structure selection and usage
- Code optimization techniques
- Debugging complex AI systems

### Problem-Solving
- State space representation
- Heuristic design
- Evaluation function crafting
- Feature engineering
- Parameter tuning
- Balancing exploration vs exploitation

## Performance Benchmarks

### Search (mediumMaze)
- DFS: ~370 nodes expanded
- BFS: ~269 nodes expanded (optimal path)
- UCS: ~269 nodes expanded
- A*: ~221 nodes expanded (best performance)

### Multi-Agent (smallClassic)
- Minimax (depth=3): Average score ~800
- Alpha-Beta (depth=3): Same score, 60% fewer nodes
- Expectimax (depth=3): Average score ~900

### Reinforcement Learning (smallGrid)
- Q-Learning: 90% win rate after 2000 training episodes
- Approximate Q-Learning: 95% win rate with feature generalization

## AI Concepts Demonstrated

✅ **Search**: Uninformed/informed search, heuristics, optimality  
✅ **Game Playing**: Minimax, alpha-beta pruning, evaluation functions  
✅ **Probability**: Bayesian inference, belief updates, observation models  
✅ **Learning**: Value iteration, Q-learning, function approximation  
✅ **Decision Making**: MDPs, policies, expected utility  
✅ **Uncertainty**: Probabilistic reasoning, noisy observations

## Academic Context

**Course**: Introduction to Artificial Intelligence  
**Institution**: UC Berkeley (adapted for coursework)  
**Topics Covered**:
- Search and Planning
- Game Playing and Adversarial Search
- Probabilistic Reasoning
- Markov Decision Processes
- Reinforcement Learning

## Future Enhancements

- [ ] Monte Carlo Tree Search (MCTS) agent
- [ ] Deep Q-Network (DQN) with neural networks
- [ ] Policy gradient methods
- [ ] Multi-agent reinforcement learning
- [ ] Transfer learning between layouts
- [ ] Visualization dashboard for learning progress

## License

This project is based on UC Berkeley's CS188 course materials. Educational use only.

## Attribution

The Pacman AI projects were developed at UC Berkeley by John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu). Student-side autograding was added by Brad Miller, Nick Hay, and Pieter Abbeel (pabbeel@cs.berkeley.edu).

## Contact

**Samantha Katovich**
- Email: samanthakatovich@gmail.com
- GitHub: [@samanthakatovich](https://github.com/samanthakatovich)
- Portfolio: [samanthakatovich.github.io](https://samanthakatovich.github.io)

---

*Part of my AI/ML and Data Science portfolio at Purdue University*
```

## Additional Files

### `.gitignore`
```
__pycache__/
*.pyc
*.pyo
.Python
*.swp
*.swo

# Layouts and test cases (if not uploading)
layouts/
test_cases/

# Generated files
*.replay
*.solution

# IDE
.vscode/
.idea/
*.code-workspace

# OS
.DS_Store
Thumbs.db
```

### `requirements.txt`
```
# No external dependencies required
# Uses Python standard library only
