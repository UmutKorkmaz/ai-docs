# Chapter 6: Computational Theory

> **Prerequisites**: [Mathematical Foundations](05_Mathematical_Foundations.md)
>
> **Learning Objectives**:
> - Understand fundamental models of computation
> - Learn about computational complexity and its implications for AI
> - Master algorithm design principles for AI systems
> - Explore specialized data structures for AI applications
>
> **Related Topics**: [Cognitive Science Foundations](07_Cognitive_Science_Foundations.md) | [Classic Machine Learning Algorithms](09_Classic_ML_Algorithms.md)

## Turing Machines and Computability

Computational theory provides the foundation for understanding what problems can be solved by machines and how efficiently they can be solved. This is crucial for AI as it helps us understand the fundamental limits of intelligent systems.

### Theoretical Models of Computation

#### Turing Machine

**Definition**: A Turing machine is an abstract computational device consisting of:
- **Tape**: Infinite memory divided into cells
- **Head**: Reads/writes symbols and moves left/right
- **State register**: Finite set of states
- **Transition function**: Rules for state changes

**Formal Definition**: M = (Q, Σ, Γ, δ, q₀, q_accept, q_reject)

Where:
- Q: Finite set of states
- Σ: Input alphabet
- Γ: Tape alphabet (Σ ⊂ Γ)
- δ: Transition function δ: Q × Γ → Q × Γ × {L, R}
- q₀: Initial state
- q_accept, q_reject: Accept/reject states

**Example**: Simple Turing Machine for Binary Addition
```
States: {q0, q1, q2, q_accept}
Tape: Binary numbers separated by '+'
Transition: Add bits, handle carry
```

#### Church-Turing Thesis

**Statement**: Every effectively calculable function is computable by a Turing machine

**Implications**:
- **Universality**: Turing machines can simulate any computational model
- **Limits**: Some problems are fundamentally uncomputable
- **Equivalence**: All reasonable models of computation are equivalent

**Strong Church-Turing Thesis**: Every effectively calculable function can be computed by a Turing machine with polynomial time complexity

#### Universal Turing Machine

**Definition**: A Turing machine that can simulate any other Turing machine
- **Input**: Description of target machine + its input
- **Output**: Same as target machine would produce
- **Significance**: Foundation of modern programmable computers

**Components**:
- **Simulation of states**: Track current state of target machine
- **Tape management**: Handle multiple tapes
- **Transition emulation**: Apply target machine's rules

**Applications**:
- **Computer architecture**: Von Neumann architecture
- **Programming languages**: Universal computation
- **Virtualization**: Machine simulation

### Uncomputable Problems

#### The Halting Problem

**Problem**: Determine whether a given program will halt or run forever
**Turing's Proof (1936)**: By contradiction, shows halting problem is undecidable

**Proof Sketch**:
1. Assume halting problem is decidable
2. Construct a program that halts iff it doesn't halt
3. Contradiction implies assumption is false

**Implications for AI**:
- **Program verification**: Cannot automatically verify all programs
- **AI safety**: Cannot guarantee all AI systems are safe
- **Automated reasoning**: Limits of theorem proving

#### Other Uncomputable Problems

**Hilbert's 10th Problem**: Solving Diophantine equations
**Post's Correspondence Problem**: String matching problem
**Tiling Problem**: Can a given shape tile the plane?
**Kolmogorov Complexity**: Finding minimal program description

### Computational Complexity

#### Complexity Classes

**P (Polynomial Time)**: Problems solvable in polynomial time
- **Definition**: Problems with O(nᵏ) time complexity
- **Examples**: Sorting, shortest path, matrix multiplication
- **Significance**: Considered "tractable" problems

**NP (Non-deterministic Polynomial)**: Problems with solutions verifiable in polynomial time
- **Definition**: Solutions can be checked quickly
- **Examples**: Sudoku, traveling salesman, satisfiability
- **P vs NP Problem**: Is P = NP? (Millennium Prize Problem)

**NP-Complete**: Hardest problems in NP
- **Definition**: All NP problems reduce to these
- **Examples**: SAT, 3-SAT, Graph Coloring
- **Significance**: If any NP-complete problem is in P, then P = NP

**NP-Hard**: At least as hard as NP-complete problems
- **Definition**: May not be in NP
- **Examples**: Halting problem, optimization problems

#### The P vs NP Problem

**Current Status**: Unsolved, widely believed that P ≠ NP
**Implications**:
- **Cryptography**: Security of encryption schemes
- **Optimization**: Many practical problems are NP-hard
- **AI**: Limits of automated reasoning

**NP-Complete Problems in AI**:
- **Boolean Satisfiability (SAT)**: Logic-based reasoning
- **Constraint Satisfaction**: Planning and scheduling
- **Graph Coloring**: Resource allocation
- **Traveling Salesman**: Route optimization

#### Approximation Algorithms

**Definition**: Algorithms that find near-optimal solutions
**Approximation Ratio**: α = (algorithm solution) / (optimal solution)

**Common Techniques**:
- **Greedy algorithms**: Local optimal choices
- **Local search**: Hill climbing, simulated annealing
- **Linear programming relaxation**: Relax integer constraints
- **Randomized algorithms**: Probabilistic methods

**Examples**:
```python
# Vertex Cover (2-approximation)
def vertex_cover_approx(graph):
    cover = set()
    edges = graph.edges.copy()
    while edges:
        u, v = edges.pop()
        cover.add(u)
        cover.add(v)
        edges = [e for e in edges if u not in e and v not in e]
    return cover  # 2 * optimal

# Set Cover (ln n-approximation)
def set_cover_approx(universe, sets):
    cover = set()
    covered = set()
    while covered != universe:
        best_set = max(sets, key=lambda s: len(s - covered))
        cover.add(best_set)
        covered |= best_set
    return cover  # ln(n) * optimal
```

### AI Relevance of Complexity Theory

#### Problem Complexity Analysis

**Understanding Limits**:
- **Fundamental limits**: Some problems are inherently difficult
- **Resource requirements**: Time, space, energy considerations
- **Scalability**: How algorithms perform with large inputs

**Algorithm Selection**:
- **Exact vs approximate**: Trade-offs between optimality and efficiency
- **Heuristic methods**: When exact methods are infeasible
- **Specialized algorithms**: Domain-specific optimizations

#### Practical Implications

**Real-World Constraints**:
- **Time constraints**: Real-time AI applications
- **Memory limitations**: Embedded systems, mobile devices
- **Energy efficiency**: Battery-powered devices

**Design Considerations**:
- **Problem decomposition**: Break large problems into smaller ones
- **Parallel processing**: Distribute computation
- **Approximation methods**: Accept good enough solutions
- **Randomization**: Probabilistic algorithms

## Algorithm Design and Analysis

### Algorithmic Paradigms

#### Divide and Conquer

**Strategy**: Break problem into smaller subproblems, solve recursively, combine solutions

**Key Components**:
1. **Divide**: Split into smaller subproblems
2. **Conquer**: Solve subproblems recursively
3. **Combine**: Merge subproblem solutions

**Examples in AI**:
```python
# Merge Sort
def merge_sort(arr):
    if len(arr) ≤ 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

# Binary Search
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left ≤ right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# Quick Sort (for feature selection)
def quick_sort_features(features, scores):
    if len(features) ≤ 1:
        return features
    pivot = features[0]
    pivot_score = scores[0]
    left = [f for f, s in zip(features[1:], scores[1:]) if s ≤ pivot_score]
    right = [f for f, s in zip(features[1:], scores[1:]) if s > pivot_score]
    return quick_sort_features(left, [s for f, s in zip(features[1:], scores[1:]) if s ≤ pivot_score]) + \
           [pivot] + \
           quick_sort_features(right, [s for f, s in zip(features[1:], scores[1:]) if s > pivot_score])
```

#### Dynamic Programming

**Strategy**: Solve complex problems by breaking into overlapping subproblems, storing solutions to avoid recomputation

**Key Principles**:
- **Optimal substructure**: Optimal solution contains optimal solutions to subproblems
- **Overlapping subproblems**: Same subproblems recur multiple times
- **Memoization**: Store computed solutions

**Examples in AI**:
```python
# Fibonacci Sequence (memoization)
def fib_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n ≤ 2:
        return 1
    memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
    return memo[n]

# Edit Distance (sequence alignment)
def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n+1) for _ in range(m+1)]

    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j

    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    return dp[m][n]

# Longest Common Subsequence
def lcs(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n+1) for _ in range(m+1)]

    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]
```

#### Greedy Algorithms

**Strategy**: Make locally optimal choices at each step, hoping to find global optimum

**When Greedy Works**:
- **Greedy choice property**: Local optimal leads to global optimal
- **Optimal substructure**: Problem has optimal substructure

**Examples in AI**:
```python
# Huffman Coding
def huffman_coding(frequencies):
    heap = [[weight, [symbol, ""]] for symbol, weight in frequencies.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        low = heapq.heappop(heap)
        high = heapq.heappop(heap)
        for pair in low[1:]:
            pair[1] = '0' + pair[1]
        for pair in high[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [low[0] + high[0]] + low[1:] + high[1:])

    return heap[0]

# Dijkstra's Algorithm (shortest path)
def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]

    while pq:
        current_distance, current_node = heapq.heappop(pq)

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))

    return distances

# Activity Selection
def activity_selection(activities):
    activities.sort(key=lambda x: x[1])  # Sort by end time
    selected = [activities[0]]
    last_end = activities[0][1]

    for activity in activities[1:]:
        if activity[0] >= last_end:
            selected.append(activity)
            last_end = activity[1]

    return selected
```

#### Randomized Algorithms

**Strategy**: Use random numbers to make decisions or guide computation

**Types**:
- **Monte Carlo**: Always fast, probably correct
- **Las Vegas**: Always correct, probably fast

**Examples in AI**:
```python
# Randomized Quicksort
def randomized_quicksort(arr):
    if len(arr) ≤ 1:
        return arr
    pivot = random.choice(arr)
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return randomized_quicksort(left) + middle + randomized_quicksort(right)

# Randomized Rounding for LP relaxation
def randomized_rounding(lp_solution):
    rounded = []
    for var in lp_solution:
        if random.random() ≤ var:
            rounded.append(1)
        else:
            rounded.append(0)
    return rounded

# Markov Chain Monte Carlo (MCMC)
def metropolis_hastings(target_dist, proposal_dist, iterations):
    current = random.choice(list(target_dist.keys()))
    samples = [current]

    for _ in range(iterations):
        proposed = proposal_dist(current)
        acceptance_ratio = target_dist[proposed] / target_dist[current]

        if random.random() < acceptance_ratio:
            current = proposed

        samples.append(current)

    return samples
```

### Analysis Techniques

#### Time Complexity

**Big-O Notation**: Upper bound on growth rate
**Big-Ω Notation**: Lower bound on growth rate
**Big-θ Notation**: Tight bound on growth rate

**Common Complexities**:
- **O(1)**: Constant time
- **O(log n)**: Logarithmic time
- **O(n)**: Linear time
- **O(n log n)**: Linearithmic time
- **O(n²)**: Quadratic time
- **O(2ⁿ)**: Exponential time
- **O(n!)**: Factorial time

**Analysis Methods**:
- **Loop analysis**: Count iterations
- **Recurrence relations**: Solve recursive algorithms
- **Amortized analysis**: Average cost over operations
- **Probabilistic analysis**: Expected performance

#### Space Complexity

**Definition**: Memory requirements as function of input size
**Analysis**:
- **Auxiliary space**: Extra space used by algorithm
- **Input space**: Space required for input
- **Total space**: Sum of auxiliary and input space

**Space-Time Trade-offs**:
- **Memoization**: Space for speed
- **Caching**: Store computed results
- **Precomputation**: Store lookup tables

#### Amortized Analysis

**Definition**: Average performance over sequence of operations
**Methods**:
- **Aggregate method**: Total cost divided by operations
- **Accounting method**: Assign credits to operations
- **Potential method**: Define potential function

**Example**: Dynamic Array
```python
class DynamicArray:
    def __init__(self):
        self.capacity = 1
        self.size = 0
        self.array = [None] * self.capacity

    def append(self, item):
        if self.size == self.capacity:
            self.resize(2 * self.capacity)
        self.array[self.size] = item
        self.size += 1

    def resize(self, new_capacity):
        new_array = [None] * new_capacity
        for i in range(self.size):
            new_array[i] = self.array[i]
        self.array = new_array
        self.capacity = new_capacity
```

**Amortized Cost**: O(1) per append operation

#### Competitive Analysis

**Definition**: Compare online algorithm to optimal offline algorithm
**Competitive Ratio**: c = (online cost) / (optimal cost)

**Applications**:
- **Paging algorithms**: Cache management
- **Scheduling**: Job scheduling
- **Online learning**: Adaptive algorithms

## AI-Specific Algorithms

### Search Algorithms

#### Uninformed Search

**Breadth-First Search (BFS)**:
- **Strategy**: Explore level by level
- **Completeness**: Complete (finds solution if exists)
- **Optimality**: Optimal for unweighted graphs
- **Time**: O(bᵈ) where b = branching factor, d = depth
- **Space**: O(bᵈ)

```python
def bfs(graph, start, goal):
    from collections import deque
    queue = deque([(start, [start])])
    visited = set([start])

    while queue:
        node, path = queue.popleft()
        if node == goal:
            return path

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None
```

**Depth-First Search (DFS)**:
- **Strategy**: Explore as deep as possible
- **Completeness**: Complete in finite spaces
- **Optimality**: Not optimal
- **Time**: O(bᵐ) where m = maximum depth
- **Space**: O(bm)

```python
def dfs(graph, start, goal, visited=None):
    if visited is None:
        visited = set()

    if start == goal:
        return [start]

    visited.add(start)

    for neighbor in graph[start]:
        if neighbor not in visited:
            path = dfs(graph, neighbor, goal, visited)
            if path:
                return [start] + path

    return None
```

**Uniform Cost Search (UCS)**:
- **Strategy**: Expand least cost path
- **Completeness**: Complete
- **Optimality**: Optimal
- **Time**: O(b^(C*/ε)) where C* = optimal cost
- **Space**: O(b^(C*/ε))

```python
def uniform_cost_search(graph, start, goal):
    import heapq
    pq = [(0, start, [start])]
    visited = set()

    while pq:
        cost, node, path = heapq.heappop(pq)

        if node == goal:
            return path

        if node in visited:
            continue

        visited.add(node)

        for neighbor, edge_cost in graph[node].items():
            if neighbor not in visited:
                heapq.heappush(pq, (cost + edge_cost, neighbor, path + [neighbor]))

    return None
```

#### Informed Search

**A* Search**:
- **Strategy**: Best-first search with heuristic
- **Evaluation function**: f(n) = g(n) + h(n)
- **Admissible heuristic**: h(n) ≤ h*(n) (never overestimates)
- **Completeness**: Complete with consistent heuristic
- **Optimality**: Optimal with admissible heuristic

```python
def a_star_search(graph, start, goal, heuristic):
    import heapq
    pq = [(0 + heuristic[start], start, [start], 0)]
    visited = {}

    while pq:
        f_score, node, path, g_score = heapq.heappop(pq)

        if node == goal:
            return path

        if node in visited and visited[node] ≤ g_score:
            continue

        visited[node] = g_score

        for neighbor, edge_cost in graph[node].items():
            new_g = g_score + edge_cost
            new_f = new_g + heuristic[neighbor]
            heapq.heappush(pq, (new_f, neighbor, path + [neighbor], new_g))

    return None
```

**Iterative Deepening A* (IDA*)**:
- **Strategy**: Iterative deepening with heuristic cutoff
- **Advantage**: Memory efficient
- **Complete**: Complete
- **Optimal**: Optimal with admissible heuristic

```python
def ida_star_search(graph, start, goal, heuristic):
    def search(path, g, bound):
        node = path[-1]
        f = g + heuristic[node]

        if f > bound:
            return f
        if node == goal:
            return "FOUND"

        min_cost = float('inf')
        for neighbor, edge_cost in graph[node].items():
            if neighbor not in path:
                new_g = g + edge_cost
                t = search(path + [neighbor], new_g, bound)
                if t == "FOUND":
                    return "FOUND"
                if t < min_cost:
                    min_cost = t

        return min_cost

    bound = heuristic[start]
    path = [start]

    while True:
        t = search(path, 0, bound)
        if t == "FOUND":
            return path
        if t == float('inf'):
            return None
        bound = t
```

### Planning Algorithms

#### STRIPS Planning

**Components**:
- **States**: Set of positive literals
- **Actions**: Preconditions, add list, delete list
- **Goals**: Set of positive literals

**Planning as Search**:
- **State space planning**: Search in state space
- **Plan space planning**: Search in plan space
- **Graphplan**: Planning graph construction

**Forward State Space Planning**:
```python
def strips_planner(actions, initial_state, goal):
    from collections import deque

    def applicable_actions(state):
        return [action for action in actions
                if all(pre in state for pre in action['preconditions'])]

    def apply_action(state, action):
        new_state = state.copy()
        for literal in action['delete']:
            if literal in new_state:
                new_state.remove(literal)
        for literal in action['add']:
            new_state.add(literal)
        return new_state

    queue = deque([(initial_state, [])])
    visited = set()

    while queue:
        state, plan = queue.popleft()
        state_tuple = tuple(sorted(state))

        if state_tuple in visited:
            continue

        visited.add(state_tuple)

        if goal.issubset(state):
            return plan

        for action in applicable_actions(state):
            new_state = apply_action(state, action)
            new_plan = plan + [action['name']]
            queue.append((new_state, new_plan))

    return None
```

#### PDDL Planning

**PDDL (Planning Domain Definition Language)**:
- **Domain**: Defines actions and types
- **Problem**: Defines initial state and goal
- **Extended features**: Types, conditional effects, quantifiers

**Example PDDL Domain**:
```
(define (domain logistics)
  (:requirements :strips :typing)

  (:types location truck package)

  (:predicates
    (at ?x - ?y)
    (in ?p - package ?t - truck)
    (connected ?from ?to - location)
  )

  (:action drive-truck
    :parameters (?t - truck ?from ?to - location)
    :precondition (and (at ?t ?from) (connected ?from ?to))
    :effect (and (not (at ?t ?from)) (at ?t ?to))
  )

  (:action load-package
    :parameters (?p - package ?t - truck ?l - location)
    :precondition (and (at ?p ?l) (at ?t ?l))
    :effect (and (not (at ?p ?l)) (in ?p ?t))
  )
)
```

### Learning Algorithms

#### Supervised Learning Algorithms

**Linear Regression**:
```python
def linear_regression(X, y):
    # Normal equation: θ = (XᵀX)⁻¹Xᵀy
    X_with_bias = np.column_stack([np.ones(len(X)), X])
    theta = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
    return theta
```

**Logistic Regression**:
```python
def logistic_regression(X, y, learning_rate=0.01, iterations=1000):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0

    for _ in range(iterations):
        # Forward pass
        linear_model = np.dot(X, weights) + bias
        predictions = 1 / (1 + np.exp(-linear_model))

        # Backward pass
        dw = (1/n_samples) * np.dot(X.T, (predictions - y))
        db = (1/n_samples) * np.sum(predictions - y)

        # Update
        weights -= learning_rate * dw
        bias -= learning_rate * db

    return weights, bias
```

**Decision Trees**:
```python
class DecisionNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def build_decision_tree(X, y, depth=0, max_depth=10):
    if depth >= max_depth or len(set(y)) == 1:
        return DecisionNode(value=np.bincount(y).argmax())

    n_samples, n_features = X.shape

    # Find best split
    best_gain = 0
    best_feature = None
    best_threshold = None

    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            left_mask = X[:, feature] <= threshold
            right_mask = ~left_mask

            if len(left_mask) == 0 or len(right_mask) == 0:
                continue

            gain = information_gain(y, y[left_mask], y[right_mask])
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold

    if best_gain == 0:
        return DecisionNode(value=np.bincount(y).argmax())

    left_tree = build_decision_tree(X[X[:, best_feature] <= best_threshold],
                                   y[X[:, best_feature] <= best_threshold],
                                   depth + 1, max_depth)
    right_tree = build_decision_tree(X[X[:, best_feature] > best_threshold],
                                    y[X[:, best_feature] > best_threshold],
                                    depth + 1, max_depth)

    return DecisionNode(feature=best_feature, threshold=best_threshold,
                       left=left_tree, right=right_tree)
```

#### Unsupervised Learning Algorithms

**K-Means Clustering**:
```python
def k_means(X, k, max_iterations=100):
    n_samples, n_features = X.shape

    # Initialize centroids
    centroids = X[np.random.choice(n_samples, k, replace=False)]

    for _ in range(max_iterations):
        # Assign clusters
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)

        # Update centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return centroids, labels
```

**Hierarchical Clustering**:
```python
def hierarchical_clustering(X, n_clusters):
    n_samples = X.shape[0]
    clusters = [[i] for i in range(n_samples)]

    while len(clusters) > n_clusters:
        # Find closest clusters
        min_distance = float('inf')
        closest_pair = (0, 1)

        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                distance = linkage_distance(X[clusters[i]], X[clusters[j]])
                if distance < min_distance:
                    min_distance = distance
                    closest_pair = (i, j)

        # Merge closest clusters
        i, j = closest_pair
        clusters[i].extend(clusters[j])
        clusters.pop(j)

    return clusters

def linkage_distance(cluster1, cluster2, method='single'):
    if method == 'single':
        return min([np.linalg.norm(x1 - x2) for x1 in cluster1 for x2 in cluster2])
    elif method == 'complete':
        return max([np.linalg.norm(x1 - x2) for x1 in cluster1 for x2 in cluster2])
    elif method == 'average':
        return np.mean([np.linalg.norm(x1 - x2) for x1 in cluster1 for x2 in cluster2])
```

### Optimization Algorithms

#### Gradient Descent
```python
def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0

    for _ in range(iterations):
        # Predictions
        y_pred = np.dot(X, weights) + bias

        # Gradients
        dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
        db = (1/n_samples) * np.sum(y_pred - y)

        # Update
        weights -= learning_rate * dw
        bias -= learning_rate * db

    return weights, bias
```

#### Stochastic Gradient Descent
```python
def stochastic_gradient_descent(X, y, learning_rate=0.01, epochs=100):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0

    for epoch in range(epochs):
        for i in range(n_samples):
            # Single sample prediction
            y_pred = np.dot(X[i], weights) + bias

            # Gradients
            dw = (y_pred - y[i]) * X[i]
            db = y_pred - y[i]

            # Update
            weights -= learning_rate * dw
            bias -= learning_rate * db

    return weights, bias
```

#### Adam Optimizer
```python
def adam_optimizer(X, y, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, iterations=1000):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0

    m_w = np.zeros(n_features)  # First moment
    v_w = np.zeros(n_features)  # Second moment
    m_b = 0  # First moment
    v_b = 0  # Second moment

    for t in range(1, iterations + 1):
        # Predictions
        y_pred = np.dot(X, weights) + bias

        # Gradients
        dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
        db = (1/n_samples) * np.sum(y_pred - y)

        # Update biased first moment estimate
        m_w = beta1 * m_w + (1 - beta1) * dw
        m_b = beta1 * m_b + (1 - beta1) * db

        # Update biased second raw moment estimate
        v_w = beta2 * v_w + (1 - beta2) * (dw ** 2)
        v_b = beta2 * v_b + (1 - beta2) * (db ** 2)

        # Compute bias-corrected first moment estimate
        m_w_hat = m_w / (1 - beta1 ** t)
        m_b_hat = m_b / (1 - beta1 ** t)

        # Compute bias-corrected second raw moment estimate
        v_w_hat = v_w / (1 - beta2 ** t)
        v_b_hat = v_b / (1 - beta2 ** t)

        # Update parameters
        weights -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
        bias -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

    return weights, bias
```

## Data Structures for AI

### Fundamental Data Structures

#### Arrays and Lists

**Arrays**: Fixed-size, contiguous memory
- **Advantages**: O(1) access, cache-friendly
- **Disadvantages**: Fixed size, expensive insertion/deletion
- **AI Applications**: Feature vectors, weight matrices

**Linked Lists**: Dynamic, node-based
- **Advantages**: Dynamic size, efficient insertion/deletion
- **Disadvantages**: O(n) access, extra memory overhead
- **AI Applications**: Dynamic data structures, linked neural networks

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, val):
        new_node = ListNode(val)
        if not self.head:
            self.head = new_node
            return

        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
```

#### Trees and Graphs

**Binary Trees**: Hierarchical structure with at most two children
- **Types**: Binary search trees, AVL trees, Red-Black trees
- **Applications**: Decision trees, expression trees, hierarchical clustering

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class BinarySearchTree:
    def __init__(self):
        self.root = None

    def insert(self, val):
        if not self.root:
            self.root = TreeNode(val)
            return

        self._insert_recursive(self.root, val)

    def _insert_recursive(self, node, val):
        if val < node.val:
            if not node.left:
                node.left = TreeNode(val)
            else:
                self._insert_recursive(node.left, val)
        elif val > node.val:
            if not node.right:
                node.right = TreeNode(val)
            else:
                self._insert_recursive(node.right, val)
```

**Graphs**: Nodes connected by edges
- **Types**: Directed, undirected, weighted, unweighted
- **Representations**: Adjacency matrix, adjacency list
- **Applications**: Neural networks, knowledge graphs, social networks

```python
class Graph:
    def __init__(self, directed=False):
        self.adjacency_list = {}
        self.directed = directed

    def add_edge(self, u, v, weight=1):
        if u not in self.adjacency_list:
            self.adjacency_list[u] = []
        if v not in self.adjacency_list:
            self.adjacency_list[v] = []

        self.adjacency_list[u].append((v, weight))
        if not self.directed:
            self.adjacency_list[v].append((u, weight))
```

#### Hash Tables

**Hash Tables**: Key-value pairs with constant-time access
- **Operations**: Insert, delete, search in O(1) average time
- **Collision resolution**: Chaining, open addressing
- **Applications**: Feature hashing, cache systems, symbol tables

```python
class HashTable:
    def __init__(self, size=10):
        self.size = size
        self.table = [[] for _ in range(size)]

    def hash_function(self, key):
        return hash(key) % self.size

    def insert(self, key, value):
        index = self.hash_function(key)
        for i, (existing_key, existing_value) in enumerate(self.table[index]):
            if existing_key == key:
                self.table[index][i] = (key, value)
                return
        self.table[index].append((key, value))

    def get(self, key):
        index = self.hash_function(key)
        for existing_key, value in self.table[index]:
            if existing_key == key:
                return value
        raise KeyError(key)
```

#### Priority Queues

**Priority Queues**: Elements with associated priorities
- **Operations**: Insert, extract-min/max in O(log n)
- **Implementations**: Binary heap, Fibonacci heap
- **Applications**: Dijkstra's algorithm, A* search, event scheduling

```python
import heapq

class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        heapq.heappush(self.heap, (priority, self.count, item))
        self.count += 1

    def pop(self):
        if not self.heap:
            raise IndexError("Priority queue is empty")
        return heapq.heappop(self.heap)[2]

    def peek(self):
        if not self.heap:
            raise IndexError("Priority queue is empty")
        return self.heap[0][2]

    def is_empty(self):
        return len(self.heap) == 0
```

### Specialized AI Data Structures

#### Knowledge Graphs

**Knowledge Graphs**: Semantic networks representing relationships
- **Components**: Entities, relations, attributes
- **Querying**: SPARQL, graph traversal
- **Applications**: Semantic search, question answering, recommendation systems

```python
class KnowledgeGraph:
    def __init__(self):
        self.entities = {}
        self.relations = {}

    def add_entity(self, entity_id, attributes=None):
        if attributes is None:
            attributes = {}
        self.entities[entity_id] = attributes

    def add_relation(self, subject, relation, object):
        if subject not in self.relations:
            self.relations[subject] = {}
        if relation not in self.relations[subject]:
            self.relations[subject][relation] = []
        self.relations[subject][relation].append(object)

    def query(self, subject, relation=None, object=None):
        results = []

        if subject is not None:
            if relation is not None:
                if object is not None:
                    # Specific triple
                    if (subject in self.relations and
                        relation in self.relations[subject] and
                        object in self.relations[subject][relation]):
                        results.append((subject, relation, object))
                else:
                    # Subject and relation
                    if subject in self.relations and relation in self.relations[subject]:
                        for obj in self.relations[subject][relation]:
                            results.append((subject, relation, obj))
            else:
                # Only subject
                if subject in self.relations:
                    for rel, objects in self.relations[subject].items():
                        for obj in objects:
                            results.append((subject, rel, obj))

        return results
```

#### Trie Structures

**Tries**: Prefix trees for string processing
- **Advantages**: Efficient prefix search, autocomplete
- **Applications**: Text processing, spell checking, IP routing

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.frequency = 0

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
        node.frequency += 1

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end

    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

    def autocomplete(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        suggestions = []
        self._collect_words(node, prefix, suggestions)
        return sorted(suggestions, key=lambda x: x[1], reverse=True)

    def _collect_words(self, node, prefix, suggestions):
        if node.is_end:
            suggestions.append((prefix, node.frequency))

        for char, child in node.children.items():
            self._collect_words(child, prefix + char, suggestions)
```

#### Bloom Filters

**Bloom Filters**: Probabilistic data structure for membership testing
- **Advantages**: Space-efficient, fast lookups
- **Disadvantages**: False positives possible, no deletions
- **Applications**: Cache filtering, spam detection, network routing

```python
import mmh3
from bitarray import bitarray

class BloomFilter:
    def __init__(self, size, hash_count):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = bitarray(size)
        self.bit_array.setall(0)

    def add(self, item):
        for i in range(self.hash_count):
            index = mmh3.hash(str(item), i) % self.size
            self.bit_array[index] = 1

    def __contains__(self, item):
        for i in range(self.hash_count):
            index = mmh3.hash(str(item), i) % self.size
            if not self.bit_array[index]:
                return False
        return True

    def false_positive_rate(self):
        # Approximate false positive rate
        n = self.bit_array.count() / self.size
        return (1 - (1 - 1/self.size) ** (self.hash_count * n)) ** self.hash_count
```

#### Persistent Data Structures

**Persistent Data Structures**: Previous versions preserved
- **Types**: Persistent lists, trees, arrays
- **Advantages**: Immutable, thread-safe, efficient versioning
- **Applications**: Functional programming, version control, undo/redo

```python
class PersistentListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class PersistentList:
    def __init__(self):
        self.head = None
        self.versions = []

    def append(self, val):
        if not self.head:
            new_head = PersistentListNode(val)
            self.head = new_head
            self.versions.append(new_head)
            return

        # Copy existing nodes
        new_head = PersistentListNode(self.head.val)
        current = new_head
        original_current = self.head.next

        while original_current:
            current.next = PersistentListNode(original_current.val)
            current = current.next
            original_current = original_current.next

        # Add new node at end
        current.next = PersistentListNode(val)
        self.head = new_head
        self.versions.append(new_head)

    def get_version(self, version_index):
        if 0 <= version_index < len(self.versions):
            return self.versions[version_index]
        return None
```

### Efficiency Considerations

#### Memory Hierarchy

**Cache Optimization**:
- **Spatial locality**: Access nearby memory
- **Temporal locality**: Reuse recently accessed data
- **Cache-aware algorithms**: Consider cache size and line size
- **Cache-oblivious algorithms**: Work well regardless of cache size

**Memory Access Patterns**:
- **Sequential access**: Cache-friendly
- **Random access**: Cache-unfriendly
- **Strided access**: Moderate efficiency
- **Blocked access**: Good for large datasets

#### Parallel Data Structures

**Concurrent Data Structures**:
- **Thread-safe**: Multiple threads can access safely
- **Lock-free**: Avoid locking for better performance
- **Wait-free**: Guaranteed progress in bounded steps

**Examples**:
- **Concurrent hash tables**: Sharding, fine-grained locking
- **Concurrent queues**: Lock-free implementations
- **Concurrent trees**: Red-black trees with fine-grained locks

```python
import threading

class ConcurrentHashTable:
    def __init__(self, size=16):
        self.size = size
        self.buckets = [[] for _ in range(size)]
        self.locks = [threading.Lock() for _ in range(size)]

    def hash_function(self, key):
        return hash(key) % self.size

    def insert(self, key, value):
        index = self.hash_function(key)
        with self.locks[index]:
            bucket = self.buckets[index]
            for i, (k, v) in enumerate(bucket):
                if k == key:
                    bucket[i] = (key, value)
                    return
            bucket.append((key, value))

    def get(self, key):
        index = self.hash_function(key)
        with self.locks[index]:
            bucket = self.buckets[index]
            for k, v in bucket:
                if k == key:
                    return v
            raise KeyError(key)
```

#### Streaming Data Structures

**Streaming Algorithms**: Process data with limited memory
- **Approximate answers**: Trade accuracy for efficiency
- **Single pass**: Process each item once
- **Applications**: Network monitoring, sensor data, financial data

**Examples**:
- **Count-min sketch**: Frequency estimation
- **HyperLogLog**: Cardinality estimation
- **Reservoir sampling**: Random sampling from streams

```python
import random
import math

class CountMinSketch:
    def __init__(self, width, depth):
        self.width = width
        self.depth = depth
        self.counts = [[0] * width for _ in range(depth)]
        self.hash_seeds = [random.randint(0, 1000000) for _ in range(depth)]

    def hash_function(self, item, seed):
        return (hash(str(item)) + seed) % self.width

    def add(self, item, count=1):
        for i in range(self.depth):
            index = self.hash_function(item, self.hash_seeds[i])
            self.counts[i][index] += count

    def estimate(self, item):
        min_count = float('inf')
        for i in range(self.depth):
            index = self.hash_function(item, self.hash_seeds[i])
            min_count = min(min_count, self.counts[i][index])
        return min_count

class HyperLogLog:
    def __init__(self, b=10):
        self.b = b
        self.m = 1 << b  # 2^b registers
        self.registers = [0] * self.m
        self.alpha = self._get_alpha()

    def _get_alpha(self):
        if self.m == 16:
            return 0.673
        elif self.m == 32:
            return 0.697
        elif self.m == 64:
            return 0.709
        else:
            return 0.7213 / (1 + 1.079 / self.m)

    def add(self, item):
        x = hash(str(item))
        # Get first b bits for register index
        index = x & (self.m - 1)
        # Get remaining bits for leading zeros
        w = x >> self.b
        self.registers[index] = max(self.registers[index], self._rho(w))

    def _rho(self, w):
        return len(bin(w)) - 2 if w > 0 else 32

    def estimate(self):
        sum_register_inv = sum([2 ** (-r) for r in self.registers])
        estimate = self.alpha * self.m * self.m / sum_register_inv

        # Small range correction
        if estimate <= 2.5 * self.m:
            zeros = self.registers.count(0)
            if zeros != 0:
                estimate = self.m * math.log(self.m / zeros)

        return estimate
```

#### Distributed Data Structures

**Distributed Systems**: Data across multiple machines
- **Partitioning**: Split data across nodes
- **Replication**: Copy data for reliability
- **Consistency**: Ensure data consistency across nodes

**Examples**:
- **Distributed hash tables**: Consistent hashing
- **Distributed queues**: Message queues
- **Distributed caches**: Memcached, Redis

```python
import hashlib

class ConsistentHashRing:
    def __init__(self, virtual_nodes=3):
        self.virtual_nodes = virtual_nodes
        self.ring = {}
        self.nodes = set()

    def _hash(self, key):
        return int(hashlib.md5(str(key).encode()).hexdigest(), 16)

    def add_node(self, node):
        self.nodes.add(node)
        for i in range(self.virtual_nodes):
            virtual_node = f"{node}:{i}"
            hash_value = self._hash(virtual_node)
            self.ring[hash_value] = node

    def remove_node(self, node):
        self.nodes.remove(node)
        for i in range(self.virtual_nodes):
            virtual_node = f"{node}:{i}"
            hash_value = self._hash(virtual_node)
            del self.ring[hash_value]

    def get_node(self, key):
        if not self.ring:
            return None

        hash_value = self._hash(key)
        # Find first node with hash ≥ key hash
        keys = sorted(self.ring.keys())
        for key_hash in keys:
            if key_hash >= hash_value:
                return self.ring[key_hash]

        # Wrap around
        return self.ring[keys[0]]
```

---

**Next Chapter**: [Cognitive Science Foundations](07_Cognitive_Science_Foundations.md) - Understanding human cognition and its implications for AI

**Related Topics**: [Introduction to Machine Learning](08_Introduction_to_ML.md) | [Classic Machine Learning Algorithms](09_Classic_ML_Algorithms.md)

**Algorithm Reference**: See [Appendix A](A_Mathematical_Reference.md) for mathematical foundations