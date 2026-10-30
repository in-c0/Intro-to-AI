---
chapter: 2
description: States and actions, search spaces, BFS / DFS / iterative deepening, informed search.
---

# 2. Search

You will learn:
- the theory behind search in AI
- to develop a smart agent with search capability to solve interesting problem



### 2.1 State, Action, Transition Function

A problem can be described by
- State
- Action
- Transition Function

State: variable, e.g. position on the grid, (x,y)
Action: function, e.g. "move up" "move down"
Transition Function: The function that determines how the action changes the state. T(s,a) = s'
(where s is state, a is action, s' is the new state) This is deterministic transition function, where next state is predictable.
If the next state is probabilistic, use Stochastic transition function:
 T(s,a,s') = P(s'|s,a)
(where P(s'|s,a) is the probability of transitioning to new state s', given state and action)

Types of states:
- Initial state
- Goal state
- Terminal state
- Intermediate state
- Deadend state

### 2.2 Search Space

Search space
= set of all possible states a system can be in

e.g. Chess
**State**: The configuration of the chessboard at any given time, including the positions of all pieces.
**State Space**: All the possible ways the chessboard can be arranged, which is finite but very large (approximately 10^43 possible positions)

State space could be infinite, e.g. robot navigation:
**State**: position (x, y) and direction the robot could occupy,
**State Space**:  finite (if the space is discretized) or infinite (if the space is continuous)


(Assuming Graph Theory knowledge)
State space can be represented as a graph

Node = State
Edge = Action

(0,0) --move right--> (1,0) --move right--> (2,0)
   |                      |                       |
 move down            move down          move down
   |                      |                       |
(0,1) --move right--> (1,1) --move right--> (2,1)
   |                      |                       |
 move down            move down          move down
   |                      |                       |
(0,2) --move right--> (1,2) --move right--> (2,2)

If the environment is deterministic, each action leads to one specific new state, corresponding to one edge between two nodes. 
In a stochastic environment, edges could represent probabilistic transitions.

move up (0.7)
   (0,0)
     | 
   (0,1) ------> (1,1) move right (0.15)
     |
   (0,2) 
move down (0.15)

70% chance the robot moves up 
15% chance the robot moves right
15% chance the robot moves down

In some cases, the graph edges are weighted, meaning each edge has a cost or value associated with the transition between two states. 
The weights can represent things like the cost of an action (time, energy, or distance)
The goal is often to find the path from the initial state to the goal state that minimizes this cost. (Shortest path search, e.g. Dijkstra's algorithm or A* search) Covered more later

### 2.3 Uninformed Search vs Informed Search

Uninformed = No info about which path to explore next. No clue how to get to the goal state faster
Example:
Breadth-First Search (BFS): Explores all possible nodes at the present depth level before moving deeper into the search tree.
Depth-First Search (DFS): Explores as far down a path as possible before backtracking to explore other paths.


Uninformed means "No heuristic knowledge"
A "heuristic" is a function that estimates how close a given state is to the goal, helping the algorithm prioritize which paths to explore.
More on that later.

Informed search uses domain-specific knowledge and heuristics to determine which options to explore next often leading to faster solutions.
Example:
A* search, Greedy Best-First Search



### 2.4 Uninformed Search

Tree is often used to represent Uninformed Search algorithms, since we dont have to worry about revisiting nodes (since there are no cycles).


![image](https://github.com/user-attachments/assets/519e3fc9-0139-4211-8967-2d9ebf454f19)

A node is said to be Expanded* when the algorithm has examined all their neighboring nodes and added them to the search space.

**4 Types of Nodes in Uninformed Search Tree:**

1. Root Node = Where the search begins. contains the initial state of the algorithm.

2. Expanded* Nodes (Black)

4. Generated Nodes (Black and Red) = Nodes that the algorithm has already visited (i.e. the algorithm has either already expanded* or is about to expand the node.) 

5. Frontier (Red): Nodes that have been generated but not yet expanded*. In other words, the algorithm will explore these next by expanding them and looking at their neighbors.



#### 2.4.1 Breadth-First Search (BFS)

BFS explores all nodes at each depth level before going deeper

BFS treats the frontier as a queue

It selects the first element in the queue to explore next
If the list of paths on the frontier is [p1, p2, ..., pr]
p1 is selected. Its children are added to the end of the queue, after pr. Then p2 is selected next.

All nodes are expanded at a same depth in the tree before any nodes at the next level are expanded.
Can be implemented by using a queue to store frontier nodes.

**Example of BFS in AI:**

![35765050-f46564cc-08ff-11e8-98b2-e093ba83a66e](https://github.com/user-attachments/assets/9b344547-6550-47b9-b3ce-0ec093ae1d1e)
source: https://gist.github.com/kira924age/c14ec7424a966d1e48a9b601289907f0

![Breadth-First-Tree-Traversal](https://github.com/user-attachments/assets/31603d83-42a7-40cd-af63-d83b23e5936d)
source: https://www.codecademy.com/article/tree-traversal

![image](https://github.com/user-attachments/assets/0101c080-e049-4d8f-9e76-3f2bc82c2558)

**Pros:**
- Complete (guaranteed to find a path to the solution, even if the graph is cyclic)
- Optimal (guaranteed to find the shortest path (if the graph is unweighted or has uniform edge weight))

**Cons:**
- Exponential memory usage (has O(b^d) space complexity where b is the 'branching factor' (= number of child nodes for each node) and d is the depth of the search tree)


#### 2.4.2 Depth-First Search (DFS)


![35765045-e1ef0078-08ff-11e8-91af-30ce1cc22767](https://github.com/user-attachments/assets/92bb8acd-599a-43bb-842f-292c4326a6c9)
source: https://gist.github.com/kira924age/c14ec7424a966d1e48a9b601289907f0



iterative psudocode
![image](https://github.com/user-attachments/assets/abc56fa8-20b7-4d8c-8d25-8e400f412a69)


recursive approach can sometimes be more efficient. depends on the tree structure.
![image](https://github.com/user-attachments/assets/e07fb86a-6521-4859-a71f-80e0d4daa3c5)

- Uses stack (Call stack if recursive)

 
**Pros**
- Low memory usage = more space efficient, since it only needs to store the nodes in the current path (has O(d), where d is the depth of the tree)

**Cons**
- Not optimal (Not guaranteed to find the shortest path to the goal, as it may go down very deep paths and miss shorter paths.)
- Not complete (Not guaranteed to find a path to the goal, as it may get stuck in an infinite loop in cyclic graphs)


#### 2.4.3 Depth-Limited Search & Iterative Deepening

Same as DFS, except it doesnt search beyond nodes at a set depth limit. Nodes at this depth limit are treated as if they had no successors.

IDS (Iterative deepening Depth-first Search, aka IDDFS) takes this one step further; it repeats DLS with increasing depth limits until the goal is found.  If it starts with a depth limit of 1, it is essentially the same as BFS, except that it incrementally increases the limit in subsequent iterations. essentailly BFS + DFS

Good explaination: https://ai-master.gitbooks.io/classic-search/content/what-is-depth-limited-search.html


**Pros:**
- Memory-efficient like DFS, O(d) space complexity
- Finds the shortest path like BFS. Optimal and complete
- Avoids the disadvantages of DFS (like getting stuck in cycles or deep branches)

**Cons**:
- Not time-efficient. It repeatedly searches the same nodes at shallow depths, leading to redundant work
- Can be inefficient if branching factor is large

In general, IDS is the preferred search strategy for a large search space with solution that has unknown depth. 


**Bidirectional Search**

 runs two simultaneous searches: one forward from the starting node and one backward from the goal node. The searches meet in the middle, reducing the search space significantly. hence both time and space complexity to O(b^(d/2))

 Requires both the start and goal states to be known.


**Uniform Cost Search**

Similar to BFS, but orders nodes by cost -> useful for weighted graph

 Uses a priority queue to expand the lowest-cost node first

  

### 2.5 Informed Search

Heuristics
Greedy Search
A* search

Finding Heuristic function
Dominance



### 2.6 Solving Problems Using Search

Exercise -
1. Define the states, actions and transition function
2. Convert above definition into a machine readable format
3. Choose a search strategy, considering the context & problem
   
 (give ~5 problems .. what did you choose? why?)
