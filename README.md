# Intro-to-AI

Welcome to the **Intro-to-AI** repository! This repository is crafted for self-learners who want to develop a solid foundation in Artificial Intelligence (AI). It covers a wide array of essential AI topics including neural networks, reinforcement learning, computer vision, and natural language processing.

The content is inspired by my personal notes and learnings from the course COMP9814: Extended Artificial Intelligence at UNSW (24T3). It assumes readers have a basic understanding of Python and mathematics at a sophomore university level, as well as a solid grasp of fundamental data structures and algorithms.

⚠️ Site is currently under active development, frequent changes are expected

## Resources

For a deeper understanding of the topics covered, the following textbooks are recommended:

* Poole, D.L. & Mackworth, A. [Artificial Intelligence: Foundations of Computational Agents](https://artint.info/3e/html/ArtInt3e.html). Second Edition. Cambridge University Press, Cambridge, 2017.
* Russell, S.J. & Norvig, P. **Artificial Intelligence: A Modern Approach**. Fourth Edition, Pearson Education, Hoboken, NJ, 2021.
* Sutton, R. & Barto, A. [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html). MIT press, 2018.
* Jurafsky, D. & Martin, J. H. [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/). Stanford, 2023.

## Content Structure

### 1. [Introduction](#1-introduction)
- 1.1 History of AI  
- 1.2 Agents  
- 1.3 Knowledge Representation  
  - 1.3.1 Feature-based vs Iconic Representations  
  - 1.3.2 Logic  
  - 1.3.3 Learning Rules  

### 2. [Search](#2-search)
- State, Action, Transition Function
- 2.2 Search Space
- 2.3 Uninformed Search vs Informed Search
- 2.4 Uninformed Search  
- 2.5 Informed Search  

### 3. Neural Networks
- 3.1 Neurons - Biological and Artificial  
- 3.2 Single-layer Perceptron  
- 3.3 Linear Separability  
- 3.4 Multi-layer Networks  
- 3.5 Backpropagation  
- 3.6 Neural Engineering Methodology  

### 4. Rewards Instead of Goals
- 4.1 Elements of Reinforcement Learning  
- 4.2 Exploration vs Exploitation  
- 4.3 The Agent-Environment Interface  
- 4.4 Value Functions  
- 4.5 Temporal-Difference Prediction  

### 5. Metaheuristics
- 5.1 Asymptotic Complexity  
- 5.2 Classes of Problems  
- 5.3 Linear Programming  
- 5.4 Search Space  
- 5.5 Metaheuristics with and without Memory  
- 5.6 Population-Based Methods  

### 7. Computer Vision
- 7.1 Image Processing  
- 7.2 Scene Analysis  
- 7.3 Cognitive Vision  

### 8. Language Processing
- 8.1 Formal Languages  
  - 8.1.1 Chomsky’s Hierarchy  
  - 8.1.2 Grammars  
- 8.2 Regular Expressions  
- 8.3 Minimum Edit Distance and Words  
- 8.4 Natural Languages: N-Gram Models  

### 9. Reasoning with Uncertain Information
- 9.1 Confidence Factors  
- 9.2 Probability and Probabilistic Inference  
- 9.3 Bayes Nets  
- 9.4 Fuzzy Logic  

### 10. Human-Aligned Intelligent Robotics
- 10.1 Human Interaction and Human-in-the-Loop Robot Learning  
- 10.2 Explainability and Interpretability  
- 10.3 Safe Robot Exploration  
- 10.4 Ethics in AI  

## Contribution

Your feedback and contributions are greatly appreciated! If you'd like to improve this material, correct errors, or add new sections, feel free to fork the repository and submit a pull request.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. If you find this project helpful, credits and appreciation are always welcome!



---



## 1. [Introduction](#1-introduction)
Artificial intelligence is, well, intelligence that is **artificial**. Artificial just means it is machine-based. But what does *Integllience* mean? We throw the word around, but defining it is surprisingly tricky.

Is intelligence just being really good at math? Or is it creativity? Is it the ability to learn, adapt, or make decisions?Humans are intelligent because we can navigate a complex world, learn from experience, solve problems, create art, etc. We've seen machines could do this too at least to some extent. (Some may argue that it's merely "copying" the original works of humans, but let's put that discussion aside for now). 
Take for instance, a chess AI, which is a "narrow" AI — it’s designed to do one thing, but do it really well. But is it truly intelligent? That machine doesn’t actually “understand” what it’s doing—at least not in the way humans do—but it’s performing tasks we’d consider intelligent.

A good umbrella definition of the term Intellgence might be:
 "the ability to perceive or infer information, and to retain it as knowledge to be applied towards adaptive behaviours within an environment or context"
... or to put it simply,
 "the ability to take in information, process it, and then make decisions based on it."

In fact, AI isn't just a product of modern times. The history of AI goes back thousands of years—long before computers or even electricity.

### 1.1 History of AI
 Let's rewind the clock—way back to 350 BC. Here we find Aristotle, not just a philosopher, but arguably one of the first minds to engage in what we would today call artificial intelligence. He pioneered logic—deductive reasoning, a way of drawing conclusions from facts, which has become the backbone of problem-solving and decision-making in AI today.

> I think, therefore I am
> – René Descartes

Aristotle's system laid the groundwork for what would later become **Boolean logic**, thanks to George Boole in 1848, and Frege’s work on **Formal logic** in 1879.

[Boolean Logic](https://en.wikipedia.org/wiki/Boolean_algebra) is the foundation for modern digital computers, where circuits and operations work using binary (1s and 0s) to perform calculations and make decisions. Boole's work provided the basis for digital logic circuits, which today underpin everything from computer processors to simple calculators.

![image](https://github.com/user-attachments/assets/503c4304-e28c-4cc1-b563-1f46970d1bb2)[2]

* AND: Both conditions must be true (e.g., A AND B is true only if both A and B are true).
* OR: At least one condition must be true (e.g., A OR B is true if either A or B is true).
* NOT: A negation of a condition (e.g., NOT A is true if A is false).

Frege's formal system took a huge step beyond Boolean logic by introducing the concept of quantifiers and variables, which allowed more complex relationships to be expressed in a logical framework. While Boolean logic deals with simple true/false statements, Formal logic translates natural language arguments into a formal language, like first-order logic, to assess whether they are valid using quantifiers such as:

* Universal quantifier (∀): Indicates that a property holds for all elements in a domain (e.g., ∀x: P(x) means "for all x, P(x) is true").
* Existential quantifier (∃): Indicates that there is at least one element in a domain for which a property holds (e.g., ∃x: P(x) means "there exists an x for which P(x) is true").

![image](https://github.com/user-attachments/assets/b8ff5830-17db-47c9-9c5b-c3833b40f8eb)[3]

Examples of formal logic (proportional calculus):

![image](https://github.com/user-attachments/assets/ccc15b01-28d7-4d5e-8691-4f8b605cd206)
![image](https://github.com/user-attachments/assets/875f0f60-55a2-4a58-87f9-ce941fc007b0) [4]


During the 17th–19th century, the invention of formal logic and computational machines took significant steps: 
 René Descartes (1596–1650) and Thomas Hobbes (1588–1679) began exploring the idea of [mind as a machine](https://www.meaningcrisis.co/ep-22-awakening-from-the-meaning-crisis-descartes-vs-hobbes/).
 Charles Babbage (1791–1871) and Ada Lovelace (1815–1852) conceptualized the [Analytical Engine](https://en.wikipedia.org/wiki/Analytical_engine), which is considered the first mechanical general-purpose computer.
 Alan Turing (1912–1954) introduced key concepts foundational to AI in the 1930s and 1940s. His 1936 paper "[On Computable Numbers](https://en.wikipedia.org/wiki/Turing%27s_proof)"[5] laid the groundwork for modern computing.

**The Birth of modern AI as a field** is considered to be in the year of 1956, when the term "Artificial Intelligence" was first coined in 1956 by John McCarthy at the [Dartmouth Conference](https://en.wikipedia.org/wiki/Dartmouth_workshop). [6]
>  "We propose that a 2-month, 10-man study of artificial intelligence be carried out during the summer of 1956 at Dartmouth College in Hanover, New Hampshire. The study is to proceed on the basis of the conjecture that every aspect of learning or any other feature of intelligence can in principle be so precisely described that a machine can be made to simulate it. An attempt will be made to find how to make machines use language, form abstractions and concepts, solve kinds of problems now reserved for humans, and improve themselves. We think that a significant advance can be made in one or more of these problems if a carefully selected group of scientists work on it together for a summer."

Projects like [ELIZA (a chatbot program by Joseph Weizenbaum, 1966)](https://en.wikipedia.org/wiki/ELIZA) and [Shakey the Robot ](https://en.wikipedia.org/wiki/Shakey_the_robot)(the first general-purpose mobile robot, 1969) demonstrated the possibilities of AI.
However, the ambitious early goals led to disappointment when real-world complexities proved harder to solve than expected. This led to the first AI Winter, a period of reduced funding and interest.
AI saw a brief resurgence in the 1980s, particularly with the development of [Expert Systems](https://en.wikipedia.org/wiki/Expert_system), which were rule-based systems designed to emulate the decision-making abilities of human experts.
During this time, machine learning techniques like neural networks began to be explored again after being dormant for decades, following John Hopfield’s work on neural networks in 1982.
The limitations of expert systems and continued difficulties in scaling AI led to a second AI Winter (late 1980s - mid 1990s)[7].
Then, as the computer hardware industry advanced, researchers noticed that they could perform more complex calculations and simulations on faster machines. Companies like Intel, IBM, and others continuously pushed the boundaries of chip design and computational power, enabling AI researchers to explore more intensive tasks that were previously impossible. The rise of the Internet, and the notable successes, such as [IBM’s Deep Blue](https://en.wikipedia.org/wiki/Deep_Blue_(chess_computer)) defeating world chess champion Garry Kasparov in 1997, demonstrated to the broader public that AI was becoming more capable.

AI since then has been embedded in many aspects of daily life, especially with the development of a **Deep learning** model in 2012 that became a game-changer. 

More recent notable events in the history of AI include:
   - 2012: The deep learning model **[AlexNet](https://en.wikipedia.org/wiki/AlexNet)** revolutionized image recognition by winning the ImageNet Large Scale Visual Recognition Challenge (ILSVRC)
   - 2014: Ian Goodfellow introduces **[Generative Adversarial Networks (GANs)](https://en.wikipedia.org/wiki/Generative_adversarial_network)**, a new approach for generating synthetic data (birth of "AI art")
   - 2016: Google DeepMind’s **[AlphaGo](https://en.wikipedia.org/wiki/AlphaGo)** becomes the first AI to beat a professional Go player, showing AI's ability to master complex, intuitive tasks
   - 2020: Release of **[GPT-3](https://en.wikipedia.org/wiki/GPT-3)**, a language model with 175 billion parameters, setting a new standard for natural language generation and understanding
   - 2022: Release of **[DALL·E 2](https://en.wikipedia.org/wiki/DALL-E)**, an AI model capable of generating detailed images from text prompts, marking a significant advancement in AI-driven creativity and image generation


[2](https://en.wikipedia.org/wiki/Boolean_algebra#/media/File:Vennandornot.svg)
[3](https://en.wikipedia.org/wiki/Logic#/media/File:First-order_logic.png)
[4](https://www.britannica.com/topic/formal-logic/Interdefinability-of-operators)
[5] https://www.cs.virginia.edu/~robins/Turing_Paper_1936.pdf
[6] http://jmc.stanford.edu/articles/dartmouth/dartmouth.pdf
[7] https://www.holloway.com/g/making-things-think/sections/the-second-ai-winter-19871993


### 1.2 Agents


#### Types of Agents
* Reactive Agent
* Model-Based Agent
* Planning Agent
* Utility-based agent
* Game Playing Agent
* Learning Agent



**Reactive Agent:**
"If condition, then action".
The Reactive agent is based on the condition-action rule. It has no memory or "state". 

![image](https://github.com/user-attachments/assets/fe8894d6-6833-42e3-825d-8b41389e4f8c) [1]

**Model-based Agent**
aka Model-based "reflex" agents 

handle partially observable environments. Percept history and impact of action on the environment can be determined by using the internal model. It then chooses an action in the same way as reflex agent. Can look into the past, but not into the future. Therefore performs poorly at tasks that require multiple steps of reasoning.

![image](https://github.com/user-attachments/assets/aae3e015-cb1a-489b-b2b4-e5161f4fbce5) [2]

**Planning Agent**
aka "Goal-based" agent (expands on the capabilities of the model-based agents)
"What will happen if I do such and such?" "Will that make me happy?"

![image](https://github.com/user-attachments/assets/63e69574-fe3b-451a-97de-593c765fbabb) [3]

**Utility-based Agent**
(Also expands on the capabilities of the model-based agents)
"How happy will I be in such a state?"
Tries to maximise expected 'happiness'.
![image](https://github.com/user-attachments/assets/f49b0bc9-2ad4-4b5e-9dd7-afafd86720a0)[4]

**Game Playing Agent**
( not to be confused with a General Game Playing (GGP) Agent )
 These agents are often fine-tuned and optimized for a single game. Examples of game playing agents include:

* Chess AI like Stockfish or AlphaZero (for Chess).
* Go AI like AlphaGo.
* Poker AI like Libratus.

The diagram of Game Playing Agent would be similar to the Utility-based Agent, with an additional step of modeling the opponent and evaluating moves with the goal of winning. The focus is more on minimizing opponent advantage or maximizing its chance of winning (via adversarial search like Minimax, or alpha beta pruning).
![chrome_LxrkTHZIFE](https://github.com/user-attachments/assets/9c1bd7e7-1fa8-46be-ba58-5ee5cfa53c61)

**Learning Agent**

A typical learning agent has four components:

   - **Learning Element**:
uses feedback from critic to makes adjustments for future actions. It refines its decision-making processes, often using algorithms like reinforcement learning, supervised learning, or unsupervised learning.
   - **Performance Element**:
a.k.a. actor, an element that takes actions; uses the current knowledge to make decisions and take actions.
   - **Critic**:
evaluates the performance of the agent by comparing the actual outcomes of its actions to the desired outcomes (i.e., its goals).
The feedback from the critic helps the learning element understand which actions were successful and which need improvement.
   - **Problem Generator**:
creates new tasks to provide new challenges, or to gain information from new experiences.
It generates new scenarios or tasks for the agent to tackle, encouraging the agent to explore new solutions and learn from them. This part is critical for expanding the agent’s understanding beyond routine tasks.


![image](https://github.com/user-attachments/assets/eac69188-2c93-4144-b4d1-4502a0228aea) [5]

Wikipedia and the original source of the classification ([Russel & Norvig (2003) pp.46-54](https://aima.cs.berkeley.edu/)) provides a great overview of these different types of agents.[6]



### 1.3 Knowledge Representation

**Representation and Search**

The world model must be represented in a way that makes reasoning easy
Reasoning in AI (i.e. problem solving and planning) almost always involves some kind of search among possible solutions





Here’s a hierarchical framework used to control a robot (soccer robot), moving from sensor data gathering (low-level) to game strategy (high-level).

![image](https://github.com/user-attachments/assets/b98b869f-7f7f-4eb7-9256-5449f7f1c7a4)



 Note how abstraction transforms raw numerical data into higher-level, qualitative information.

![image](https://github.com/user-attachments/assets/ea756f01-e72c-436d-9dbe-ee36e18a7d02)



#### 1.3.1 Feature-based vs Iconic Representations  

**Iconic Representations (Low-level):**
* Analogous to the real world
  * Pixel representations like the first layer of ANN
  * Maps 
* Fast, but difficult to generalize
* Numeric/statistical, offering a detailed representation
* No relational information, making reasoning difficult
* Memory-intensive
* Suited for tasks like vision and processing sequential data
* Difficult to perform inferences beyond pattern-response

**Feature-based (Symbolic) Representations (High-level):**
* State represented by a set of abstract features and relations
  * Logical expressions
  * Entity-relation graphs
* Can do complex reasoning over **knowledge base**
* Contains relational information, making it easier to reason about
* Facilitates generalizations, and is memory efficient
* Not well-suited for "low-level" tasks like perception

**Knowledge Base?**

A knowledge base is an explicit set of sentences about some domain expressed in a suitable **formal representation language**.
Sentences express facts or non-facts (true or false) e.g. "Q1 revenue was $10 million."

The knowledge base may use rules as a mechanism for reasoning (i.e. It is a **Rule-based** system):
e.g. "If <...> then <...>, If a patient has a fever and a headache, then the patient can potentially benefit from malaria testing."

Rules can define a network (aka "Inference" network) which shows the interdependencies among the rules.
Inference network shows which facts can be logically combined to form new facs or conclusions. The facts can be combined using "and", "or", "not"
**Inference**: Deriving new information or conclusions from the facts and rules, e.g. "Given a drop in sales and increased competition, product prices should be lowered to maintain market share."

![image](https://github.com/user-attachments/assets/1fc3efc7-1e25-446a-9d71-220f4192e722)

There are three reasoning processes to make inferences or draw conclusions from a set of premises or observations:
 1. **Deduction**: based on concrete facts, the process of reasoning from general principles or rules to specific conclusions. If the premises are true, the conclusion must be true.
    "If _<rule> and <cause>_, then _<effect>_" e.g. "If _Joe Bloggs works for ACME_ and _is in a stable relationship_, then _he is happy_."
 3. **Abduction**: hypothesis-driven, 'flipped' version of deduction, starts with observations or facts and infer the most likely rule that could explain the observation.
    "Given _<rule> and <effect>_, infer by abduction _<cause>_" e.g. "If _Joe Bloggs is happy_, infer by abduction _Joe Bloggs enjoys domestic bliss and professional contentment_"
    (Scientists develop medicines using abduction)
 5. **Induction**: probabilistic, pattern-based, generilizing from repeated observations, but the conclusion might not always be true:
    "If _<cause> and <effect>_ then _<rule>_" e.g. "If _every crow I have seen is blue_, _all crows are blue_"

( Later in chapter 9 we will explore different ways to deal with uncertainty in rules, e.g.:
 - **Vague rule**: Fuzzy logic, where truth values range between 0 and 1, representing the degree to which a statement is true.
 - **Uncertain link between evidence and conclusion**: Bayesian inference, which relates the conditional probability of a hypothesis given some observed data to the likelihood of the data under that hypothesis.
 - **Uncertain evidence**: 🤷 )

**Fundamental questions:**
- How do we write down knowledge about a domain/problem?
- How do we automate reasoning to deduce new facts or ensure consistency of a knowledge base?



#### 1.3.2 Logic  

**Propositional Logic
 - Letters stand for "basic" propositions
 - Combine into more complex sentences using AND, OR, NOT, IFF, ... 
 - e.g.
    - P = "It rains", Q = "I will bring an umbrella"
    - P → Q "If it rains, I will bring an umbrella."

**First-order logic**
 - An extension of Propositional Logic for a more expressive representation (including relations and ontologies)
 - Terms: constants, variables, functions applied to terms
    - e.g. a, f(a), motherOf(Mary), ...
 - Atomic Formulae: predicates applied to tuples of terms
    - e.g. likes(Mary, motherOf(Mary)), likes(x, a), ...
 - Quantified formulae:
    - e.g. ∀x Animal(x) → HasHeart(x), ∃x loves(x, Mary)
      - "For all x, if x is an animal, then x has a heart"
      - "There exists at least one person who loves Mary"

#### 1.3.3 Learning Rules  

**Oncological engineering** 

An **Ontology** is, in a simple term, information mapping. It is a formal representation of knowledge that describes concepts, categories, and relationships within a particular domain. 
Like OOP, child concept is a specialisation of parent concept. Also, child inherits property of parent. 
Ontology includes not just the hierarchy of concepts (aka Taxonomic hierarchy), but also the properties and relationships between them, enabling reasoning and inference.
e.g. Dog is a subclass of Mammal. Thus, dogs can have diseases. Diseases can be trated by medication.


![image](https://github.com/user-attachments/assets/4cbbeb64-ff39-48dc-8a77-910a363c8108)

We can infer the type of object from its attributes, e.g.

Guess what this object is:

- Fruit category
- Green and yellow mottled skin
- 30cm diameter
- Ovoid shape
- Red flesh, black seeds
  
= Watermelon




**Reasoning System for categories**


Two closely related families of systems:

- **Semantic networks** aka. Associative network
   - A graph-based representation of knowledge that emphasizes the relationships between entities (concepts)
   - Models Facts, Objects, Attributes, Relationships
   - Its application includes Natural Language Processing (NLP), Knowledge Graphs

![image](https://github.com/user-attachments/assets/cb1ad1c3-32b6-4046-a808-20f5b362cddb)


- **Description logics**
   - A more formal and rigorous knowledge representation system than semantic networks
   - provides precise mathematical definitions for Facts, Objects, Attributes, Relationships
   - e.g. defines "Parent" as "someone who has at least one child" or "a mammal is an animal with fur and gives live birth."
   - a foundation for ontology-based systems




[1] https://en.wikipedia.org/wiki/Intelligent_agent#/media/File:Simple_reflex_agent.png
[2] https://en.wikipedia.org/wiki/Intelligent_agent#/media/File:Model_based_reflex_agent.png
[3] https://en.wikipedia.org/wiki/Intelligent_agent#/media/File:Model_based_goal_based_agent.png
[4] https://en.wikipedia.org/wiki/Intelligent_agent#/media/File:Model_based_utility_based.png
[5] https://en.wikipedia.org/wiki/Intelligent_agent#/media/File:IntelligentAgent-Learning.png
[6] https://en.wikipedia.org/wiki/Intelligent_agent


(Add Mountain Car problem, Sudoku, simple webGL based simulation to test viewer's understanding of KRR)


---

## 2 [Search](#2-search)

You will learn:
- the theory behind search in AI
- to develop a smart agent with search capability to solve interesting problem




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


State space
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

## 2.3 Uninformed vs Informed Search

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


## 2.4 Uninformed Search

Tree is often used to represent Uninformed Search algorithms, since we dont have to worry about revisiting nodes (since there are no cycles).


![image](https://github.com/user-attachments/assets/519e3fc9-0139-4211-8967-2d9ebf454f19)

A node is said to be Expanded* when the algorithm has examined all their neighboring nodes and added them to the search space.

**4 Types of Nodes in Uninformed Search Tree:**
1. Root Node = Where the search begins. contains the initial state of the algorithm.

2. Expanded* Nodes (Black)

4. Generated Nodes (Black and Red) = Nodes that the algorithm has already visited (i.e. the algorithm has either already expanded* or is about to expand the node.) 

5. Frontier (Red): Nodes that have been generated but not yet expanded*. In other words, the algorithm will explore these next by expanding them and looking at their neighbors.


### 2.4.1 BFS

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

### 2.4.2 DFS 


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

### 2.4.3 Depth Limited Search

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


### 2.6 Solving Problems using Search

Exercise -
1. Define the states, actions and transition function
2. Convert above definition into a machine readable format
3. Choose a search strategy, considering the context & problem
   
 (give ~5 problems .. what did you choose? why?)




## 3. Artificial Neural Networks

Humans can perform complex tasks: Shape recognition, Speech processing, Image processing ... 
To emulate these behaviours, a branch of artifficial intelligence formed inspired by [neural circuitry](https://en.wikipedia.org/wiki/Neural_circuit):
[**ANN (ARtificial Neural Networks)**](https://developer.nvidia.com/discover/artificial-neural-network)

**High Connectivity**
ANNs are composed of layers of nodes (neurons). Each node in one layer connects to nodes in the next layer, forming a network, where it learns patterns from the relationships and make prediction. 

**Parallelism**
Neurons process multiple tasks simultaneously rather than sequentially
![KY2I87](https://github.com/user-attachments/assets/13c40609-a1dd-4043-af65-aa20f2219453)


ANN:
- excels at Pattern recognition and Forecasting
- uses a newer, non-algorithmic paradigm to process information through learning, adaptation, and parallel processing
- is essentially a black box; you can observe the inputs and outputs transformed through multiple layers, but the inner workings are unknown

![image](https://github.com/user-attachments/assets/ad86256f-31e1-4479-87cc-2c429eacd291)
(image source: https://developer.nvidia.com/discover/artificial-neural-network)



**How does ANN learn?**
- uses **Generalization** to perform well on unseen data that has not been encountered before
- uses **Function Approximation** to estimate a function that maps intputs to outputs based on a set of observed data points.
  The aim is to find a function that closely represents the relationship between the input features and output targets.
  Even if the exact function is unknown.



**Function Approximation**
 can either be:
  - used on an entity (=a set of input variables, either continuous or discrete) to output discrete values that represents which class the entity belongs to (=membership) -> **Classification**  (e.g. character recognition, cats vs dogs from image)
  - used to predict continuous outcomes based on input variables. -> **Regression** (e.g. predicting the house price based on location, number of bedrooms, etc)


### 3.1  Biological Neurons vs Artificial Neurons
The brain is made up of [neurons (nerve cells)](https://en.wikipedia.org/wiki/Neuron) which have
• a cell body (soma)
• dendrites (inputs)
• an axon (outputs)
• synapses (connections between cells)

![image](https://github.com/user-attachments/assets/5a27c3e2-5319-4f66-8569-4e98e90e7a17)
(Image: [Alan Woodruff ; De Roo et al / CC BY-SA 3.0 via Commons](https://qbi.uq.edu.au/brain/brain-anatomy/what-neuron))

(Kinda looks similar to a tree; it receives energy through its leaves (dendrites, inputs), the energy goes through the body (soma), and reaches the root (axon, outputs). The axon is connected to other neurons and transfers electrical signals to them through synapse. When the inputs reach some threshold, an action potential (electrical pulse) is sent along the axon to the outputs.
This threshhold may change over time, depending on how **excitatory** (promoting the firing) or **inhibitory** (reducing the likelihood of firing) the synapse is.

We call this **synaptic plasticity**, which is crucial for learning and memory.
The strength of a connection between two neurons is called **synaptic weight**; it corresponds to how much influence the firing of a neuron has on another.


The synaptic weight is changed by using a learning rule, the most basic of which is Hebb's rule, which is usually stated in biological terms as
```
 Neurons that fire together, wire together.
```
**Hebbian learning (1949)**
"When a neuron A persistently activates another nearby neuron B, the connection between the two neurons becomes stronger. Specifically, a growth process occurs that increases how effective neuron A is in activating neuron B. As a result, the connection between those two neurons is strengthened over time"


More reading: https://en.wikipedia.org/wiki/Synaptic_weight

**Artificial Neurons** 

The first computational model of a neuron() was proposed by Warren MuCulloch (neuroscientist) and Walter Pitts (logician) in 1943.


More reading: https://towardsdatascience.com/mcculloch-pitts-model-5fdf65ac5dd1

It has 4 components:
- Inputs
- Weights
- Transfer Function
- Activation Function

![image](https://github.com/user-attachments/assets/187c34da-d4a8-47e1-8843-aa1e842329a9)
(Image: https://en.wikipedia.org/wiki/Artificial_neuron)

vs.

![image](https://github.com/user-attachments/assets/0dd12d34-d844-4bdc-9e20-7c13c1fb5263)
Image source: https://en.wikipedia.org/wiki/File:Neuron3.svg

The inputs, after being multiplied by their respective **weights**, which reflect the importance or influence of that input to the node, are summed and then passed through an activation function to produce the final activation level.

The activation function has a **threshhold** to determine whether the neuron has fired or not. ( 0 or 1 activation value )


![image](https://github.com/user-attachments/assets/b328bc3d-005c-447b-a8bd-0cf9aeca87ac)

- x_i  = input values
- w_ij = weights corrresponding to the inputs
- g(s_j) = the activation function applied to the weighted sum

![image](https://github.com/user-attachments/assets/76b45693-b9da-4e49-84fd-119c69b7be1f)

Activation Functions:
- Sign (Step) function
- Semi-linear (piecewise linear) function  
- Sigmoid: Smooth and non-linear


![image](https://github.com/user-attachments/assets/d46c4613-dc64-4e40-b5c3-11b66375b37c)

Examples of Non-linear Activation Functions:
- Sigmoid
- ReLU (Rectified Linear Unit)
-  Tanh (Hyperbolic Tangent)

Non-linear functions enable the network to stack multiple layers and allow each layer to capture increasingly abstract and complex features from the data


The activation level is the result of the node's internal computation, which is usually a non-linear function of the inputs. 

Some limitations of MP artificial neuron:
(1) it only works with binary inputs and outputs; not with real numbers
(2) it does not evolve or learn.  its functionality is limited to problems that can be derived by the modeler

More reading: https://jontysinai.github.io/jekyll/update/2017/09/24/the-mcp-neuron.html
https://com-cog-book.github.io/com-cog-book/features/mp-artificial-neuron.html


### 3.2 Single Layer Perceptron


Frank Rosenblatt, an American psychologist, proposed the classical **Perception** model in 1958. It is more generalized computational model than the McCulloch-Pitts neuron where weights and thresholds can be learnt over time.

Early ideas on how information is stored and processed in artificial intelligence and cognitive science was divided into two approaches: **coded representations** and **connectionist approaches**. Let's say you perceived a triangle, then, the former states that a triangle-shaped image would be "carved" in the memory space. Once the shape is carved, you would be able to precisely retrieve the information stored in a particular location in the brain. 

The latter states that memories are stored as preferences for a particular "response" rather than "topographic representations". Instead of a unique cluster of neurons wired in an invariant manner, that "code" the memory of a triangle, what we have instead is a series of associations among a set of neurons that "tend to react" to the stimulus that may remind you of a trinagle.


Rosenblatt took elements from the works of Hebb and summarized the nature of the cognition as the problems of (1) detection, (2) storage, and (3) the effect of the stored information.
```
1. The physical connections participating in learning and recognizing a stimulus can vary from organism to organism.
2. The cells involved in learning and recognition are flexible, meaning that the probability of activation, when exposed to a stimulus, can change over time.
3. Similar stimuli will tend to trigger similar response patterns in the brain.
4. When memory is forming, the application of a positive and/or a negative reinforcement may facilitate or hinder the process.
5. Similarity is not determined by the perceptual properties of the stimuli, but by a combination of the perceiving system, the stimuli, the context, and the history of interactions between those elements. This last principle indicates that you can't decouple perception and representation from the perceiving system (the organism itself).
```


**Perceptron** is an algorithm used to classify inputs into one of two possible categories.

The perceptron makes the classification decision by applying its discriminant function to the input. 
This function is typically linear and divides the input space into two regions, one for each class.

The perceptron draws a hyperplane in the input space. Any input that falls on one side of the hyperplane is classified as one class, and inputs on the other side are classified as the other class.

![image](https://github.com/user-attachments/assets/b48fe94d-1f06-4462-aae9-1bd06e3fe4c2)


![image](https://github.com/user-attachments/assets/bddf10c1-ffea-489e-a297-f12032fe5e42)

The perceptron is able to solve only linearly separable functions,  (or hyperplane in higher dimensions)

e.g. 
AND Gate
![image](https://github.com/user-attachments/assets/23db5e67-addd-4f64-a051-2d94a05d5de5)

(0, 0) → 0
(0, 1) → 0
(1, 0) → 0
(1, 1) → 1

When plotted, only the point (1,1) is activated.
Can be divided into two classes by a single straight line (linear decision boundary)

XOR Gate

![image](https://github.com/user-attachments/assets/a8055a9b-bccd-4c3c-92e3-64941b22257a)

(0, 0) → 0
(0, 1) → 1
(1, 0) → 1
(1, 1) → 0

When plotted, the points (0,1) and (1,0) are situated diagonally opposite.
= Cannot be divided into two classes by a single straight line (linear decision boundary)
= the XOR gate is non-linearly separable: it needs something more than just a single line to classify the points correctly.


**Learning Rule**

A learning rule is a set of instructions that governs how a model's weights are adjusted during the training process to improve its performance.
The goal is to minimize errors in its predictions by modifying its internal parameters (weights) based on the input data and corresponding outputs (labels).

If the perceptron makes a correct prediction, the weights remain the same.
If the perceptron underpredicts (i.e., it predicts -1, but the correct label is 1), the weights are increased.
If the perceptron overpredicts (i.e., it predicts 1, but the correct label is -1), the weights are decreased.

This process is repeated until the perceptron can correctly classify all training examples (or until it hits the max attempt)

![image](https://github.com/user-attachments/assets/12234fd3-d1ee-43e4-83fd-8ebbbbc82d7f)



### 3.3 Multilayer Perceptron (MLP)

While the single-layer perceptron is simple and efficient for binary classification, it can only separate linearly separable data. 
To solve non-linearly separable data such as XOR, you need a multi-layer perceptron (MLP)

MLP (Multi-Layer Perceptron) consists of multiple layers of neurons organized in such a way that it can model complex relationships. 
It is one of the foundational architectures in deep learning and machine learning.

MLP is a **Feedforward** network:
- Data flows through the network in one direction, from the input layer to the hidden layers and finally to the output layer. (No feedback loops or backward connection)
- Each neuron in a layer is connected to every neuron in the next layer (this is called a fully connected layer).
- At each neuron, the input is multiplied by weights, summed up, and passed through an activation function (e.g., sigmoid, ReLU).

MLP neural network architecture

![image](https://github.com/user-attachments/assets/d1bac3c3-4286-4562-99dd-d7d23328675b)


**Backpropagation**
If there is an error between the actual output and the target, we calculate the error (difference between the target and actual output) and propagate errors back through the network to update the weights.

![image](https://github.com/user-attachments/assets/4a86d578-86cd-43df-b4bf-203be18c0536)


Backpropagation is typically paired with **gradient descent** (or its variants, such as stochastic gradient descent) to minimize the loss function by adjusting the weights in the direction that reduces the error.

![image](https://github.com/user-attachments/assets/480c03e8-8149-4bb8-a365-85bc0f5faa7c)

**Error function (Loss function)**
-> measures how well or poorly the neural network is performing
![image](https://github.com/user-attachments/assets/d411e107-2580-4ae4-9264-4a9e83bbebaa)


**Gradient descent**
-> aims to minimize the error function by adjusting the weights
![image](https://github.com/user-attachments/assets/356b24c5-a999-4329-8a5d-d2503452a018)


**Backpropagation Diagram**
![image](https://github.com/user-attachments/assets/4755d9fa-0141-43e5-b21c-6b0c4d035cb1)

The error (E(t)) is computed as the difference between the predicted output Y(t) and the target value.
The error is then propagated backward through the network, and gradients are calculated for each weight.
The weights (W_o, W_i) are updated using gradient descent to reduce the error based on the computed gradients.

![image](https://github.com/user-attachments/assets/ded640c1-17b2-41b9-a037-671c27ba339a)

![image](https://github.com/user-attachments/assets/c6430ad2-afec-4e61-91d0-88c6d14fda5a)




### 3.4 Neural Network Design

### 3.5 Neural Network Architectures


