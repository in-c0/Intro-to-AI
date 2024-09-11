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

### 2. Search
- 2.1 Uninformed Search  
- 2.2 Informed Search  
- 2.3 Informed vs Uninformed Search  

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



## 1. Introduction
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



**Representation and Search**

The world model must be represented in a way that makes reasoning easy
Reasoning in AI (i.e. problem solving and planning) almost always involves some kind of search among possible solutions.


[1] https://en.wikipedia.org/wiki/Intelligent_agent#/media/File:Simple_reflex_agent.png
[2] https://en.wikipedia.org/wiki/Intelligent_agent#/media/File:Model_based_reflex_agent.png
[3] https://en.wikipedia.org/wiki/Intelligent_agent#/media/File:Model_based_goal_based_agent.png
[4] https://en.wikipedia.org/wiki/Intelligent_agent#/media/File:Model_based_utility_based.png
[5] https://en.wikipedia.org/wiki/Intelligent_agent#/media/File:IntelligentAgent-Learning.png
[6] https://en.wikipedia.org/wiki/Intelligent_agent
