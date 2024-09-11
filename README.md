# Intro-to-AI

Welcome to the **Intro-to-AI** repository! This repository is crafted for self-learners who want to develop a solid foundation in Artificial Intelligence (AI). It covers a wide array of essential AI topics including neural networks, reinforcement learning, computer vision, and natural language processing.

The content is inspired by my personal notes and learnings from the course COMP9814: Extended Artificial Intelligence at UNSW (24T3). It assumes readers have a basic understanding of Python and mathematics at a sophomore university level, as well as a solid grasp of fundamental data structures and algorithms.

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
Artificial intelligence is, well, intelligence that is **artificial**. But what does *Integllience* really mean? We throw the word around, but defining it is surprisingly tricky.

Is intelligence just being really good at math? Or is it creativity? Is it the ability to learn, adapt, or make decisions?
Humans are intelligent because we can navigate a complex world, learn from experience, solve problems, and, yes, create art, write stories, even have meaningful conversations. We've seen machines could do this too at least to some extent. (Some may argue that it's merely "copying" the original works of humans, but let's put that discussion aside for now). Take for instance, a chess AI, which is a "narrow" AI — it’s designed to do one thing, but do it really well. But is it truly intelligent? That machine doesn’t actually “understand” what it’s doing—at least not in the way humans do—but it’s performing tasks we’d consider intelligent.

A good umbrella definition might be:
 "the ability to perceive or infer information, and to retain it as knowledge to be applied towards adaptive behaviours within an environment or context"
... or to put it simply,
 "the ability to take in information, process it, and then make decisions based on it."

In fact, AI isn't just a product of modern times. The history of AI goes back thousands of years—long before computers or even electricity.

### 1.1 History of AI
 Let's rewind the clock—way back to 350 BC. Here we find Aristotle, not just a philosopher, but arguably one of the first minds to engage in what we would today call artificial intelligence. He pioneered logic—deductive reasoning, a way of drawing conclusions from facts. 

 It’s the backbone of problem-solving and decision-making in AI today. 

Aristotle's system laid the groundwork for what would later become Boolean logic, thanks to George Boole in 1848, and Frege’s work on formal logic in 1879.

[Boolean Logic](https://en.wikipedia.org/wiki/Boolean_algebra) is the foundation for modern digital computers, where circuits and operations work using binary (1s and 0s) to perform calculations and make decisions. Boole's work provided the basis for digital logic circuits, which today underpin everything from computer processors to simple calculators.

![image](https://github.com/user-attachments/assets/503c4304-e28c-4cc1-b563-1f46970d1bb2)[2]

* AND: Both conditions must be true (e.g., A AND B is true only if both A and B are true).
* OR: At least one condition must be true (e.g., A OR B is true if either A or B is true).
* NOT: A negation of a condition (e.g., NOT A is true if A is false).

Frege's formal system took a huge step beyond Boolean logic by introducing the concept of quantifiers and variables, which allowed more complex relationships to be expressed in a logical framework. While Boolean logic deals with simple true/false statements, Formal logic translates natural language arguments into a formal language, like first-order logic, to assess whether they are valid using quantifiers such as:

* Universal quantifier (∀): Indicates that a property holds for all elements in a domain (e.g., ∀x: P(x) means "for all x, P(x) is true").
* Existential quantifier (∃): Indicates that there is at least one element in a domain for which a property holds (e.g., ∃x: P(x) means "there exists an x for which P(x) is true").

![image](https://github.com/user-attachments/assets/b8ff5830-17db-47c9-9c5b-c3833b40f8eb)[3]



[2](https://en.wikipedia.org/wiki/Boolean_algebra#/media/File:Vennandornot.svg)
[3](https://en.wikipedia.org/wiki/Logic#/media/File:First-order_logic.png)
