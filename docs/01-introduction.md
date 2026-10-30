---
chapter: 1
description: What intelligence means, a short history of AI, agent types, knowledge representation.
---

# 1. Introduction

Artificial intelligence is, well, intelligence that is artificial. Artificial just means it is machine-based. But what does *Integllience* mean? We throw the word around, but defining it is surprisingly tricky. Is intelligence just being really good at math? Or is it creativity? Is it the ability to learn, adapt, or make decisions?

Humans are intelligent because we can navigate a complex world, learn from experience, solve problems, create art, etc. We've seen machines could do this too at least to some extent. (Some may argue that it's merely "copying" the original works of humans, but let's put that discussion aside for now). 
Take for instance, a chess AI, which is a "narrow" AI — it’s designed to do one thing, but do it really well. But is it truly intelligent? That machine doesn’t actually “understand” what it’s doing—at least not in the way humans do—but it’s performing tasks we’d consider intelligent.

 Here is a good umbrella definition (still abstract, but it suggests a direction):
  > Intelligence is the ability to perceive information, and to retain it as knowledge to be applied towards adaptive behaviours within an environment or context.

 So how did we even begin to come up with the concept of artificial intelligence? In fact, it goes back thousands of years—long before computers or even electricity.

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

 Agents in AI are entities that perceive their environment and act upon it, aiming to achieve specific goals. Different types of agents vary in complexity, decision-making abilities, and adaptability. 

#### Types of Agents
* [Reactive Agent](#reactive-agents)
* [Model-Based Agent](#model-based-agent)
* [Planning Agent](#planning-agent)
* [Utility-based agent](#utility-based-agent)
* [Game Playing Agent](#game-playing-agent)
* [Learning Agent](#learning-agent)



##### Reactive Agents {#reactive-agents}

At the most basic level, we have the Reactive Agent, a simple yet effective problem-solver. Think of it as the AI version of a reflex action: "If this condition is met, perform this action." There’s no deliberation, no contemplation of past actions or future consequences—just action based on the current environment.

For instance, imagine a thermostat that turns on the air conditioning when the temperature hits 75°F. It doesn't remember what happened yesterday or predict the weather tomorrow—it reacts purely to the present moment.

While this simplicity is useful in fast, real-time systems, reactive agents are limited. Without memory or state, they can’t learn or improve over time. They can only react.


> "If condition, then action".

It is simply following the rule, without memory or state.

![image](https://github.com/user-attachments/assets/fe8894d6-6833-42e3-825d-8b41389e4f8c)

 

##### Model-based Agent or Model-based "Reflex" Agent {#model-based-agent}


Now, let’s add a layer of sophistication. Unlike the reactive agent, this type of agent has an internal **model** of the world, allowing it to handle *partially* observable environments. It takes into account not just the current input, but also a history of past actions and their outcomes, allowing for a more informed decision-making process.

Consider a robot navigating a maze. It doesn’t just rely on what it currently sees but remembers where it has already been. This memory helps it avoid dead ends and make smarter choices moving forward. However, despite this extra capability, model-based agents are still "reflex" agents in a sense—they *respond* based on the past but does not have any capability to predict the future based on learning. Tasks requiring multiple steps of reasoning can stump them.

> "If condition, then action. ...wait, I've been here before."

![image](https://github.com/user-attachments/assets/aae3e015-cb1a-489b-b2b4-e5161f4fbce5)

##### Planning Agent or "Goal-based" agent {#planning-agent}

The Planning Agent expands upon the model-based agent by contemplating future consequences. It doesn’t just react to the past or present—it actively plans ahead, asking itself whether a given action will help achieve a specific goal.

This forward-thinking ability allows the planning agent to tackle complex tasks that require multiple steps, such as navigating a city using a map or solving a puzzle. By simulating possible actions and outcomes, it determines the best path to success.

> "What will happen if I take this action?"
> "Will that make me happy?"

![image](https://github.com/user-attachments/assets/63e69574-fe3b-451a-97de-593c765fbabb)

##### Utility-based Agent {#utility-based-agent}

While planning agents focus on reaching a goal, Utility-Based Agents aim for something even more nuanced, that is, **maximizing happiness**. They evaluate each potential action, not just based on whether it achieves a goal, but on how "happy" it will make them in the end. In other words, they assign a utility or value to each possible state and then pursue the path that promises the highest utility.

Take, for example, an autonomous car deciding between two routes. One route might be faster but more dangerous, while the other is slower but safer. A utility-based agent would weigh the pros and cons of each option, taking into account not only the time saved but the overall satisfaction of avoiding risk.

> "What will happen if I take this action?"
> "*How happy* will I be in such a state?"

![image](https://github.com/user-attachments/assets/f49b0bc9-2ad4-4b5e-9dd7-afafd86720a0)

##### Game Playing Agent {#game-playing-agent}
(not to be confused with the concept of [General Game Playing (GGP)](https://en.wikipedia.org/wiki/General_game_playing))

Game-playing agents take these ideas of planning and utility to a competitive level. These agents are specifically designed to excel at a specific game or domain. They’re not just interested in their own happiness—they’re actively working to minimize their opponent’s advantage (via adversarial search like [Minimax](https://en.wikipedia.org/wiki/Minimax), or [Alpha Beta Pruning](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning)).

Examples of game playing agents include:

* Chess AI like Stockfish or AlphaZero (for Chess)
* Go AI like AlphaGo
* Poker AI like Libratus

They usually excel in their domains, but their expertise doesn't easily transfer to other areas.


> "What will happen if I take this action?"
> "*How happy* will I be in such a state?"
> "Will my enemy be happier?"

![chrome_LxrkTHZIFE](https://github.com/user-attachments/assets/9c1bd7e7-1fa8-46be-ba58-5ee5cfa53c61)

##### Learning Agent {#learning-agent}

At the cutting edge of AI is the Learning Agent. This type of agent doesn’t just act based on predefined rules or models—it learns from its experiences and continually improves over time. 

 The learning agent is the closest thing we have to an agent that grows in intelligence, and it has several key components:
   - **Learner**
The agent receives feedback on its actions and adjusts its future decisions based on this feedback. Techniques like reinforcement learning and supervised learning are commonly used here.
   - **Actor**
This is the part of the agent that actually takes action. It uses the knowledge and strategies it has learned to make decisions.
   - **Critic**
Acting as an internal evaluator, the critic compares the outcomes of the agent’s actions against the desired results. This feedback is essential for the learning process, helping the agent understand which actions are beneficial and which are not.
   - **Problem Generator**:
To grow smarter, an agent must explore new challenges. The problem generator encourages the agent to tackle new tasks, forcing it to learn and adapt to unfamiliar situations. 

Initially, it may be slow and make mistakes. But as it learns from feedback, it optimizes its tasks and becomes more reliable and efficient.

![image](https://github.com/user-attachments/assets/eac69188-2c93-4144-b4d1-4502a0228aea)

The original source of the classification ([Russel & Norvig (2003)](https://aima.cs.berkeley.edu/)) provides a holistic overview of these different types of agents.



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
Sentences express facts or non-facts (true or false) e.g. "Q1 revenue was \$10 million."

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

**Propositional Logic**
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


**Ontology**  

An ontology is a formal structure that defines the relationships between different concepts in a domain. It’s often used in AI to infer new facts from known data.

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
