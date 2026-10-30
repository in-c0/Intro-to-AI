---
chapter: 3
description: Neurons, perceptrons, MLPs and a practical network-design workflow.
---

# 3. Artificial Neural Networks

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


### 3.1 Neurons - Biological and Artificial

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


### 3.2 Single-Layer Perceptron


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

More reading: https://en.wikipedia.org/wiki/Multilayer_perceptron

https://scikit-learn.org/stable/modules/neural_networks_supervised.html



### 3.4 Neural Network Design

#### Step 1: Exhaustive System Analysis

Before choosing a neural network, you should consider other traditional models (e.g. phenomelogical models) that might be more appropriate for the problem. Neural networks are powerful but also more complex, resource-intensive, and data-hungry. Simpler models may be easier to work with and could perform just as well depending on the problem.

Phenomenological Models: models based on domain-specific knowledge and physical laws, often used in scientific and engineering fields. e.g In physics, models based on Newton's laws to describe motion, or in biology, models to describe population growth.

Neural networks are powerful and flexible but should be considered a "second best solution" when classic models cannot perform well. It's a good idea to try simpler models first to save time and resource.
Many traditional models work well even with small datasets. Neural networks, on the other hand, perform better as the amount of data increases, while simpler models may struggle to scale. 

The **preprocessing** step in the development of a neural network or any machine learning model, involves identifying the most relevant features (input variables) and ensuring that unnecessary variables are removed. 
The output should also be clearly defined so that the neural network can be properly structured to predict the desired outcome.
- In classification tasks, the output might be a category or label (e.g., is the email spam or not?)
- In regression tasks, the output might be a continuous value (e.g., the position of Y over period X)

**Example:**
Let’s say you’re building a neural network to predict whether a person will buy a product based on the following features:

Age
Gender
Income
City
Favorite Color
Education Level
Number of Children
Marital Status
Temperature on the day of purchase

In this scenario:

Step 1 would involve identifying which variables (inputs) are truly important. For example, "Favorite Color" and "Temperature" might not be very important or relevant in predicting whether someone will buy a product, so you could remove those variables. You could also find that "Income" and "Education Level" are highly correlated, so you might only keep one of them in the model. The goal is to simplify the model by only keeping the most important features that are directly related to the prediction task.

#### Step 2: Preprocessing

When designing and training a neural network:
- Ensure that your data is representative of the problem you’re trying to solve
- Obtain high quality data, e.g. properly designed survey/experimental plan
- Make sure you have enough data to train the network
- Visualise and examine the data to eliminate any outliers, null values, and redundant variables (if two variables are highly correlated, it might indicate redundancy)
  
 Neural networks are **empirical** model (i.e. learns from experience and observation rather than logic) for **interpolation**, not **extrapolation**— meaning it is not suitable for making predictions outside the range of the data they were trained on. For example, if a neural network is trained on data within a certain range (e.g., temperatures between 50°F and 100°F), it may perform poorly if asked to predict temperatures outside that range (e.g., 120°F).

 **Normalization** of input variables on different scales or in different units is essential. 
 Example normalization of variables so that they lie within a similar range (e.g., [0, 1] or [-1, 1]):
 ![image](https://github.com/user-attachments/assets/731753db-36b9-476a-8c57-5aceb12ffb0e)

 
#### Step 3: Neural Model Design

The number of input neurons and output neurons depends on the results from Steps 1 and 2. 
But deciding on the number of neurons in the hidden layer (Nh) is not straightforward and often requires experimentation or following certain heuristics.
Adding too many neurons can lead to overfitting (the model memorizes the training data rather than learning general patterns)

The rule of thumb is to carefully choose Nh to keep the number of parameters/weights (Nw) below 10% of the total number of samples in the training dataset.
```
Nw < (Number of samples) / 10
```
This rule helps ensure that the model has enough training data relative to its complexity (number of parameters) to learn general patterns rather than overfitting.

Formula for the number of parameters (weights) for an MLP:
![image](https://github.com/user-attachments/assets/5cceb768-8f86-48ae-8085-ccd61ba95fcc)

e.g. Let's say it is a binary classification problem, and you have 1000 samples in your dataset,

you have 5 input neurons, [ 10 ] neurons in the hidden layer, 2 output neurons,
Then the total number of weights Nw in the network would be: (5+1)*10 + (10+1)*2 = 82, which is smaller than the 10% of the samples (100), which makes it an adequate size relative to the dataset. 


[Universal Approximation Theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem) states that an MLP with one hidden layer and a sufficient number of neurons can approximate any non-linear function to any desired degree of accuracy, though in practice, deeper networks are often preferred for better performance.

ReLU and Sigmoid functions are commonly used in hidden layers to introduce non-linearity and allow the network to learn complex patterns.
In the output layer, the choice of activation function depends on the task:
- Linear activation for regression tasks
- Sigmoid for binary classification
- Softmax for multi-class classification


#### Step 4: Training

The process of training involves finding the set of weights that minimize the loss function (=the error function that represents how far off the model’s predictions are from the actual values).
However, this process can be difficult due to the complexity of the error function's landscape (solution space).

The error function solution space is often highly complex, meaning it can have many:
**Local minima**: Points where the error is lower than the surrounding points but is not the global minimum (the true lowest point in the solution space). Gradient descent may get stuck in these local minima.
**Saddle points** (minimax points): Points where the gradient is zero, but they are neither a maximum nor a minimum. These points can slow down training since the model might get stuck in flat areas of the solution space.

Local minima are points in the error (or loss) function where the gradient (slope) is zero, but they are not the global minimum (the best possible solution). During training, neural networks use optimization techniques like gradient descent to minimize the error function, but the error surface is often very complex with many local minima. Due to this complexity, depending on the starting point (initial weights), the network might settle into a local minimum instead of the global minimum, resulting in a suboptimal solution.

**Solution**: To increase the chance of finding the global minimum, run the training processes multiple times (20+) with different random initial weights. 

• **Bias**
= the error introduced by assuming that the underlying relationship between the inputs (features) and outputs (target) is simpler than it actually is, aka the model "underfitting" the data
![image](https://github.com/user-attachments/assets/939bb0a5-7374-4362-b395-907ddd3225c3)

On the other hand, we can also overdo the training and induce:
• Overparameterization (High variance)
• Overfitting (High variance)

where the model learns from too much noise and specific details around the underlying patterns.

**High variance** (of the model'sperformance depending on the dataset, i.e. performs well on the training set but poorly on new, unseen set), aka "overfitting"
![image](https://github.com/user-attachments/assets/700a8836-183d-4b78-8530-f059eccaf5bd)

##### Data Splitting 

To avoid overfitting, we use two sets for training: the **Training Set** and **Test Set** (plus optional **Validation Set**) 

**Training Set**
=> the model tries to fit the training data as closely as possible. (i.e. the model computes and minimizes the error function based on the training set)

**Test Set**
=> the model cannot see the test set during training. It is used to evaluate the model's performance after training, as a a simulated real-world check using unseen data

A common practice is to split the data into 80% training set and 20% test set. This allows the model to learn from a majority of the data while still having enough data to evaluate its performance.

**Simultaneously visualizing the error function** on both the training set and test set is a useful practice to understand how well your model is performing over time.

You can plot the training error and test error as a function of the training iterations. This allows you to observe patterns such as:
- If the training error continues to decrease but the test error starts increasing, the model is likely overfitting. This is because it is memorizing the training data while failing to generalize to the unseen test data.
- If both the training and test errors decrease and converge, the model is learning well and generalizing appropriately

![image](https://github.com/user-attachments/assets/7a5642e6-c920-4412-bb72-03d9db7810c5)

As training progresses, (as the number of **epochs** increases = the number of iterations of all the training data in one cycle for training the model increases)
you can monitor both training and test errors and stop the training process once you reach the minimum test error. This is called early stopping and is a common technique to prevent overfitting.

You can determine the optimal number of epochs as the point at which the test error curve (dotted line) is at its minimum (i.e., right where the graph marks "Minimum test error").

Both the training and test sets should be large enough to be statistically representative of the entire dataset.

More reading: https://en.wikipedia.org/wiki/Training,_validation,_and_test_data_sets

##### Cross-Validation
A more advanced technique is **k-fold** cross-validation, where the dataset is divided into k subsets. The model is trained k times, each time using a different subset as the test set and the remaining k−1 subsets as the training set. 
The results can be averaged to produce a more reliable estimate of the model's performance.
Or, sometimes, different neural network models are developed using the available data, splitting the training and test sets in different ways. The model that achieves the minimum error on the test set is chosen.


#### Step 5: Generalisation

To test the generalisation capability of the network, that is, its performance on a different (never seen) set of data, a small (but representative) third set might be reserved, the generalisation set


### 3.5 Neural Network Architectures

FNN: [Feed-forward network](https://en.wikipedia.org/wiki/Feedforward_neural_network)
-  connections between the nodes do not form cycles/loops
-  single-direction
-  fully connected (every node receives input from upstream nodes and delivers output to downstream nodes) = makes it prone to overfitting
-  no feedback

RNN: [Recurrent network](https://en.wikipedia.org/wiki/Recurrent_neural_network)
-  feeds outputs back into its own inputs
-  may react a stable state, or exhibit oscilations or even chaotic behaviour
-  can support short-term memory

CNN: [Convolutional Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network)
- deep learning network (de-facto standard for computer vision and image processing)
- inspired by animal visual cortex
- regularized FNN to prevent overfitting



**[Exercise: Build a Feedforward Neural Network](Feedforward-Neural-Network.md)**
