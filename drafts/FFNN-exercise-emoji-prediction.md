Emoji Prediction from Sentences

Theoretical Background
The task is to create a feedforward neural network that predicts the most appropriate emoji for a given sentence. The model will use text embeddings as input and classify the sentence into one of several emoji categories. By doing so, we will teach the neural network to understand sentiment or context from text, a key aspect of natural language processing (NLP).

The architecture is a simple feedforward network with an input layer, one or two hidden layers, and an output layer representing different emojis.

Setup
Dataset: You can create your own dataset by labeling sentences with emojis (e.g., happy, sad, angry) or use an existing emoji prediction dataset from sources like Kaggle.

Input and Target Vectors:

Input: Sentences in textual form, converted into embeddings using techniques like TF-IDF or pre-trained word embeddings (e.g., GloVe).
Target: Corresponding emoji labels, such as 😊 (happy), 😢 (sad), 😡 (angry), etc.
Feedforward Neural Network Architecture:

Input Layer: Text input (embedding of the sentence).
Hidden Layers: 1–2 layers with 64–128 neurons, ReLU activation.
Output Layer: Multiple neurons (softmax activation) corresponding to the number of emoji categories.
Loss Function: Cross-entropy (for multiclass classification).
Model Parameters:

Learning rate: 0.01 (for initial training).
Number of epochs: 500 (for initial training).
Optimizer: Stochastic Gradient Descent (SGD) or Adam.
Experiments
Preprocessing:

Convert sentences into numerical representations using TF-IDF or word embeddings.
Split the dataset into training and test sets (80-20 split is recommended).
Training the Model:

Train the network on labeled data (sentence and emoji pairs).
Use backpropagation and the cross-entropy loss function to adjust weights during training.
Evaluation:

Measure the accuracy of the model on the test set.
Compare predicted emojis with actual labels.
Tasks:
Task 1: Data Visualization:

Plot the sentence embedding vectors. Optionally, reduce dimensionality with techniques like PCA (Principal Component Analysis) to visualize the data distribution.
Task 2: Training Visualization:

Plot the training loss (cross-entropy) over each epoch to observe how the model improves.
Plot the model’s accuracy on the test set during training.
Task 3: Varying Hyperparameters:

Vary the learning rate and observe how it affects the training process:
Keep epochs fixed at 500, and try learning rates of 0.1, 0.01, and 0.001.
Vary the number of epochs:
Keep the learning rate fixed at 0.01, and try epochs of 50 and 1000.
Task 4: Impact of Initial Weights:

Initialize the weights with different values (e.g., small random values vs. all zeros). How does this affect the training process?
Task 5: Optimizer Comparison:

Compare the performance of different optimizers (e.g., SGD vs. Adam) in terms of training time and accuracy.
Implement the neural network using TensorFlow or Keras and compare the results with the basic implementation.
Bonus Task:
Use a pre-trained language model like BERT to obtain more contextualized embeddings for sentences. Compare the performance of using basic word embeddings (like GloVe) vs. BERT embeddings for emoji prediction.