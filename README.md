# Computer-Vision-Task

Overview
This project involves building a neural network model to classify handwritten digits from the MNIST grayscale image dataset. The focus was on understanding data loading, preprocessing, model architecture design, training, and result interpretation.

Dataset Understanding
The dataset consists of 70,000 grayscale images of handwritten digits (0-9) sized 28x28 pixels.

The dataset is split into 60,000 images for training and 10,000 images for testing.

Pixel values range from 0 to 255 and represent grayscale intensity.

Preprocessing
Pixel intensities were scaled to the range by dividing by 255 to normalize the input features.

No missing values or labels needed treatment in this well-prepared dataset.

The labels represent digit classes and require no encoding since they are already integers.

Model Building
A simple Sequential neural network was constructed with:

An input layer accepting 28x28 images.

A Flatten layer to convert 2D images to 1D vectors for dense layers.

A Dense layer with 128 neurons and ReLU activation for learning complex representations.

A Dropout layer to randomly disable 20% of neurons during training to reduce overfitting.

A final Dense layer outputting logits for 10 classes (digits 0-9).

The loss function used was Sparse Categorical Crossentropy suitable for integer labels and logits.

The Adam optimizer was chosen for efficient weight adjustment during training.

Training
The model was trained for 5 epochs on the training set with batch size defaulted by TensorFlow.

Training accuracy progressively improved from about 86% to over 97%.

Loss steadily decreased indicating effective model learning.

Evaluation and Results
On the test set, the model achieved an accuracy of approximately 97.7%.

Sample predictions produce probability distributions over the 10 digit classes, showing high confidence for correct classifications.

The results demonstrate the model's strong ability to generalize to unseen handwritten digits.

Summary
This project successfully implements a basic yet powerful digit classification pipeline using TensorFlow. It covers important ML concepts including data preprocessing, neural network modeling, and accuracy evaluation, providing a solid foundation for further experimentation with image classification tasks.
