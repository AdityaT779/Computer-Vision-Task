# Computer-Vision-Task

Overview :

This project involves building a neural network model to classify handwritten digits from the MNIST grayscale image dataset. The focus was on understanding data loading, preprocessing, model architecture design, training, and result interpretation.

Dataset Understanding :

1. The dataset consists of 70,000 grayscale images of handwritten digits (0-9) sized 28x28 pixels.

2. The dataset is split into 60,000 images for training and 10,000 images for testing.

3. Pixel values range from 0 to 255 and represent grayscale intensity.


Preprocessing :

1. Pixel intensities were scaled to the range by dividing by 255 to normalize the input features.

2. No missing values or labels needed treatment in this well-prepared dataset.

3. The labels represent digit classes and require no encoding since they are already integers.


Model Building :

1. A simple Sequential neural network was constructed with:

2. An input layer accepting 28x28 images.

3. A Flatten layer to convert 2D images to 1D vectors for dense layers.

4. A Dense layer with 128 neurons and ReLU activation for learning complex representations.

5. A Dropout layer to randomly disable 20% of neurons during training to reduce overfitting.

6. A final Dense layer outputting logits for 10 classes (digits 0-9).

7. The loss function used was Sparse Categorical Crossentropy suitable for integer labels and logits.

8. The Adam optimizer was chosen for efficient weight adjustment during training.


Training :

1. The model was trained for 5 epochs on the training set with batch size defaulted by TensorFlow.

2. Training accuracy progressively improved from about 86% to over 97%.

3. Loss steadily decreased indicating effective model learning.


Evaluation and Results :

1. On the test set, the model achieved an accuracy of approximately 97.7%.

2. Sample predictions produce probability distributions over the 10 digit classes, showing high confidence for correct classifications.

3. The results demonstrate the model's strong ability to generalize to unseen handwritten digits.


Summary

This project successfully implements a basic yet powerful digit classification pipeline using TensorFlow. It covers important ML concepts including data preprocessing, neural network modeling, and accuracy evaluation, providing a solid foundation for further experimentation with image classification tasks.
