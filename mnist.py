import tensorflow as tf

mnist = tf.keras.datasets.mnist #MNIST is grayscale dataset consisting of numbers

#Scaling down pixels to a lower range [0,1]
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#Create model workflow
model = tf.keras.models.Sequential([
  tf.keras.layers.Input(shape=(28, 28)), #since its greyscale we only need to mention height and width in shape
  tf.keras.layers.Flatten(), #converts the 2D image to 1D for purpose of computing through dense layer
  tf.keras.layers.Dense(128, activation='relu'), #creates network of neural layers with 128 neurons(more neurons=more learning capacity).
                                                 #uses relu activation.
                                                 #dense layer means every neuron is connected to each neuron of previous layer. 
                                                 #good for learning global relationships.
  tf.keras.layers.Dropout(0.2), #drops 20% neurons for training set to avoid overfitting
  tf.keras.layers.Dense(10) #creates final output layer which outputs 10 logits for classifying each image
])

#Defining loss/error function
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) #Logits are not probabilities. They can be any real number (positive or negative). They are 
                                                                          #used to compare how strongly the model believes each class is the correct one, like confidence scores
#Compiling model - prepares model for training
model.compile(optimizer='adam', #adam is a smart optimizer to adjust weights in training
              loss=loss_fn, 
              metrics=['accuracy'])

#Training model
model.fit(x_train, y_train, epochs=5)#epochs is no. of times the model computes the whole training set

#Evaluating model
model.evaluate(x_test,  y_test, verbose=2) #verbose controls how much information is printed out during evaluation
                                          #evaluate prints the loss and accuracy of model on unseen data

#seeing probabilities of predicted numbers 
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

preds = probability_model(x_test[:5]).numpy()
print(preds)
