import tensorflow as tf

# Download pre-trained MNIST CNN model
model = tf.keras.models.load_model(
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist_cnn.h5"
)
print("Model loaded successfully!")