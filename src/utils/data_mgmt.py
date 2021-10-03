import tensorflow as tf

def get_data(validation_datasize):
     (X_train_full, y_train_full),( X_test, y_test) = tf.keras.datasets.mnist.load_data()
     X_valid = X_train_full[:validation_datasize] / 255.
     X_train = X_train_full[validation_datasize:] / 255.
     y_valid = y_train_full[:validation_datasize]
     y_train = y_train_full[validation_datasize:]
     return (X_train, y_train), (X_valid, y_valid), (X_test,y_test)