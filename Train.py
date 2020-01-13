import keras
import time
from keras.datasets import cifar10
from keras.models import Sequential
from keras import optimizers
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from pathlib import Path
from matplotlib import pyplot as plt

# Loading CIFAR-10 data sets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalizing data set to 0-to-1 range
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Setting Hyperparameters
batchSize = 64
epoc = 100

# Converting class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Defining Numerical Optimizers
sgd = optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.0, nesterov=False)
rmsp = optimizers.RMSprop(learning_rate=0.001, rho=0.9)
adag = optimizers.Adagrad(learning_rate=0.01)
adam = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

# Create a model and add layers
model_adam = Sequential()

model_adam.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3), activation="relu"))
model_adam.add(Conv2D(32, (3, 3), activation="relu"))
model_adam.add(MaxPooling2D(pool_size=(2, 2)))
model_adam.add(Dropout(0.25))

model_adam.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
model_adam.add(Conv2D(64, (3, 3), activation="relu"))
model_adam.add(MaxPooling2D(pool_size=(2, 2)))
model_adam.add(Dropout(0.25))

model_adam.add(Flatten())
model_adam.add(Dense(512, activation="relu"))
model_adam.add(Dropout(0.5))
model_adam.add(Dense(10, activation="softmax"))

# Compile the model
model_adam.compile(
    loss='categorical_crossentropy',
    optimizer=adam,
    metrics=['accuracy']
)

start_time = time.time()

# Train the model
history = model_adam.fit(
    x_train,
    y_train,
    batch_size=batchSize,
    epochs=epoc,
    validation_data=(x_test, y_test),
    shuffle=True
)

elapsed_time = time.time() - start_time

# Plot for model accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('adam_acc.png', dpi=600)
plt.show()

# Plot for model loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('adam_loss.png', dpi=600)
plt.show()

# Save total training time
print('Execution time: %.3f' % elapsed_time)
file1 = Path("adam_time.txt")
file1.write_text(str(elapsed_time))

# Save neural network structure
model_structure = model_adam.to_json()
f = Path("adam_model_structure.json")
f.write_text(model_structure)

# Save neural network's trained weights
model_adam.save_weights("adam_model_weight.h5")

time.sleep(1000)

# Create a model and add layers
model_sgd = Sequential()

model_sgd.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3), activation="relu"))
model_sgd.add(Conv2D(32, (3, 3), activation="relu"))
model_sgd.add(MaxPooling2D(pool_size=(2, 2)))
model_sgd.add(Dropout(0.25))

model_sgd.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
model_sgd.add(Conv2D(64, (3, 3), activation="relu"))
model_sgd.add(MaxPooling2D(pool_size=(2, 2)))
model_sgd.add(Dropout(0.25))

model_sgd.add(Flatten())
model_sgd.add(Dense(512, activation="relu"))
model_sgd.add(Dropout(0.5))
model_sgd.add(Dense(10, activation="softmax"))

# Compile the model
model_sgd.compile(
    loss='categorical_crossentropy',
    optimizer=sgd,
    metrics=['accuracy']
)

start_time = time.time()

# Train the model
history = model_sgd.fit(
    x_train,
    y_train,
    batch_size=batchSize,
    epochs=epoc,
    validation_data=(x_test, y_test),
    shuffle=True
)

elapsed_time = time.time() - start_time

# Plot for model accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('sgd_acc.png', dpi=600)
plt.show()

# Plot for model loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('sgd_loss.png', dpi=600)
plt.show()

# Save total training time
print('Execution time: %.3f' % elapsed_time)
file1 = Path("sgd_time.txt")
file1.write_text(str(elapsed_time))

# Save neural network structure
model_structure = model_sgd.to_json()
f = Path("sgd_model_structure.json")
f.write_text(model_structure)

# Save neural network's trained weights
model_sgd.save_weights("sgd_model_weight.h5")

time.sleep(1000)

# Create a model and add layers
model_rmsp = Sequential()

model_rmsp.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3), activation="relu"))
model_rmsp.add(Conv2D(32, (3, 3), activation="relu"))
model_rmsp.add(MaxPooling2D(pool_size=(2, 2)))
model_rmsp.add(Dropout(0.25))

model_rmsp.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
model_rmsp.add(Conv2D(64, (3, 3), activation="relu"))
model_rmsp.add(MaxPooling2D(pool_size=(2, 2)))
model_rmsp.add(Dropout(0.25))

model_rmsp.add(Flatten())
model_rmsp.add(Dense(512, activation="relu"))
model_rmsp.add(Dropout(0.5))
model_rmsp.add(Dense(10, activation="softmax"))

# Compile the model
model_rmsp.compile(
    loss='categorical_crossentropy',
    optimizer=rmsp,
    metrics=['accuracy']
)

start_time = time.time()

# Train the model
history = model_rmsp.fit(
    x_train,
    y_train,
    batch_size=batchSize,
    epochs=epoc,
    validation_data=(x_test, y_test),
    shuffle=True
)

elapsed_time = time.time() - start_time

# Plot for model accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('rmsp_acc.png', dpi=600)
plt.show()

# Plot for model loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('rmsp_loss.png', dpi=600)
plt.show()

# Save total training time
print('Execution time: %.3f' % elapsed_time)
file1 = Path("rmsp_time.txt")
file1.write_text(str(elapsed_time))

# Save neural network structure
model_structure = model_rmsp.to_json()
f = Path("rmsp_model_structure.json")
f.write_text(model_structure)

# Save neural network's trained weights
model_rmsp.save_weights("rmsp_model_weight.h5")

time.sleep(1000)

# Create a model and add layers
model_adag = Sequential()

model_adag.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3), activation="relu"))
model_adag.add(Conv2D(32, (3, 3), activation="relu"))
model_adag.add(MaxPooling2D(pool_size=(2, 2)))
model_adag.add(Dropout(0.25))

model_adag.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
model_adag.add(Conv2D(64, (3, 3), activation="relu"))
model_adag.add(MaxPooling2D(pool_size=(2, 2)))
model_adag.add(Dropout(0.25))

model_adag.add(Flatten())
model_adag.add(Dense(512, activation="relu"))
model_adag.add(Dropout(0.5))
model_adag.add(Dense(10, activation="softmax"))

# Compile the model
model_adag.compile(
    loss='categorical_crossentropy',
    optimizer=adag,
    metrics=['accuracy']
)

start_time = time.time()

# Train the model
history = model_adag.fit(
    x_train,
    y_train,
    batch_size=batchSize,
    epochs=epoc,
    validation_data=(x_test, y_test),
    shuffle=True
)

elapsed_time = time.time() - start_time

# Plot for model accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('adag_acc.png', dpi=600)
plt.show()

# Plot for model loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('adag_loss.png', dpi=600)
plt.show()

# Save total training time
print('Execution time: %.3f' % elapsed_time)
file1 = Path("adag_time.txt")
file1.write_text(str(elapsed_time))

# Save neural network structure
model_structure = model_adag.to_json()
f = Path("adag_model_structure.json")
f.write_text(model_structure)

# Save neural network's trained weights
model_adag.save_weights("adag_model_weight.h5")