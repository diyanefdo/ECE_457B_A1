from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
from mappings import f1, f2
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from model_generation import generate_data



if __name__ == "__main__":
    f1_data, f2_data = generate_data()

    opt1 = tf.keras.optimizers.Adam(learning_rate=0.9)
    model1 = tf.keras.Sequential()
    model1.add(tf.keras.layers.Dense(100, activation='sigmoid', input_shape=(1,), name='hidden_layer', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.), bias_initializer=tf.keras.initializers.Zeros()))
    model1.add(tf.keras.layers.Dense(1, name='output_layer', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.), bias_initializer=tf.keras.initializers.Zeros()))
    model1.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=opt1, metrics=['mse'])

    es1 = tf.keras.callbacks.EarlyStopping(monitor='loss', patience = 50, mode='min', verbose=1)
    history1 = model1.fit(f1_data["200"]["inputs"]["train"], f1_data["200"]["targets"]["train"], epochs=1000, callbacks=[es1])
    plt.plot(history1.history['loss'])
    plt.show()

    originalX1 = np.linspace(-1, 1, 200)
    originalY1 = []
    for xVal in originalX1:
        originalY1.append(f1(xVal))

    plt.scatter(originalX1, originalY1)
    plt.show()

    preditedY1 = model1.predict(f1_data["200"]["inputs"]["test"])
    plt.scatter(f1_data["200"]["inputs"]["test"], preditedY1)
    plt.show()

    dataPoints = 80
    opt2 = tf.keras.optimizers.Adam(learning_rate=0.5)
    model2 = tf.keras.Sequential()
    model2.add(tf.keras.layers.Dense(100, activation='sigmoid', input_shape=(1,), name='hidden_layer', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.), bias_initializer=tf.keras.initializers.Zeros()))
    model2.add(tf.keras.layers.Dense(1, name='output_layer', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.), bias_initializer=tf.keras.initializers.Zeros()))
    model2.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=opt2, metrics=['mse'])

    es2 = tf.keras.callbacks.EarlyStopping(monitor='loss', patience = 50, mode='min', verbose=1)
    history2 = model2.fit(f2_data[str(dataPoints)]["inputs"]["train"], f2_data[str(dataPoints)]["targets"]["train"], epochs=1000, callbacks=[es2])
    plt.plot(history2.history['loss'])
    plt.show()

    originalX2 = np.linspace(-2, 2, 200)
    originalY2 = []
    for xVal in originalX2:
        originalY2.append(f2(xVal))

    plt.scatter(originalX2, originalY2)
    plt.show()

    preditedY2 = model2.predict(f2_data[str(dataPoints)]["inputs"]["test"])
    plt.scatter(f2_data[str(dataPoints)]["inputs"]["test"], preditedY2)
    plt.show()
