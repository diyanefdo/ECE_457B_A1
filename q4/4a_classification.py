from cgi import test
from gc import callbacks
import numpy as np
import tensorflow.keras as keras
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def readData():
    with open('randomized_data.txt') as f:
        lines = f.readlines()
        input_feature_list = []
        output_label_list = []
        for line in lines:
            line = line.strip('\n')
            features = line.split(",")
            feature_entry = []
            for idx, feature in enumerate(features):
                if idx == 0:
                    output_label_list.append(int(feature))
                else:
                    feature_entry.append(float(feature))
            input_feature_list.append(feature_entry)
        
        input_features = np.array(input_feature_list)
        output_labels = np.array(output_label_list)

        return input_features, output_labels

# Run function to iterate through different combinations of hidden layers and number of nodes
def modelExperimentation():
    hidden_layers = [1, 2, 3, 4]
    num_nodes = [1, 2, 3, 4]

    input_features, output_labels = readData()
    output_labels = keras.utils.to_categorical(output_labels)

    input_features = minmax_scale(input_features)
    x_train, x_test, y_train, y_test = train_test_split(input_features, output_labels, test_size=0.25, shuffle=True)

    test_accuracies = np.zeros((4, 4))

    for idx1, hidden_size in enumerate(hidden_layers):
        for idx2, nodes in enumerate(num_nodes):
            model = keras.Sequential()
            model.add(keras.Input(shape=(13,)))
            for layer in range(hidden_size):
                initial_weights = keras.initializers.RandomNormal(mean=0., stddev=1.)
                initial_biases = keras.initializers.Zeros()
                model.add(keras.layers.Dense(nodes, activation='sigmoid', kernel_initializer=initial_weights, bias_initializer=initial_biases))
            model.add(keras.layers.Dense(4, activation='softmax', kernel_initializer=keras.initializers.RandomNormal(mean=0., stddev=1.), bias_initializer=keras.initializers.Zeros()))
            es = keras.callbacks.EarlyStopping(monitor='val_loss', patience = 50, mode='min', verbose=1)
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.07), loss='categorical_crossentropy', metrics=['accuracy'])
            history = model.fit(x_train, y_train, epochs=3000, validation_split=0.1, callbacks=[es])
            test_accuracy = model.evaluate(x_test, y_test, verbose=0)
            print(test_accuracy[1])
            test_accuracies[idx1][idx2] = test_accuracy[1]
    print(test_accuracies)

def newDataClassification():
    input_features, output_labels = readData()
    output_labels = keras.utils.to_categorical(output_labels)
    input_features = minmax_scale(input_features)
    x_train, x_test, y_train, y_test = train_test_split(input_features, output_labels, test_size=0.25, shuffle=True)

    newDataToClassify = np.array([[13.72, 1.43, 2.5, 16.7, 108, 3.4, 3.67, 0.19, 2.04, 6.8, 0.89, 2.87, 1285], 
        [12.04, 4.3, 2.38, 22, 80, 2.1, 1.75, 0.42, 1.35, 2.6, 0.79, 2.57, 580], 
        [14.13, 4.1, 2.74, 24.5, 96, 2.05, 0.76, 0.56, 1.35, 9.2, 0.61, 1.6, 560]], dtype=float)
    newDataToClassify = minmax_scale(newDataToClassify)

    model = keras.Sequential()
    initial_weights = keras.initializers.RandomNormal(mean=0., stddev=1.)
    initial_biases = keras.initializers.Zeros()
    model.add(keras.layers.Dense(4, activation='sigmoid', input_shape=(13,), kernel_initializer=initial_weights, bias_initializer=initial_biases))
    model.add(keras.layers.Dense(4, activation='sigmoid', kernel_initializer=initial_weights, bias_initializer=initial_biases))
    model.add(keras.layers.Dense(4, activation='sigmoid', kernel_initializer=initial_weights, bias_initializer=initial_biases))
    model.add(keras.layers.Dense(4, activation='sigmoid', kernel_initializer=initial_weights, bias_initializer=initial_biases))
    model.add(keras.layers.Dense(4, activation='softmax', kernel_initializer=keras.initializers.RandomNormal(mean=0., stddev=1.), bias_initializer=keras.initializers.Zeros()))
    es = keras.callbacks.EarlyStopping(monitor='val_loss', patience = 50, mode='min', verbose=1)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.07), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=3000, validation_split=0.1, callbacks=[es])
    
    # Perform prediction
    predictions = model.predict(newDataToClassify)
    print(predictions, '\n')
    print(np.rint(predictions))
    

if __name__ == "__main__":
    newDataClassification()