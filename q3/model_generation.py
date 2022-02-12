from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
from mappings import f1, f2
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

dataPoints_i = [10, 40, 80, 200]
hiddenNodes_j = [2, 10, 40, 100]

# generates data and target values for the two functions
def generate_data():
    data_dict_f1 = {}
    data_dict_f2 = {}
    for dataSize in dataPoints_i:
        f1_input_data_list = []
        f2_input_data_list = []
        f1_target_data_list = []
        f2_target_data_list = []
        for point in range(dataSize):
            f1_x = random.uniform(-1, 1)
            f2_x = random.uniform(-2, 2)
            f1_target = f1(f1_x)
            f2_target = f2(f2_x)
            f1_input_data_list.append(f1_x)
            f2_input_data_list.append(f2_x)
            f1_target_data_list.append(f1_target)
            f2_target_data_list.append(f2_target)
        
        x1_train, x1_test, y1_train, y1_test = train_test_split(f1_input_data_list, f1_target_data_list, test_size=0.2, random_state=42)
        x2_train, x2_test, y2_train, y2_test = train_test_split(f2_input_data_list, f2_target_data_list, test_size=0.2, random_state=43)

        data_dict_f1[str(dataSize)] = {
            "inputs" : {
                "train" : np.array(x1_train),
                "test" : np.array(x1_test)
            },
            "targets" : {
                "train" : np.array(y1_train),
                "test" : np.array(y1_test)
            }
        }
        data_dict_f2[str(dataSize)] = {
            "inputs" : {
                "train" : np.array(x2_train),
                "test" : np.array(x2_test)
            },
            "targets" : {
                "train" : np.array(y2_train),
                "test" : np.array(y2_test)
            }
        }
    return data_dict_f1, data_dict_f2

if __name__ == "__main__":
    f1_data, f2_data = generate_data()

    f1_training_errors = np.zeros((4,4))
    f1_val_errors = np.zeros((4,4))
    f2_training_errors = np.zeros((4,4))
    f2_val_errors = np.zeros((4,4))

    for data_idx, dataSize in enumerate(dataPoints_i):
        f1_data_inputs = f1_data[str(dataSize)]["inputs"]
        f1_data_targets = f1_data[str(dataSize)]["targets"]
        f2_data_inputs = f2_data[str(dataSize)]["inputs"]
        f2_data_targets = f2_data[str(dataSize)]["targets"]
        cv_splits = 10
        if dataSize == 10:
            cv_splits = 8
        for hiddenSize_idx, hiddenSize in enumerate(hiddenNodes_j):
            avg_model_training_err1 = 0
            avg_model_val_err1 = 0
            avg_model_training_err2 = 0
            avg_model_val_err2 = 0
            for repeat in range(5):
                rkf = KFold(n_splits=cv_splits, shuffle=True, random_state=2652124)
                avg_val_err_1 = 0
                avg_training_err_1 = 0
                avg_val_err_2 = 0
                avg_training_err_2 = 0
                for train_index, validation_index in rkf.split(f1_data_inputs["train"]):
                    x1_train = f1_data_inputs["train"][train_index]
                    y1_train = f1_data_targets["train"][train_index]
                    x1_val = f1_data_inputs["train"][validation_index]
                    y1_val = f1_data_targets["train"][validation_index]

                    x2_train = f2_data_inputs["train"][train_index]
                    y2_train = f2_data_targets["train"][train_index]
                    x2_val = f2_data_inputs["train"][validation_index]
                    y2_val = f2_data_targets["train"][validation_index]

                    opt1 = tf.keras.optimizers.Adam(learning_rate=0.7)
                    opt2 = tf.keras.optimizers.Adam(learning_rate=0.5)
                    # model 1 for function 1
                    model1 = tf.keras.Sequential()
                    model1.add(tf.keras.layers.Dense(hiddenSize, activation='sigmoid', input_shape=(1,), name='hidden_layer', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.), bias_initializer=tf.keras.initializers.Zeros()))
                    model1.add(tf.keras.layers.Dense(1, name='output_layer', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.), bias_initializer=tf.keras.initializers.Zeros()))
                    model1.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=opt1, metrics=['mse'])

                    # model 2 for function 2
                    model2 = tf.keras.Sequential()
                    model2.add(tf.keras.layers.Dense(hiddenSize, activation='sigmoid', input_shape=(1,), name='hidden_layer', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.), bias_initializer=tf.keras.initializers.Zeros()))
                    model2.add(tf.keras.layers.Dense(1, name='output_layer', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.), bias_initializer=tf.keras.initializers.Zeros()))
                    model2.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=opt2, metrics=['mse'])

                    es1 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 50, mode='min', verbose=1)
                    es2 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 50, mode='min', verbose=1)
                    history1 = model1.fit(x1_train, y1_train, validation_data=(x1_val, y1_val), epochs=1000, callbacks=[es1])
                    history2 = model2.fit(x2_train, y2_train, validation_data=(x2_val, y2_val), epochs=1000, callbacks=[es2])

                    avg_training_err_1 += history1.history['mse'][-1]
                    avg_val_err_1 += history1.history['val_mse'][-1]

                    avg_training_err_2 += history2.history['mse'][-1]
                    avg_val_err_2 += history2.history['val_mse'][-1]
                    # plt.plot(history2.history['val_loss'])
                    # plt.show()
                    # y_vals = model1.predict(x1_train)
                    # plt.scatter(x1_train, y_vals)
                    # plt.show()

                avg_training_err_1 = avg_training_err_1/cv_splits
                avg_val_err_1 = avg_val_err_1/cv_splits
                avg_model_training_err1 += avg_training_err_1
                avg_model_val_err1 += avg_val_err_1

                avg_training_err_2 = avg_training_err_2/cv_splits
                avg_val_err_2 = avg_val_err_2/cv_splits
                avg_model_training_err2 += avg_training_err_2
                avg_model_val_err2 += avg_val_err_2

            avg_model_training_err1 = avg_model_training_err1/5
            avg_model_val_err1 = avg_model_val_err1/5
            f1_training_errors[data_idx][hiddenSize_idx] = avg_model_training_err1
            f1_val_errors[data_idx][hiddenSize_idx] = avg_model_val_err1

            avg_model_training_err2 = avg_model_training_err2/5
            avg_model_val_err2 = avg_model_val_err2/5
            f2_training_errors[data_idx][hiddenSize_idx] = avg_model_training_err2
            f2_val_errors[data_idx][hiddenSize_idx] = avg_model_val_err2

    print(f1_training_errors, '\n')
    print(f1_val_errors, '\n')
    print(f2_training_errors, '\n')
    print(f2_val_errors, '\n')

    