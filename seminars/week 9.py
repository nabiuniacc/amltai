from keras.src.metrics.accuracy_metrics import accuracy
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import SGD
from numpy import mean, std

#loadinf mnist dataset
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
#load dataset
(trainX, trainY), (testX, testY) = mnist.load_data()
#summarise
print('Train: X=%s, Y=%s' % (trainX.shape, trainY.shape))
print('Test: X=%s, Y=%s' % (testX.shape, testY.shape))
#plot first few images
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(trainX[i], cmap=plt.get_cmap('gray'))
plt.show()

#preprocess data to develop a baseline model
(trainX, trainY), (testX, testY) = mnist.load_data()
#reshape dataset to hav a single channel
trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))

from tensorflow.keras.utils import to_categorical
#one hot encode target values
trainY = to_categorical(trainY)
testY = to_categorical(testY)

#load train and test datast
def load_dataset():
    (trainX, trainY), (testX, testY) = mnist.load_data()
    # reshape dataset to hav a single channel
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # one hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY

#prepare pixel data
#scale pixels
def prep_pixels(train, test):
    #int to float
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    #normalise ti range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    return train_norm, test_norm

#define model
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    #compile model
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

#eval model using kfold cross validation
def evaluate_model(dataX, dataY, n_folds = 5):
    scores, histories = list(), list()
    #prep crossval
    kfold = KFold(n_folds, shuffle =True, random_state=1)
    #enumerate splits
    for train_ix, test_ix in kfold.split(dataX):
        #define model
        model = define_model()
        #slect rows for train and test
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        #fit model
        history = model.fit(trainX, trainY, epochs = 10, batch_size = 32, validation_data = (testX, testY), verbose = 0)
        #eval model
        _, acc = model.evaluate(testX, testY, verbose=0)
        print('> %.3f' % (acc * 100.0))
        #store scores
        scores.append(acc)
        histories.append(history)

    return scores, histories

#present results
#plot diagnostic learning curves
def summarize_diagnostics(histories):
    for i in range(len(histories)):
        #plot loss
        plt.subplot(2, 1, 1)
        plt.title('Cross Entropy Loss')
        plt.plot(histories[i].history['loss'], color='blue', label='train')
        plt.plot(histories[i].history['val_loss'], color='red', label='test')
        #plot acuracys
        plt.subplot(2, 1, 2)
        plt.title('Classification Accuracy')
        plt.plot(histories[i].history['accuracy'], color='blue', label='train')
        plt.plot(histories[i].history['val_accuracy'], color='red', label='test')
        plt.show()

#summarize model performance
def summarize_performance(scores):
    print('Accuracy: mean=%.3f  std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
    plt.boxplot(scores)
    plt.show()

def run_test_harness():
    trainX, trainY, testX, testY = load_dataset()
    trainX, testX = prep_pixels(trainX, testX)
    scores, histories = evaluate_model(trainX, trainY)
    summarize_diagnostics(histories)
    summarize_performance(scores)

run_test_harness()