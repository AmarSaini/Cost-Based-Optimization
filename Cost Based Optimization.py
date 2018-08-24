import math
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras import callbacks

# ----- Global Variables -----

# Feature Size
featureSize = 0


def fit(x, a, b):
    return a/(x**b)

def loadData(fileName, dataSize, featureSize):

    myFile = open(fileName)

    # 57514 negatives, 1731 positives
    examples = np.zeros((dataSize, featureSize), float)
    labels = np.zeros(dataSize, int)

    index = 0

    for myLine in myFile:
        #print(lineNum)
        myPairs = myLine.split(" ")

        if(myPairs[0] == "+1"):
            labels[index] = 1

        else:
            labels[index] = -1

        myPairs.pop(0)
        myPairs.pop()
        for onePair in myPairs:
            splitPair = onePair.split(":");
            examples[index][int(splitPair[0])-1] = float(splitPair[1])
        index += 1

    myFile.close()

    return [examples, labels]

def fitGD(deltaLosses, timePerIter, desiredError):

    # Fit the error to our curve. 
    # Our curve is a function that takes in an error, and outputs the number of iterations needed to achieve that error.
    
    # popt holds the optimal fitted parameter values
    iterations = np.arange(len(deltaLosses))
    iterations += 1

    popt, pcov = curve_fit(fit, deltaLosses, iterations)

    # Get the estimated number of iterations for the desired error
    estimatedIter = int(fit(desiredError, *popt))
    estimatedTotalTime = estimatedIter*timePerIter

    # Plot the error points, and the fitted curve
    plt.plot(deltaLosses, iterations, 'ro')
    plt.plot(deltaLosses, fit(deltaLosses, *popt), 'b', label='Speculated Change In Loss: %.5f \nEstimated Iter: %d \nTime Per Iter: %.4f \nEstimated Total Time: %.2f \nfit: a=%.4f b=%.4f' % (desiredError, estimatedIter, timePerIter, estimatedTotalTime, popt[0], popt[1]))

    plt.ylabel('Iterations')
    plt.xlabel('Change in Loss')
    plt.legend()

    print('Speculated Change In Loss: {0}, Estimated Iter: {1}, Time per Iter: {2}, ETA for Convergence {3}'.format(desiredError, estimatedIter, timePerIter, estimatedTotalTime))

    return estimatedIter, estimatedTotalTime

def testKeras(examples, labels, subsetPercent = 0.2, desiredError = 0.001, timeLimit = 30):

    # Test each algorithm on a smaller dataset.

    exampleSubset = examples[0:int(len(examples)*subsetPercent)]
    labelSubset = labels[0:int(len(labels)*subsetPercent)]

    max_iterations = 10000
    estimatedIters = []

    allResults = []

    for i in range(7):

        plt.figure(i+1)

        # Create Model for Keras
        model = Sequential()
        model.add(Dense(units=1, activation='linear', input_dim=featureSize))

        # Choose GD Algorithm for Keras
        if (i == 0):
            myOpt = optimizers.SGD(lr=0.01, momentum=0., decay=0., nesterov=False)
            plt.title("SGD")
        elif (i == 1):
            myOpt = optimizers.SGD(lr=0.01, momentum=0.9, decay=0., nesterov=False)
            plt.title("Momentum")
        elif (i == 2):
            myOpt = optimizers.SGD(lr=0.01, momentum=0.9, decay=0., nesterov=True)
            plt.title("Nesterov-Momentum")
        elif (i == 3):
            myOpt = optimizers.Adagrad(lr=0.01, epsilon=1e-6)
            plt.title("Adagrad")
        elif (i == 4):
            myOpt = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-6)
            plt.title("Adadelta")
        elif (i == 5):
            myOpt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-6)
            plt.title("RMSprop")
        elif (i == 6):
            myOpt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
            plt.title("Adam")

        model.compile(optimizer=myOpt, loss=logloss)

        # Create Custom Callback. Run GD. History saved in output. Use it to find the changes in loss per iteration
        customCallback = EarlyStoppingByDeltaLossOrTime(desiredError, timeLimit)
        myCallbacks = [customCallback]
        output = model.fit(exampleSubset, labelSubset, epochs=max_iterations, batch_size=int(len(exampleSubset)/50), callbacks=myCallbacks)
        losses = np.array(output.history['loss'])
        deltaLosses = -np.diff(losses)

        # Run again on the full dataset, for a few iterations. Use this to find the average time per iteration.
        # Reset callback to reset time elapsed and history of losses.
        model = Sequential()
        model.add(Dense(units=1, activation='linear', input_dim=featureSize))
        model.compile(optimizer=myOpt, loss=logloss)
        customCallback = EarlyStoppingByDeltaLossOrTime(desiredError, timeLimit)
        myCallbacks = [customCallback]
        output = model.fit(examples, labels, epochs=5, batch_size=int(len(examples)/50), callbacks=myCallbacks)
        losses = np.array(output.history['loss'])
        timePerIter = myCallbacks[0].timeElapsed/len(losses)

        # Pass in the following:
        # 1. Array of DeltaLosses, iterations is length of array.
        # 2. Average Time per Iteration on the full dataset.
        results = fitGD(deltaLosses, timePerIter, desiredError)
        estimatedIters.append(results[0])
        print("ETI: ", results[0])
        print("ETA: ", results[1])

        allResults.append(results)

    for i in range(len(allResults)):
        print("Algo", i, "Iterations:", allResults[i][0], "ETA:", allResults[i][1])

    plt.show()

    return estimatedIters

def logloss(y_true, y_pred): # define a custom tensorflow loss function
    return tf.log(1 + tf.exp(-y_true * y_pred))

class EarlyStoppingByDeltaLossOrTime(callbacks.Callback):
    def __init__(self, deltaLoss, timeLimit, logs={}):

        self.deltaLoss = deltaLoss
        self.timeLimit = timeLimit

        self.losses = [1.0]
        self.startTime = time.time()
        self.timeElapsed = 0

    def on_epoch_end(self, batch, logs={}):

        self.losses.append(logs.get('loss'))
        lossesSize = len(self.losses)

        changeinLoss = self.losses[lossesSize-2] - self.losses[lossesSize-1]
        self.timeElapsed = time.time() - self.startTime
        print("Change in Loss: ", changeinLoss)
        print("Total Time: ", self.timeElapsed)

        if (changeinLoss <= self.deltaLoss or self.timeElapsed >= self.timeLimit):
            self.model.stop_training = True

class PrintDeltaLossAndTime(callbacks.Callback):
    def __init__(self, logs={}):

        self.losses = [1.0]
        self.startTime = time.time()
        self.timeElapsed = 0

    def on_epoch_end(self, batch, logs={}):

        self.losses.append(logs.get('loss'))
        lossesSize = len(self.losses)

        changeinLoss = self.losses[lossesSize-2] - self.losses[lossesSize-1]
        self.timeElapsed = time.time() - self.startTime
        print("Change in Loss: ", changeinLoss)
        print("Total Time: ", self.timeElapsed)


def main():

    global svrg;
    global adaGrad;

    # Set up data

    # -------- Make sure you change the featureSize variable below to the correct dimension --------

    #result = loadData('a1a.t', 30956, 123)
    result = loadData('w8a.txt', 59245, 300)
    #result = loadData('covtype.libsvm.binary.scale', 581012, 54)
    
    examples = result[0]
    labels = result[1]

    #np.save("examples.npy", examples)
    #np.save("labels.npy", labels)

    #examples = np.load('examples.npy')
    #labels = np.load('labels.npy')


    global featureSize
    featureSize = 300

    # Test Keras
    estimatedIters = testKeras(examples, labels)

    print("Choosing GD Plan...")

    while(1):

        print("0 - SGD")
        print("1 - Momentum")
        print("2 - Nesterov-Momentum")
        print("3 - Adagrad")
        print("4 - Adadelta")
        print("5 - RMSprop")
        print("6 - Adam")

        algoChoice = int(input("Choose Algo: "))

        # Create Model for Keras
        model = Sequential()
        model.add(Dense(units=1, activation='linear', input_dim=featureSize))

        # Choose GD Algorithm for Keras
        if (algoChoice == 0):
            myOpt = optimizers.SGD(lr=0.01, momentum=0., decay=0., nesterov=False)
        elif (algoChoice == 1):
            myOpt = optimizers.SGD(lr=0.01, momentum=0.9, decay=0., nesterov=False)
        elif (algoChoice == 2):
            myOpt = optimizers.SGD(lr=0.01, momentum=0.9, decay=0., nesterov=True)
        elif (algoChoice == 3):
            myOpt = optimizers.Adagrad(lr=0.01, epsilon=1e-6)
        elif (algoChoice == 4):
            myOpt = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-6)
        elif (algoChoice == 5):
            myOpt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-6)
        elif (algoChoice == 6):
            myOpt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

        model.compile(optimizer=myOpt, loss=logloss)

        # To get results, use early stop, double iter.
        customCallback = PrintDeltaLossAndTime()
        myCallbacks = [customCallback]
        output = model.fit(examples, labels, epochs=estimatedIters[algoChoice], batch_size=int(len(examples)/50), callbacks=myCallbacks)

if  __name__ == '__main__':
    main()
