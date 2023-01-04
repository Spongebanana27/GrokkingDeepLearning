
import numpy as np

def neuralNetwork(input, weights):
    return input.dot(weights)

toes = np.array([8.5, 9.5, 9.9, 9.0])
wlrec = np.array([.65, .8, .8, .9])
nfans = np.array([1.2, 1.3, .5, 1.0])

hurt = np.array([.1, 0, 0, .1])
winOrLose = np.array([1, 1, 0, 1])
sad = np.array([.1, 0.0, .1, .2])

alpha = .01
weights = np.array([[.1, .1, -.3], [.1, .2, 0], [0.0, 1.3, .1]]).T

trainingData = np.array([toes[0], wlrec[0], nfans[0]])
goalPrediction = np.array([hurt[0], winOrLose[0], sad[0]])    

for i in range(50000):

    pred = neuralNetwork(trainingData, weights)
    squaredError = (pred - goalPrediction) ** 2
    rawError = pred - goalPrediction
    weightDeltas = rawError * weights

    print("Pred: " + str(pred))
    print("Goal Pred: " + str(goalPrediction))
    print("Raw Error: " + str(rawError))
    print("Weights: " + str(weights))
    print("Weight Deltas: " + str(weightDeltas))
    print()

    weights -= alpha * weightDeltas * trainingData

