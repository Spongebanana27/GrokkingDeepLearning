
import numpy as np

def neuralNetwork(input, weights):
    return input.dot(weights)

toes = np.array([8.5, 9.5, 9.9, 9.0])
wlrec = np.array([.65, .8, .8, .9])
nfans = np.array([1.2, 1.3, .5, 1.0])
winOrLose = np.array([1, 1, 0, 1])

alpha = .3
weights = np.array([.1, .2, -.1])

trainingData = np.array([toes[0], wlrec[0], nfans[0]])
goalPrediction = winOrLose[0]

for i in range(100):

    pred = neuralNetwork(trainingData, weights)
    squaredError = (pred - goalPrediction) ** 2
    rawError = pred - goalPrediction
    weightDeltas = rawError * weights

    #FREEZE weights[0]
    weightDeltas[0] = 0

    print("Pred: " + str(pred))
    print("Squared Error: " + str(squaredError))
    print("Weights: " + str(weights))
    print("Weight Deltas: " + str(weightDeltas))
    print()

    weights -= alpha * weightDeltas

