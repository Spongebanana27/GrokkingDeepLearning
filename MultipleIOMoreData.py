
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

trainingData = np.array([toes, wlrec, nfans])
goalPrediction = np.array([hurt, winOrLose, sad])    

for i in range(50000):

    for j in range(np.size(toes)):

        data = np.array([trainingData[0][j], trainingData[1][j], trainingData[2][j]])
        goal = np.array([goalPrediction[0][j], goalPrediction[1][j], goalPrediction[2][j]])

        pred = neuralNetwork(data, weights)
        squaredError = (pred - goal) ** 2
        rawError = pred - goal
        weightDeltas = rawError * weights

        weights -= alpha * weightDeltas * data

for x in range(np.size(toes)):
    print(x)
    print("Pred: " + str(neuralNetwork(np.array([trainingData[0][x], trainingData[1][x], trainingData[2][x]]), weights)))
    print("Goal Pred: " + str(np.array([goalPrediction[0][x], goalPrediction[1][x], goalPrediction[2][x]])))
    print("Raw Error: " + str(rawError))
    print()


