# import
import numpy as np

def neuralNetwork(input, weights):
    return input.dot(weights)

toes = np.array([8.5, 9.5, 9.9, 9.0])
wlrec = np.array([.65, .8, .8, .9])
nfans = np.array([1.2, 1.3, .5, 1.0])

hurt = np.array([.1, 0, 0, .1])
winOrLose = np.array([1, 1, 0, 1])
sad = np.array([.1, 0.0, .1, .2])

alpha = .1
inToHidWeights = np.array([[.1, .1, -.3], [.1, .2, 0], [0.0, 1.3, .1]]).T
hidToOutWeights = np.array([[.1, .1, -.3], [.1, .2, 0], [0.0, 1.3, .1]]).T

trainingData = np.array([toes, wlrec, nfans])
goalPrediction = np.array([hurt, winOrLose, sad])    

weights = [inToHidWeights, hidToOutWeights]

iterations = 200000

for i in range(iterations):

    for j in range(np.size(toes)):

        data = np.array([trainingData[0][j], trainingData[1][j], trainingData[2][j]])
        goal = np.array([goalPrediction[0][j], goalPrediction[1][j], goalPrediction[2][j]])

        hid = neuralNetwork(data, weights[0])
        pred  = neuralNetwork(hid, weights[1])
        squaredError = (pred - goal) ** 2
        rawError = pred - goal
        inToHidWeightDeltas = rawError * weights[0]
        hidToOutWeightDeltas = rawError * weights[1]
        weightDeltas = [inToHidWeightDeltas, hidToOutWeightDeltas]

        weights[0] -= alpha * weightDeltas[0] * data
        weights[1] -= alpha * weightDeltas[1] * goal
    
    if(i % (iterations / 10) == 0):
        print(i / iterations)



for x in range(np.size(toes)):

    hid = neuralNetwork(np.array([trainingData[0][x], trainingData[1][x], trainingData[2][x]]), weights[0])
    pred  = neuralNetwork(hid, weights[1])

    rawError = pred - np.array([goalPrediction[0][x], goalPrediction[1][x], goalPrediction[2][x]])

    print(x)
    #print("Pred: " + str(pred))
    #print("Goal Pred: " + str(np.array([goalPrediction[0][x], goalPrediction[1][x], goalPrediction[2][x]])))
    print("Raw Error: " + str(rawError))
    print()


