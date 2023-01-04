
import numpy as np

# Sets negative values to 0, else returns x
def relu(x):
    return(x>0) * x

# Sets negative values to 0, else returns 1
def relu2deriv(x):
    return x > 0

iterations = 100000
alpha = .001
hidSize1 = 4
hidSize2 = 4

dataSet = np.array([[1, 0, 1],[0, 1, 2],[0, 0, 1],[1, 1, 1],[0, 1, 1],[1, 0, 1]])
goalSet = np.array([[1, 4, 1],[3, 0, 1],[1, 17, 1],[1, 2, 1],[0, 0, 1],[1, 0, 2]])

inSize = dataSet[0].size
outSize = goalSet[0].size

weights0To1 = np.random.random((inSize, hidSize1))  # INxH1
weights1To2 = np.random.random((hidSize1, hidSize2))  # H1xH2
weights2To3 = np.random.random((hidSize2, outSize))  # H2xOUT

for iteration in range(iterations):

    SSE = 0

    for i in range(outSize):

        # PREDICT
        layer0 = dataSet[i:i+1] #1xIN
        layer1 = relu(layer0.dot(weights0To1)) # 1xH1
        layer2 = relu(layer1.dot(weights1To2)) # 1xH2
        layer3 = layer2.dot(weights2To3) # 1xOUT

        SSE += np.sum((layer3 - goalSet[i]) ** 2)

        # COMPARE
        delta2To3 = layer3 - goalSet[i:i+1] # 1xOUT
        delta1To2 = delta2To3.dot(weights2To3.T)*relu2deriv(layer2) # 1xH2
        delta0To1 = delta1To2.dot(weights1To2.T)*relu2deriv(layer1) # 1xH1

        # LEARN
        weights2To3 -= alpha * (layer2.T.dot(delta2To3)) # H2xOUT
        weights1To2 -= alpha * (layer1.T.dot(delta1To2)) # H1xH2
        weights0To1 -= alpha * (layer0.T.dot(delta0To1)) # INxH1

    if(iteration % 1000 == 5):
        print("Error: " + str(SSE))
        # print("Prediction: " + str(layer3))
        # print("Goal: " + str(goalSet[i:i+1]))
        # print("2To3: " + str(weights2To3))
        # print("1To2: " + str(weights1To2))
        # print("0To1: " + str(weights0To1))
