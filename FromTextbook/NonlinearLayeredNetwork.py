import numpy as np

# Sets negative values to 0, else returns x
def relu(x):
    return(x>0) * x

# Sets negative values to 0, else returns 1
def relu2deriv(x):
    return x > 0

alpha = .2

HIDSIZE = 4

dataSet = np.array([[1, 0, 1],[0, 1, 1],[0, 0, 1],[1, 1, 1],[0, 1, 1],[1, 0, 1]])

goal = np.array([0, 1, 0, 1, 1, 0]).T

inToHid1 = 2*np.random.random((3, HIDSIZE)) - 1
hid1ToOut = 2*np.random.random((HIDSIZE, 1)) - 1

for iterations in range(200):
    hiddenError = 0

    for i in range(np.size(goal)):
        input = dataSet[i:i+1] # 1x3
        hidden = relu(np.dot(input, inToHid1)) # 1xHIDSIZE, any negative values set to 0
        out = hidden.dot(hid1ToOut) # 1x1

        hiddenError += np.sum((out - goal[i:i+1])) ** 2

        hiddenDelta = out - goal[i:i+1] # 1x1
        inDelta = hiddenDelta.dot(hid1ToOut.T)*relu2deriv(hidden) # 1xHIDSIZE

        hid1ToOut -= (alpha * hidden.T.dot(hiddenDelta)) # HIDSIZEx1
        inToHid1 -= alpha * input.T.dot(inDelta) # 1xHIDSIZE

    if(iterations % 10 == 5):
        print("Error: " + str(hiddenError))

