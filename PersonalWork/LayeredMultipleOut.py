import numpy as np

# Sets negative values to 0, else returns x
def relu(x):
    return(x>0) * x

# Sets negative values to 0, else returns 1
def relu2deriv(x):
    return x > 0

toes = np.array([8.5, 9.5, 9.9, 9.0])
wlrec = np.array([.65, .8, .8, .9])
nfans = np.array([1.2, 1.3, .5, 1.0])

hurt = np.array([.5, .3, 0, .9])
willWin = np.array([1, 1, 0, 1])
sad = np.array([.6, .3, .1, .2])

HIDSIZE = 4

alpha = .004

dataSet = np.array([toes, wlrec, nfans]).T
goal = np.array([hurt, willWin, sad]).T

inToHidden = np.random.random((3, HIDSIZE))  # 3xHIDSIZE
#inToHidden = np.array([[.5, .5, .5, .5], [.5, .5, .5, .5], [.5, .5, .5, .5]])
hiddenToOut = np.random.random((HIDSIZE, 3))  # HIDSIZEx3
#hiddenToOut = np.array([[.5, .5, .5], [.5, .5, .5], [.5, .5, .5], [.5, .5, .5]])

for iteration in range(50000000):

    totalError = 0

    for i in range(toes.size):

        input = dataSet[i:i+1] # 1x3
        hidden = relu(np.dot(input, inToHidden)) # 1xHIDSIZE
        out = hidden.dot(hiddenToOut) # 1x3

        totalError += np.sum((out - goal[i:i+1])) ** 2

        hiddenDelta = out - goal[i:i+1] # 1x3
        inDelta = hiddenDelta.dot(hiddenToOut.T)*relu2deriv(hidden) # 1xHIDSIZE

        hiddenToOut -= (alpha * hidden.T.dot(hiddenDelta)) # HIDSIZEx3
        inToHidden -= (alpha * input.T.dot(inDelta))  # 3xHIDSIZE

    if(iteration % 10000 == 5):
        print("Error: " + str(totalError))
