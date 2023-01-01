import numpy as np

# The problem: 3 lights can be on or off. Based on these, when is it safe to walk?
# Array of the recorded states the streetlight
streetlights = np.array([[1, 0, 1],[0, 1, 1],[0, 0, 1],[1, 1, 1],[0, 1, 1],[1, 0, 1]])
isSafe = np.array([0, 1, 0, 1, 1, 0])

weights = np.array([.5, .48, -.7]).T
alpha = .1

for i in range(400):
    totalError = 0
    for j in range(len(isSafe)):

        input = streetlights[j]
        goalPred = isSafe[j]
        pred = input.dot(weights)

        squaredError = (pred - goalPred) ** 2
        totalError += squaredError

        rawError = pred - goalPred
        weights = weights - (alpha * (input * rawError))

    print("Error: " + str(totalError))



