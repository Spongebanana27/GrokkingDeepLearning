import numpy as np

# The problem: 3 lights can be on or off. Based on these, when is it safe to walk?

# Array of the recorded states the streetlight
streetLights = np.array([[1, 0, 1],[0, 1, 1],[0, 0, 1],[1, 1, 1],[0, 1, 1],[1, 0, 1]])
walked = np.array([[0], [1], [0], [1], [1], [0]])