
weight = 0.0
goalPred = .8
input = .5
pred = 0.0
alpha = .5

for i in range(100):

    pred = input * weight
    rawError = pred - goalPred
    error = (pred - goalPred) ** 2

    # Utilize raw error to calculate the necessary change for the weight
    # Multiply weightDelta by input to scale, stop, and preform negative reversal
    # Multiply weightDelta by alpha to avoid divergence
    weightDelta = rawError * input * alpha
    weight = weight - weightDelta

    print("Error: " + str(error) + " Prediction: " + str(pred))