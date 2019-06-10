import numpy as np

streetlights = np.array([[1, 0, 1],
                         [0, 1, 1],
                         [0, 0, 1],
                         [1, 1, 1],
                         [0, 1, 1],
                         [1, 0, 1]])

walk_vs_stop = np.array([[0],
                         [1],
                         [0],
                         [1],
                         [1],
                         [0]])

weights = np.array([0.5, 0.48, -0.7])
alpha = 0.1

for _ in range(40):
    error_for_all_lights = 0
    for row_index in range(len(walk_vs_stop)):
        input = streetlights[row_index]
        goal_prediction = walk_vs_stop[row_index]

        prediction = input.dot(weights)
        error = (prediction - goal_prediction) ** 2
        error_for_all_lights += error
        delta = prediction - goal_prediction
        weights = weights - alpha * input * delta

        # print('Prediction: {}'.format(prediction))
    print('Error: {}\n'.format(error_for_all_lights))

print(weights)