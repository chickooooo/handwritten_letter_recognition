import numpy as np
import scripts.utils as utils

# setup the model and other dependencies
setup = utils.Setup()

# data to test on
# letters: X, D, S
test_data = np.load('./data/sample_array.npy')

# make predictions
result = setup.predict(test_data)
# print result
print(result)
