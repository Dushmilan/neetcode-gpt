import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        
        # Subtract max for numerical stability
        z_stable = z - np.max(z)
        
        # Compute exponentials
        exp_z = np.exp(z_stable)
        
        # Compute softmax
        softmax_output = exp_z / np.sum(exp_z)
        
        return np.round(softmax_output, 4)