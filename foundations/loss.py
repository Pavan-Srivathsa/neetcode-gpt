import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        numerator = np.exp((z - np.max(z)))
        denominator = np.sum(np.exp(z - np.max(z)))
        your_answer = numerator/denominator
        return np.round(your_answer, 4)
        # pass
