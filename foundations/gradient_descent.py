class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        
        while iterations != 0:
            init -= learning_rate * 2 * init
            iterations -= 1
        
        return round(init, 5)