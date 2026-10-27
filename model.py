import numpy as np

class AdvancedNeuralNetworkEngine:
    def __init__(self, layer_dimensions, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.parameters = {}
        self.num_layers = len(layer_dimensions)
        self._initialize_weights(layer_dimensions)

    def _initialize_weights(self, dimensions):
        np.random.seed(42)
        for l in range(1, len(dimensions)):
            self.parameters[f"W{l}"] = np.random.randn(dimensions[l], dimensions[l-1]) * np.sqrt(2.0 / dimensions[l-1])
            self.parameters[f"b{l}"] = np.zeros((dimensions[l], 1))

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def _sigmoid_derivative(self, a):
        return a * (1 - a)

    def forward_propagation(self, X):
        caches = {}
        A = X
        for l in range(1, self.num_layers):
            A_prev = A
            W = self.parameters[f"W{l}"]
            b = self.parameters[f"b{l}"]
            Z = np.dot(W, A_prev) + b
            A = self._sigmoid(Z)
            caches[f"A{l-1}"] = A_prev
            caches[f"Z{l}"] = Z
            caches[f"A{l}"] = A
        return A, caches

    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(AL + 1e-8) + (1 - Y) * np.log(1 - AL + 1e-8)) / m
        return np.squeeze(cost)

    def backward_propagation(self, AL, Y, caches):
        grads = {}
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)
        dAL = -(np.divide(Y, AL + 1e-8) - np.divide(1 - Y, 1 - AL + 1e-8))
        
        current_l = self.num_layers - 1
        dA_prev = dAL
        
        for l in reversed(range(1, self.num_layers)):
            dA = dA_prev
            Z = caches[f"Z{l}"]
            A_prev = caches[f"A{l-1}"]
            W = self.parameters[f"W{l}"]
            
            dZ = dA * self.sigmoid_derivative(caches[f"A{l}"])
            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            dA_prev = np.dot(W.T, dZ)
            
            grads[f"dW{l}"] = dW
            grads[f"db{l}"] = db
            
        return grads

    def update_parameters(self, grads):
        for l in range(1, self.num_layers):
            self.parameters[f"W{l}"] -= self.learning_rate * grads[f"dW{l}"]
            self.parameters[f"b{l}"] -= self.learning_rate * grads[f"db{l}"]

    def train(self, X, Y, iterations):
        for i in range(iterations):
            AL, caches = self.forward_propagation(X)
            cost = self.compute_cost(AL, Y)
            grads = self.backward_propagation(AL, Y, caches)
            self.update_parameters(grads)
            if i % 100 == 0:
                print(f"Iteration {i} | Loss: {cost:.6f}")

if __name__ == "__main__":
    X_train = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]]).T
    Y_train = np.array([[0, 1, 1, 0]])
    network = AdvancedNeuralNetworkEngine([3, 5, 1], learning_rate=0.1)
    network.train(X_train, Y_train, iterations=500)
