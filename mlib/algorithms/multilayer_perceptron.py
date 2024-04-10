import time
import numpy as np
from mlib.utils import shuffle, logsumexp

class MultilayerPerceptron():
    '''A simple multilayer perceptron with 1 hidden layer'''

    def __init__(self, num_input_nodes, num_output_nodes, num_hidden_nodes, mini_batch_size,
                 learning_rate=0.0001, type="class", optimization="adam", params_init="kaiming"):
        supported_types = ["class", "reg"]

        if type not in supported_types:
            raise ValueError(f"Invalid type, must be: {supported_types}")

        self.training_start_time = 0
        self.num_input_nodes = num_input_nodes
        self.num_output_nodes = num_output_nodes
        self.num_hidden_nodes = num_hidden_nodes
        self.learning_rate = learning_rate
        self.mini_batch_size = mini_batch_size
        self.type = type
        self.optimization = optimization
        self.params_init = params_init
        self.loss = []

        self.init_params()

        if optimization == "adam":
            self.init_adam_params()

    def reset(self):
        self.training_start_time = 0
        self.init_params()
        if self.optimization == "adam":
            self.init_adam_params()

    def init_params(self):
        if self.params_init == "kaiming":
            # Weights (input to hidden layer)
            self.W1 = np.random.randn(self.num_input_nodes, self.num_hidden_nodes) * np.sqrt(2. / self.num_input_nodes)
            # Biases (input to hidden layer)
            self.b1 = np.zeros((self.num_hidden_nodes))
            # Weights (hidden to output layer)
            self.W2 = np.random.randn(self.num_hidden_nodes, self.num_output_nodes) * np.sqrt(2. / self.num_hidden_nodes)
            # Biases (hidden to output layer)
            self.b2 = np.zeros((self.num_output_nodes))
        else:
            # Weights (input to hidden layer)
            self.W1 = np.random.randn(self.num_input_nodes, self.num_hidden_nodes)
            # Biases (input to hidden layer)
            self.b1 = np.ones((self.num_hidden_nodes))
            # Weights (hidden to output layer)
            self.W2 = np.random.randn(self.num_hidden_nodes, self.num_output_nodes)
            # Biases (hidden to output layer)
            self.b2 = np.ones((self.num_output_nodes))

    def init_adam_params(self):
        self.t = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 10**-8
        self.init_adam_moment()

    def init_adam_moment(self):
        # Moment matrices (input to hidden layer)
        self.mW1 = np.zeros_like(self.W1)
        self.vW1 = np.zeros_like(self.W1)
        self.mb1 = np.zeros_like(self.b1)
        self.vb1 = np.zeros_like(self.b1)

        # Moment matrices (hidden to output layer)
        self.mW2 = np.zeros_like(self.W2)
        self.vW2 = np.zeros_like(self.W2)
        self.mb2 = np.zeros_like(self.b2)
        self.vb2 = np.zeros_like(self.b2)

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return np.where(z > 0, 1, 0)

    def log_softmax(self, z):
        '''Used instead of the regular softmax function
           in order to make the model more numerically stable'''
        z -= np.max(z)
        return z - logsumexp(z, axis=1, keepdims=True, stable=False)

    def nll_loss(self, y, y_pred):
        '''Negative log likelihood: used in combination with log_softmax
           in order to make the model more numerically stable'''
        return -np.sum(y * y_pred)

    def mse_loss(self, y, y_pred):
        return (1/y.shape[0]) * np.sum(np.square(y - y_pred))

    def forward(self, X):
        if self.type == "class":
            return self._class_forward(X)
        if self.type == "reg":
            return self._reg_forward(X)
        
        return None
    
    def _class_forward(self, X):
        self.A1 = X
        self.Z2 = np.dot(self.A1, self.W1) + self.b1
        self.A2 = self.relu(self.Z2)
        self.Z3 = np.dot(self.A2, self.W2) + self.b2
        self.A3 = self.log_softmax(self.Z3)
        return self.A3

    def _reg_forward(self, X):
        self.A1 = X
        self.Z2 = np.dot(self.A1, self.W1) + self.b1
        self.A2 = self.relu(self.Z2)
        self.Z3 = np.dot(self.A2, self.W2) + self.b2
        self.A3 = self.Z3
        return self.A3

    def backward(self, X, y):
        m = X.shape[0]

        # Error in weights & biases 2
        dZ3 = np.exp(self.A3) - y if self.type == "class" else self.A3 - y
        dW2 = (1./m) * np.dot(self.A2.T, dZ3)
        db2 = np.sum(dZ3, axis=0)

        # Error in weights & biases 1
        dA2 = np.dot(dZ3, self.W2.T)
        dZ2 = dA2 * self.relu_derivative(self.Z2)
        dW1 = (1./m) * np.dot(X.T, dZ2)
        db1 = (1./m) * np.sum(dZ2, axis=0)

        # Update weights & biases
        if self.optimization == "adam":
            self.t += 1
            self.W2, self.mW2, self.vW2 = self.get_adam_param_update(self.W2, dW2, self.mW2, self.vW2)
            self.b2, self.mb2, self.vb2 = self.get_adam_param_update(self.b2, db2, self.mb2, self.vb2)
            self.W1, self.mW1, self.vW1 = self.get_adam_param_update(self.W1, dW1, self.mW1, self.vW1)
            self.b1, self.mb1, self.vb1 = self.get_adam_param_update(self.b1, db1, self.mb1, self.vb1)
        else:
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
        
    def get_adam_param_update(self, param, g, m, v):
        # Update biased moment estimate
        m = self.beta1 * m + (1 - self.beta1) * g
        v = self.beta2 * v + (1 - self.beta2) * np.square(g)

        # Find bias-corrected moment estimate
        m_hat = m / (1 - self.beta1**self.t)
        v_hat = v / (1 - self.beta2**self.t)

        # Update param
        param -= (self.learning_rate * m_hat) / (np.sqrt(v_hat) + self.eps)

        return param, m, v
    
    def get_loss(self, y, y_pred):
        if self.type == "class":
            return self.nll_loss(y, y_pred)
        if self.type == "reg":
            return self.mse_loss(y, y_pred)
        
        return None

    def fit(self, X, y, epochs=50):
        self.training_start_time = time.time()

        for i in range(1, epochs + 1):
            # Shuffle the dataset and split it into mini batches
            shuffled_X, shuffled_y = shuffle(X, y)
            batch_size = shuffled_X.shape[0] if self.mini_batch_size is None else self.mini_batch_size
            mini_batches_X = np.array_split(shuffled_X, range(batch_size, shuffled_X.shape[0], batch_size), axis=0)
            mini_batches_y = np.array_split(shuffled_y, range(batch_size, shuffled_y.shape[0], batch_size), axis=0)
            epoch_loss = 0
            
            for j in range(0, len(mini_batches_X)):
                y_pred = self.forward(mini_batches_X[j])
                epoch_loss += self.get_loss(mini_batches_y[j], y_pred)
                self.backward(mini_batches_X[j], mini_batches_y[j])

            print(f"Loss epoch {i}: {epoch_loss}")
            self.loss.append(epoch_loss)
        
        self.training_start_time = 0

    def predict(self, X):
        return self.forward(X)
    
    def load_params(self, W1, W2, b1, b2):
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2