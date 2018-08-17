import numpy as np

class DeepNeuralNetwork():
	def __init__(self, layers=[10, 8, 1], alpha=0.001):
		self.layers = layers
		self.alpha = alpha
		self.parameters = None
		np.random.seed(42)
		self.epsilon = 1e-8
	
	def train_model(self, X, Y, num_iterations=20000):
		self.init_parameters(X.shape[0])
		v, s = self.init_adam()
		old_cost = self.cost(X, Y)
		print("cost after 0 iterations: {}".format(old_cost))
		for i in range(1, num_iterations):
			aL, cache = self.f_prop(X)
			grads = self.b_prop(aL, cache, Y)
			self.update_parameters(grads=grads, v=v, s=s, alpha=self.alpha)
			if i % 1000 == 0:
				new_cost = self.cost(X, Y)
				if new_cost > old_cost:
					print("Alpha updated {} -> {}".format(str(self.alpha), str(self.alpha /2.)))
					self.alpha /= 2.
				old_cost = new_cost

				print("cost after {} iterations: {}".format(i, new_cost))
		print("final cost: {}".format(self.cost(X, Y)))


	def init_adam(self):
		v = {}
		s = {}
		for i in range(len(self.layers)):
			wi = "W" + str(i)
			bi = "b" + str(i)
			v[wi] = np.zeros(self.parameters[wi].shape)
			v[bi] = np.zeros(self.parameters[bi].shape)
			s[wi] = np.zeros(self.parameters[wi].shape)
			s[bi] = np.zeros(self.parameters[bi].shape)
		return v, s


	def update_parameters(self, grads, alpha, v, s, beta1=0.9, beta2=0.999):
		s_corrected = {}
		v_corrected = {}
		for i in range(len(self.layers)):
			w_i = "W" + str(i)
			b_i = "b" + str(i)
			v[w_i] = v[w_i] * beta1 + (1-beta1) * grads[w_i]
			v[b_i] = v[b_i] * beta1 + (1-beta1) * grads[b_i]
			v_corrected[w_i] = v[w_i] / (1-beta1)
			v_corrected[b_i] = v[b_i] / (1-beta1)

			s[w_i] = s[w_i] * beta2 + (1-beta2) * np.power(grads[w_i], 2)
			s[b_i] = s[b_i] * beta2 + (1-beta2) * np.power(grads[b_i], 2)

			s_corrected[w_i] = s[w_i] / (1-beta2)
			s_corrected[b_i] = s[b_i] / (1-beta2)
				

			self.parameters[w_i] = self.parameters[w_i] - (alpha * v_corrected[w_i] / (np.sqrt(s_corrected[w_i] + self.epsilon)))
			self.parameters[b_i] = self.parameters[b_i] - (alpha * v_corrected[b_i] / (np.sqrt(s_corrected[b_i] + self.epsilon)))

	def init_parameters(self, n_x):
		self.parameters = {}
		prev_n_h = n_x
		i = 0
		for n_h in self.layers:
			self.parameters["W" + str(i)] = np.random.randn(n_h, prev_n_h) * np.sqrt(2. / (prev_n_h + self.epsilon))
			self.parameters["b" + str(i)] = np.zeros((n_h, 1))
			prev_n_h = n_h
			i += 1

	def cost(self, X, Y):
		m = X.shape[1]
		prev_aL = X 
		for i in range(len(self.layers)):
			WL = self.parameters["W" + str(i)]
			bL = self.parameters["b" + str(i)]
			zL = np.dot(WL, prev_aL) + bL
			if i == len(self.layers) - 1:
				prev_aL = sigmoid(zL)
			else:
				prev_aL = relu(zL)

		return (-1. / m) * np.sum(Y * np.log(prev_aL + self.epsilon) + (1. - Y) * np.log((1. - prev_aL) + self.epsilon), axis=1, keepdims=True)

	def b_prop(self, aL, cache, Y):
		grads = {}

		Y = Y.reshape(aL.shape)
		last_layer = len(self.layers) - 1
		dAL = - (np.divide(Y, aL + self.epsilon) - np.divide(1- Y, (1 - aL) + self.epsilon))
		dZ = dAL * gradient(cache["z" + str(last_layer)], "sigmoid")
		dW, db, dA_prev = self.b_prop_single(dZ, cache[str(last_layer)])
		grads["W" + str(last_layer)] = dW 
		grads["b" + str(last_layer)] = db

		for i in reversed(range(len(self.layers) - 1)):
			dZ = dA_prev * gradient(cache["z" + str(i)], "relu")
			dW, db, dA_prev = self.b_prop_single(dZ, cache[str(i)])
			grads["W" + str(i)] = dW 
			grads["b" + str(i)] = db
		return grads
		

	def b_prop_single(self, dZ, cache):
		A_prev, W, _ = cache
		m = A_prev.shape[1]

		dW = np.dot(dZ, A_prev.T) / m
		db = np.sum(dZ, axis=1, keepdims=True) / m
		dA_prev = np.dot(W.T, dZ)
		return dW, db, dA_prev


	def f_prop(self, X):
		cache = {}
		prev_aL = X
		for i in range(len(self.layers)):
			WL = self.parameters["W" + str(i)]
			bL = self.parameters["b" + str(i)]
			zL = np.dot(WL, prev_aL) + bL
			cache[str(i)] = (prev_aL, WL, bL) 
			cache["z" + str(i)] = zL
			if i == len(self.layers) - 1:
				prev_aL = sigmoid(zL)
			else:
				prev_aL = relu(zL)

		return prev_aL, cache



	def predict(self, X, Y, X_test, Y_test):
		Y_pred, _ = self.f_prop(X)
		Y_pred = Y_pred > 0.5 
		print("training: {}".format(np.sum(Y_pred == Y) / Y.shape[1]))
		Y_pred, _ = self.f_prop(X_test)
		Y_pred = Y_pred > 0.5 
		print("test : {}".format(np.sum(Y_pred == Y_test) / Y_test.shape[1]))
		return np.sum(Y_pred == Y_test) / Y_test.shape[1]

	
	def gradient_check_n(grads, X, Y, epsilon = 1e-7):
		pass
	
def gradient(z, grad_type):
	if grad_type == "relu":
		return relu(z)
	else: # grad_type = sigmoid
		return sigmoid(z) * (1. - sigmoid(z))

def relu(x):
	return np.maximum(x, 0)

def sigmoid(z):
	return 1. / (1. + np.exp(-z))
