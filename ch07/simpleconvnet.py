import sys, os
sys.path.appent(os.pardir)
import numpy as np
from layers import Convolution

# インスタンス引数
#　input_dim:入力データの次元（チャネル(R/G/B)、高さ、幅）
#  conv_param:畳込層のハイパーパラメータをディクショナリ型で保持
#  hidden_size:隠れ層（全結合）のニューロン数
#  output_size:出力層（全結合）のニューロン数
#  weight_init_std:初期化の際の重みの標準偏差

class SimpleConvNet:
	def __init__(self, input_dim=(1,28,28),
				 conv_param={'filter_num':30, 'filter_size':5, 
				 			 'pad':0, 'stride':1},
				 hidden_size=100, output_size=10, weight_init_std=0.01):

		filter_num = conv_param['filter_num']
		filter_size = conv_param['filter_size']
		filter_pad = conv_param['filter_pad']
		filter_stride = conv_param['filter_stride']
		input_size = input_dim[1]
		# 入力サイズから出力サイズを求める
		# Output(Height/Width) = ((Input(Height/Width) + (2*Padding size) - Filter size(Height/Width)) / (2 * Stride)) + 1
		conv_output_size = (input_size - filter_size + 2 * filter_pad) / filter_stride + 1
		pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))

		self.params = {}
		self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
		self.params['b1'] = np.zeros(filter_num)
		self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
		self.params['b2'] = np.zeros(hidden_size)
		self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
		self.params['b3'] = np.zeros(output_size)

		self.layers = OrderedDict()
		self.layers['Conv1'] = Convolution(self.params['W1'], 
										   self.params['b1'], 
										   filter_stride, 
										   filter_pad)

		self.layers['Relu'] = Relu()
		self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
		self.layers['Affine1'] = Affine(self.params['W2'],
										self.paramas['b2'],)
		self.layers['Relu2'] = Relu()
		self.layers['Affine2'] = Affine(self.params['W3'],
										self.params['b3'])
		self.last_layer = SoftmaxWithLoss()

	def predict(self, x):
		for layer in self.layers.values():
			x = layer.forward(x)
		return x

	def loss(self, x, t):
		y = self.predict(x)
		return self.lastLayer.forward(y, t)

	def gradient(self, x, t):
		# forward
		self.loss(x,t)

		# backward
		dout = 1
		dout = self.lastLayer.backward(dout)

		layers = list(self.layers.values())
		layers.revers()
		for layer in layers:
			dout = layer.backward(dout)

		grads = {}
		grads['W1'] = self.layers.['Conv1'].dW
		grads['b1'] = self.layers.['Conv1'].db
		grads['W2'] = self.layers.['Affine1'].dW
		grads['b2'] = self.layers.['Affine1'].db
		grads['W3'] = self.layers.['Affine2'].dW
		grads['b3'] = self.layers.['Affine2'].db

		return grads




