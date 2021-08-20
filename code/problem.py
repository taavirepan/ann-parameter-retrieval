import numpy as np
import tensorflow as tf
import sys

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

class LRDrop:
	def __init__(self, model):
		self.model = model
		self.history = []
		self.alpha = 0.9
		self.running = 0
		self.running0 = 0
		self.last_drop = 0
		self.drop = 1.2
		self.threshold = 1 + 1/5000
		self.window = 25

	def __call__(self, epoch, logs):
		import tensorflow.keras.backend as K

		# if logs["loss"] < 3e-6: # XXX
			# return

		self.history.append(logs["loss"])
		self.running = self.running * self.alpha + (1-self.alpha) * self.history[-1]
		if len(self.history) >= self.window:
			self.running0 = self.running0 * self.alpha + (1-self.alpha) * self.history[-self.window]
		
		if self.running0 < self.running * self.threshold and epoch > self.last_drop + 2*self.window:
			print(f"[{epoch}] progress stalled at {self.running}, dropping lr")
			lr = K.get_value(self.model.optimizer.lr)
			K.set_value(self.model.optimizer.lr, lr/self.drop)
			self.last_drop = epoch

class Model:
	def __init__(self, model=None, n_layers=3, units=30, lr=1e-3, activation="tanh", loss="mse", batch=100, optimizer="Adam", **kwargs):
		@tf.function
		def get_gradients(x, y):
			with tf.GradientTape() as t:
				t.watch(x)
				loss = tf.reduce_mean(tf.square(self.model(x, training=True) - y), axis=1) 
			gradients = t.gradient(loss, self.model.trainable_weights)
			return gradients
		self._gradients = get_gradients

		if model is not None:
			self.model = model
			return

		# l2 = tf.keras.regularizers.l2(1e-8)
		l2 = None
		y = x = tf.keras.layers.Input((2+32,))
		for i in range(n_layers):
			y = tf.keras.layers.Dense(units, activation=activation, kernel_regularizer=l2)(y)
		y = tf.keras.layers.Dense(6)(y)

		if optimizer == "Adam":
			optimizer = tf.keras.optimizers.Adam(lr=lr)
		else:
			optimizer = getattr(tf.keras.optimizers, optimizer)(lr=lr)
		self.model = tf.keras.Model(inputs=x, outputs=y)
		self.model.compile(loss=loss, optimizer=optimizer, metrics=["mse"])
		self.batch = batch

	def train(self, data, callback=None, time_limit=30):
		from time import monotonic
		EPOCH_LIMIT = 999999
		
		drop_lr = LRDrop(self.model)
		def check_time(epoch, logs):
			logs["elapsed"] = monotonic() - start
			if logs["elapsed"] >= time_limit:
				self.model.stop_training = True
			if callback is not None:
				callback(epoch, logs)
			drop_lr(epoch, logs)

		start = monotonic()
		callbacks = [tf.keras.callbacks.LambdaCallback(on_epoch_end=check_time)]
		t = self.model.fit(data.x, data.y, epochs=EPOCH_LIMIT, verbose=0, callbacks=callbacks, batch_size=self.batch, validation_split=0.1)
		self.history = t.history
	
	def get_data(self):
		layers = dict()
		for layer in self.model.layers:
			layers[layer.name] = dict([(f"weight:{i}", w) for i,w in enumerate(layer.get_weights())])
		return {".json": self.model.to_json(), **layers}

class Problem:
	@staticmethod
	def setup_arguments(parser):
		parser.add_argument("-N", type=int, default=10000, help="Maximum number of samples to load")
		parser.add_argument("-u", "--units", type=int, default=30, help="Units per hidden layers")
		parser.add_argument("--layers", type=int, default=3, help="Number of hidden layers")
		parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
		parser.add_argument("--batch", default=100, type=int, help="Batch size")
		parser.add_argument("--activation", default="tanh")
		parser.add_argument("--loss", default="mse")
		parser.add_argument("--optimizer", default="Adam")

	def __init__(self, args):
		self.training_size = args.N
		self.model_args = dict(units=args.units, n_layers=args.layers, lr=args.lr, activation=args.activation, loss=args.loss, batch=args.batch,
			optimizer=args.optimizer)

	def get_model(self):
		return Model(**self.model_args)
	
	def make_new(self, **kw):
		from copy import copy
		ret = copy(self)
		ret.model_args = copy(self.model_args)
		for key,value in kw.items():
			if key in ["lr"]:
				value = float(value)
			elif key not in ["activation", "loss", "optimizer"]:
				value = int(value)
			
			if key in ret.model_args:
				ret.model_args[key] = value
		return ret
