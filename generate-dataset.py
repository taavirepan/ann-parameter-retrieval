import numpy as np
from collections import namedtuple
from code import Data, calculate_spectra

class DataSource:
	def __init__(self):
		self.split = None
		self.data = None
	
	def query_size(self, min_coverage):
		ret = 0
		if self.data and self.data.coverage >= min_coverage:
			ret = len(self.data.x)
		elif self.data and self.data.coverage < min_coverage:
			return np.inf

		if self.split and self.split.sources:
			for src in self.split.sources:
				ret = min(ret, src.query_size(min_coverage))
		return ret
	
	def append(self, data, N):
		if self.data is None:
			self.data = data[:N]
			return
		
		m = N - len(self.data)
		assert m >= 0
		self.data += data[:m]

	def push(self, data, N):
		if len(data.x) == 0:
			return
		if self.split is not None:
			self.split.push(data, N)
		if self.data is None or len(self.data.x) < N:
			self.append(data, N)

class Definition:
	Uniform = namedtuple("Uniform", "low high")
	def __init__(self, **kwargs):
		self.parameters = kwargs

	def with_loss(self, low, high=None):
		if high is None:
			low, high = -low, low
		add_loss = lambda x: (x[0], self.Uniform(low, high))
		parameters = dict((k,add_loss(v)) for k,v in self.parameters.items())
		return Definition(**parameters)

	@property
	def config(self):
		import json
		ret = dict()
		for key,value in self.parameters.items():
			ret[key] = [[type(g).__name__, g._asdict()] for g in value]
		return json.dumps(ret)

	def __call__(self, N, rng):
		parameters = np.zeros((N, 6))
		for key, value in self.parameters.items():
			idx = dict(epsilon=[0,2], mu=[4], epsilonx=[0], epsilonz=[2])[key]
			for i, gen in enumerate(value):
				if isinstance(gen, self.Uniform):
					x = rng.uniform(gen[0], gen[1], N)
				else:
					raise NotImplementedError
				for j in idx:
					parameters[:,j+i] = x
		return parameters


class Generator(DataSource):
	def __init__(self, definition, *, seed=None, kx, k0, slab):
		self.definition = definition
		self.random = np.random.RandomState(seed)
		self.kx = kx
		self.k0 = k0
		self.slab = slab
		super().__init__()

	def generate(self, N, limit):
		# FIXME: to be reproducible for any N random parameters should be generated row-wise, not columnwise, 
		#        but this is not so super important, so I am not going to implement it yet
		parameters = self.definition(N, self.random)
		k0 = self.random.uniform(*self.k0, N)
		h = self.random.uniform(*self.slab, N)
		rt = calculate_spectra(parameters, k0=k0, h=h, kx=self.kx)
		self.push(Data(rt, parameters, coverage=1.0), limit)


class ParallelGenerator(DataSource):
	def __init__(self, definition, threads, *, seed=None, **kwargs):
		from multiprocessing import Process, Pipe
		super().__init__()
		self.random = np.random.RandomState(seed)
		self.pipes = []
		self.processes = []
		for i in range(threads):
			seed = self.random.randint(2**32-1)
			self.pipes.append(Pipe())
			self.processes.append(Process(target=ParallelGenerator.child, args=(definition, seed, self.pipes[-1][1], kwargs)))
			self.processes[-1].start()
	
	def close(self):
		for i in range(len(self.pipes)):
			self.pipes[i][0].send(-1)
			self.processes[i].join()
			
	@staticmethod
	def child(definition, seed, pipe, kwargs):
		generator = Generator(definition, seed=seed, **kwargs)
		while True:
			N = pipe.recv()
			if N == -1:
				return
			generator.data = None
			while generator.data is None or len(generator.data.x) < N:
				generator.generate(N, N)
			pipe.send(generator.data)

	def generate(self, N, limit):
		from math import ceil
		data = None
		for pipe in self.pipes:
			pipe[0].send(N)

		for pipe in self.pipes:
			data_i = pipe[0].recv()
			data = data_i if data is None else data + data_i
		self.push(data, limit)


class BaseSplitter:
	def __init__(self, n_samples):
		self.classifier = None
		self._classifier_len = n_samples
		self._classifier_data = None

	def build_classifier(self, data):
		raise NotImplementedError

	def check_classifier(self, data, limit):
		if self.classifier is not None:
			return True

		if self._classifier_data is None:
			self._classifier_data = data[:self._classifier_len]
			leftover = data[self._classifier_len:]
		else:
			n = self._classifier_len - len(self._classifier_data)
			self._classifier_data += data[:n]
			leftover = data[n:]
		
		if self._classifier_len == len(self._classifier_data):
			self.classifier = self.build_classifier(self._classifier_data)
			# Call push again, with all data, and then return false to avoid double processing
			self.push(self._classifier_data, limit)
			if len(leftover) > 0:
				self.push(leftover, limit)
		return False


class SplitBySpectrum(BaseSplitter):
	def __init__(self, cutoff):
		super().__init__(2000)
		self.sources = [DataSource(), DataSource()]
		self.cutoff = cutoff

	def build_classifier(self, data):
		print(f"{self}: Setting reference with {len(data.x)} points")
		best = np.inf, None
		for i in range(len(data.x)):
			diff = np.mean(np.square(data.x[i,2:] - data.x[:,2:]), axis=1)
			n = np.count_nonzero(diff <= self.cutoff)
			score = abs(n - len(data.x)/2)
			if score < best[0]:
				best = score, i
		i = best[1]

		self.ref_x = data.x[i,2:]
		self.ref_y = data.y[i]
		return True

	def push(self, data, limit):
		if not self.check_classifier(data, limit):
			return

		diff = np.mean(np.square(self.ref_x - data.x[:,2:]), axis=1)
		idx1, = np.nonzero(diff <= self.cutoff)
		idx2, = np.nonzero(diff > self.cutoff)
		self.sources[0].push(Data(data.x[idx1], data.y[idx1], data.coverage * len(idx1)/len(diff)), limit)
		self.sources[1].push(Data(data.x[idx2], data.y[idx2], data.coverage * len(idx2)/len(diff)), limit)

class SplitByConnected(BaseSplitter):
	def __init__(self, **opts):
		super().__init__(2000)
		self.opts = opts
		self.sources = []

	def build_classifier(self, data):
		from code import search_clusters
		from scipy import spatial
		try:
			from sklearn import cluster
		except ImportError:
			pass

		print(f"{self}: setting up classifier ({len(data)} points)")
		classifier = [spatial.cKDTree(data.y[idx]) for idx in search_clusters(data.y)]
		self.sources = [DataSource() for i in range(len(classifier))]
		return classifier

	def push(self, data, limit):
		if not self.check_classifier(data, limit):
			return
		
		distances = []
		for classifier in self.classifier:
			distances.append(classifier.query(data.y)[0])
		distances = np.argmin(np.array(distances), axis=0)

		for i, source in enumerate(self.sources):
			idx, = np.nonzero(distances == i)
			if len(idx) > 0:
				coverage = data.coverage * len(idx) / len(data.y)
				source.push(Data(data.x[idx], data.y[idx], coverage), limit)

class ParallelSink:
	def __init__(self, sink):
		from multiprocessing import Process, Pipe
		self.pipe, child_pipe = Pipe()
		self.process = Process(target=ParallelSink.child, args=(child_pipe, sink))
		self.process.start()

	def close(self):
		self.pipe.send("stop")
		self.process.join()
	
	@staticmethod
	def child(pipe, sink):
		while True:
			cmd = pipe.recv()
			if cmd == "stop":
				break
			elif cmd == "sources":
				pipe.send(sink.sources)
			else:
				data, limit = cmd
				sink.push(data, limit)

	def push(self, data, limit):
		self.pipe.send([data, limit])
	
	@property
	def sources(self):
		self.pipe.send("sources")
		return self.pipe.recv()

def above_coverage(sources, coverage):
	return list(filter(lambda src: src.data and src.data.coverage >= coverage, sources))

def write_output(file, source, min_coverage=0, *, prefix="", level=0, index=0, parent=None):
	import colored
	
	name = f"{prefix}{level:02d}.{index:03d}"

	splits = above_coverage(source.split.sources, min_coverage) if source.split else []
	if len(splits) == 1:
		print(colored.stylize(f"{name:30s}: {str(source.data.y.shape):10s} {source.data.coverage*100: 6.2f}%", colored.fg("red")))
		return write_output(file, splits[0], min_coverage, prefix=prefix, level=level, index=index, parent=parent)

	if source.data is not None and source.data.coverage >= min_coverage:
		print(f"{name:30s}: {str(source.data.y.shape):10s} {source.data.coverage*100: 6.2f}%")
		group = file.create_group(name)
		group.create_dataset("parameters", data=source.data.y)
		group.create_dataset("k0", data=source.data.x[:,0])
		group.create_dataset("h", data=source.data.x[:,1])
		group.attrs["level"] = level
		group.attrs["coverage"] = source.data.coverage
		if parent:
			group.attrs["parent"] = parent
	elif source.data is None or source.data.coverage < min_coverage:
		print(colored.stylize(f"{name:30s}: {str(source.data.y.shape):10s} {source.data.coverage*100: 6.2f}%", colored.attr("dim")))
		return 0
	
	ret = 0
	if source.split is not None:
		for i, src in enumerate(source.split.sources):
			if isinstance(source.split, SplitBySpectrum):
				ret += write_output(file, src, min_coverage, prefix=prefix, level=level+1, index=ret, parent=name)
			else:
				ret += write_output(file, src, min_coverage, prefix=prefix, level=level, index=10+ret, parent=name)
	group.attrs["leaf"] = ret == 0
	return 1

definitions = dict()
definitions["narrow"] = Definition(
	epsilon=[Definition.Uniform(-5, 5), Definition.Uniform(0, 0)],
	mu=[Definition.Uniform(-5, 5), Definition.Uniform(0, 0)])

definitions["wide"] = Definition(
	epsilon=[Definition.Uniform(-10, 10), Definition.Uniform(0, 0)],
	mu=[Definition.Uniform(-10, 10), Definition.Uniform(0, 0)])

definitions["anisotropic"] = Definition(
	epsilonx=[Definition.Uniform(-5, 5), Definition.Uniform(0, 0)],
	epsilonz=[Definition.Uniform(-5, 5), Definition.Uniform(0, 0)],
	mu=[Definition.Uniform(-5, 5), Definition.Uniform(0, 0)])

if __name__ == "__main__":
	from argparse import ArgumentParser
	import h5py
	
	parser = ArgumentParser(description="Subdivide the parameter space and generate the training data for ANNs")
	parser.add_argument("definition", choices=definitions.keys(), default="narrow", help="Dataset to generate")
	parser.add_argument("splits", type=int, help="Number of parameter space splits do perform")
	parser.add_argument("-o", "--output", help="Output file, default: data/{definition}.hdf5")

	tuning = parser.add_argument_group("Tuning parameters for splitting")
	tuning.add_argument("--coverage", default=0.25, type=float, help="Drop subspaces which cover less than N%% of the whole parameter space", metavar="N")
	tuning.add_argument("--cutoff", default=0.03, type=float, help="Cutoff for splitting similar and dissimilar spectra")
	tuning.add_argument("--alpha", default=1.03, type=float, help="Alpha parameter for the clustering algorithm")

	generation = parser.add_argument_group("Tuning sample generation")
	parser.add_argument("-N", default=10000, type=int, help="Amount of samples to generate for each subspace")
	generation.add_argument("--seed", default=1, type=int, help="Random seed to use")
	generation.add_argument("--kx", nargs=2, type=float, default=[0, 1 - 1e-6], help="Start and end for incidence angles in angular spectrum (expressed as tangential wavevector kx/k0)", metavar=("BEGIN", "END"))
	generation.add_argument("--slab", nargs=2, default=[0.1, 0.4], type=float, help="Range of slab thicknesses to generate", metavar=("BEGIN", "END"))
	generation.add_argument("--k0", nargs=2, default=[2.9, 8.4], type=float, help="Range of wavelengths to generate (expressed in free space wavenumber k0)", metavar=("BEGIN", "END"))

	parallel = parser.add_argument_group("Parallel generation options")
	parallel.add_argument("-j", "--threads", type=int, help="Use N for data generation", metavar="N")
	parallel.add_argument("-p", "--parallel", action="store_true", help="Perform clustering in separate threads")
	parallel.add_argument("-G", default=10000, type=int, help="Samples to generate per iteration (per thread). If large number of splits is required, then it makes sense to generate larger number of samples per iteration.")

	args = parser.parse_args()

	if args.threads:
		level = gen = ParallelGenerator(definitions[args.definition], args.threads, seed=args.seed, kx=args.kx, k0=args.k0, slab=args.slab)
	else:
		level = gen = Generator(definitions[args.definition], seed=args.seed, kx=args.kx, k0=args.k0, slab=args.slab)

	roots = [level]

	cluster_args = dict(factor=args.alpha)
	for level in roots:
		for n in range(args.splits):
			level.split = SplitBySpectrum(args.cutoff)
			level.split.sources[1].split = SplitBySpectrum(args.cutoff)
			level.split.sources[0].split = ParallelSink(SplitByConnected(**cluster_args)) if args.parallel else SplitByConnected(**cluster_args)
			level = level.split.sources[1]

	while True:
		gen.generate(args.G, args.N)
		if gen.query_size(min_coverage=args.coverage/100) >= args.N:
			break

	if args.threads:
		gen.close()

	with h5py.File(args.output or f"data/{args.definition}.hdf5", "w") as file:
		file.attrs["cutoff"] = args.cutoff
		file.attrs["splits"] = args.splits
		file.attrs["min_coverage"] = args.coverage
		file.attrs["seed"] = args.seed
		file.attrs["batch"] = args.G
		file.attrs["kx"] = args.kx
		file.attrs["definition"] = args.definition
		file.attrs["definition_config"] = definitions[args.definition].config
		file.attrs["slab"] = args.slab
		file.attrs["k0"] = args.k0
		file.attrs["maxiter"] = args.maxiter
		if args.threads:
			file.attrs["threads"] = args.threads
		write_output(file, gen, args.coverage/100)

	if args.parallel:
		for level in roots:
			while level.split:
				if level.split.sources[0].split:
					level.split.sources[0].split.close()
				level = level.split.sources[1]
	
