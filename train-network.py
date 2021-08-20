import h5py
import argparse
import numpy as np
from pathlib import Path
from code import Data, Problem

parser = argparse.ArgumentParser(description="Trains ANN based on a single dataset")
parser.add_argument("output", help="Output file for the trained ANN")
parser.add_argument("data", help="Data file for training data")
parser.add_argument("dataset", help="Dataset name to load")
parser.add_argument("run", help="Name for saving the ANN in the output file")

parser.add_argument("--data_loss", type=float, default=2, help="Generate imaginary parts of material parameters in range [-LOSS,LOSS]", metavar="LOSS")
parser.add_argument("--time", type=float, default=10, help="Train for T seconds", metavar="T")

Problem.setup_arguments(parser.add_argument_group("ANN-specific options"))
args = parser.parse_args()

with h5py.File(args.data, "r") as file:
	data = file[args.dataset]
	parameters = np.array(data["parameters"])
	k0 = np.array(data["k0"])
	h = np.array(data["h"])
	kx = data.attrs.get("kx")

	rng = np.random.RandomState(1)
	parameters[:,1] = rng.uniform(-args.data_loss, args.data_loss, len(parameters))
	parameters[:,3] = rng.uniform(-args.data_loss, args.data_loss, len(parameters))
	parameters[:,5] = rng.uniform(-args.data_loss, args.data_loss, len(parameters))

	attrs = dict(data.attrs)
	data = Data.from_parameters(parameters, coverage=data.attrs["coverage"], k0=k0, h=h, kx=kx)

network = Problem(args).get_model()
network.train(data, time_limit=args.time)

with h5py.File(args.output, "w") as f:
	output = f.create_group(args.run)

	loss = np.array(network.history["loss"])
	mse = np.array(network.history["mse"])
	val_loss = np.array(network.history["val_loss"]) if "val_loss" in network.history else None
	elapsed = np.array(network.history["elapsed"])

	if val_loss is not None:
		output.create_dataset("val_loss", data=val_loss)
	output.create_dataset("loss", data=loss)
	output.create_dataset("mse", data=mse)
	output.create_dataset("elapsed", data=elapsed)
	print(f"network '{args.run}' finished training, final loss {loss[-1]:.2e} after {len(loss)} epochs")

	def _save(target, data):
		for key,value in data.items():
			if key.startswith("."):
				target.attrs[key[1:]] = value
			elif isinstance(value, dict):
				_save(target.create_group(key), value)
			else:
				target.create_dataset(key, data=value)

	_save(output.create_group("model"), network.get_data())

	g = output.create_group("data")
	g.create_dataset("loss", data=loss)
	g.attrs["coverage"] = data.coverage # XXX check if None?
	# TODO: "parent" attr
	g.attrs["level"] = attrs["level"]
	g.attrs["leaf"] = attrs["leaf"]
	g["parameters"] = h5py.ExternalLink(Path(args.data).name, f"/{args.dataset}/parameters")
	g["h"] = h5py.ExternalLink(Path(args.data).name, f"/{args.dataset}/h")
	g["k0"] = h5py.ExternalLink(Path(args.data).name, f"/{args.dataset}/k0")


