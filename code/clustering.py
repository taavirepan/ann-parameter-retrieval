import numpy as np
from collections import namedtuple

class Groups:
	def __init__(self, N):
		self.group = np.arange(N)
		self.size = np.ones(N, dtype=int)
		self._len = N

	def __len__(self):
		return self._len
	
	def groups(self):
		gids = np.unique(self.group)
		idx = np.arange(len(self.group))
		for i in gids:
			yield idx[self.group == i]

	def gsize(self, i):
		return self.size[self.group[i]]

	def merge(self, i0, j0):
		i = self.group[i0]
		j = self.group[j0]
		if i == j:
			return
		
		self.group[self.group == j] = i
		self.size[i] += self.size[j]
		self._len -= 1

Pair = namedtuple("Pair", "distance i j")
def all_pairs(points):
	ret = []
	for i in range(len(points)):
		for j in range(i+1, len(points)):
			distance = np.sum(np.square(points[i] - points[j]))
			ret.append(Pair(distance, i, j))
	ret.sort()
	return ret

def search_clusters(points, factor=1.03, max_groups=20, min_distance=0.1):
	g = Groups(len(points))
	
	prev = np.inf
	for p in all_pairs(points):
		if g.group[p.i] == g.group[p.j]:
			continue
		
		if (p.distance >= min_distance**2) and (prev*factor**2 <= p.distance):
			if len(g) <= max_groups:
				break
			else:
				factor = 0 # break as soon as we have enough groups merged

		g.merge(p.i, p.j)
		prev = p.distance

	return list(g.groups())
