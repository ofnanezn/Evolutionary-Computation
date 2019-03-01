import sys

class ObjectiveFunction():
	def __init__(type, f=lambda x: x, limits):
		self.type = type
		self.f = f
		self.limits = limits


def dominate(objectives, x, y):
	count = 0
	for objective in objectives:
		f_x = objective.f(x)
		f_y = objective.f(y)
		if objective.type == 'max':
			if f_x < f_y:
				return False
			elif f_x == f_y:
				continue
			else:
				count += 1
		else:
			if f_x > f_y:
				return False
			elif f_x == f_y:
				continue
			else:
				count += 1
	return count >= 1


def fast_non_dominated_sort(P, objectives):
	S = {}
	for p in P:
		gen_q = (x for x in P if x != p)
		for q in gen_q:
			P_star.append(P)
			gen_q = (y for y in P_star if y != p)
			for q in gen_q:
				if dominate(objectives, p, q):

	return P_star

def crowding_distance_assignment(I, objectives):
	l = len(I)
	I = [(i, 0) for i in I]
	for m in objectives:
		I.sort(key=lambda x: m(x[0]))
		I[0] = I[-1] = float('inf')
		for i in range(1, l-2):
			I[i][1] += m(I[i+1][0]) - m(I[i-1][0])

