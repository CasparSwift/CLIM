import torch


wrapper_factory ={
	''

}


class graph_causal_optimizer(object):
	def __init__(self, opts):
		self.opts = opts
		self.round = [[0,1,2,6],[3],[4],[5]]
		self.term = 7
		assert len(opts)==4

	def step(self, loss, global_step):
		path = global_step%self.term

		for r, opt in zip(self.round, self.opts):
			if path in r:
				opt.zero_grad()
				loss.backward()
				opt.step()
				break
