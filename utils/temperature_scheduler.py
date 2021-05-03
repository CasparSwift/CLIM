class constant_scheduler(object):
	def __init__(self, args):
		self.temperature = args.temperature

	def step(self):
		return self.temperature

class linear_scheduler(object):
	def __init__(self, args):
		self.temperature = args.temperature
		self.temperature_low = args.temperature_end
		
		self.cooldown = args.temperature_cooldown
		total_steps = args.total_steps
		self.cooldown_steps = total_steps*self.cooldown
		self.current_step = 0
		self.increment = (self.temperature_low - self.temperature) / self.cooldown_steps

	def step(self):
		assert self.temperature > 0
		if self.current_step > self.cooldown_steps:
			return self.temperature
		else:
			self.current_step += 1
			self.temperature += self.increment
			return self.temperature

temperature_scheduler_factory = {
	'linear': linear_scheduler,
	'constant': constant_scheduler
}