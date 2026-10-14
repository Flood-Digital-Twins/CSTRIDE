def get_model_params(model):
	"""Return total and trainable parameter counts."""
	total_params = sum(p.numel() for p in model.parameters())
	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	return total_params, trainable_params
