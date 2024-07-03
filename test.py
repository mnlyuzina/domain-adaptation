import torch
from matplotlib import pyplot as plt

from support_functions import plot

class UnNormalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def inference(dl, model, device, n_ims = 15, masks=True):
	if masks:
		cols = n_ims // 2
	else:
		cols = n_ims // 2
	rows = n_ims // cols
	
	count = 1
	ims, preds = [], []
	if masks:
		gts = []
	for idx, data in enumerate(dl):
		if masks:
			im, gt = data
		else:
			im = data

		# Get predicted mask
		with torch.no_grad():
			pred = torch.argmax(model(im.to(device)), dim = 1)
		ims.append(im)
		if masks:
			gts.append(gt)
		preds.append(pred)
		
	plt.figure(figsize = (20, 15))
	if masks:
		inputs = zip(ims, gts, preds)
	else:
		inputs = zip(ims, preds)
	for idx, data in enumerate(inputs):
		if masks:
			im, gt, pred = data
		else:
			im, pred = data
		if idx == cols:
			break
		
		# First plot
		if not masks:
			count = plot(cols, rows, count, im)

		if masks:
			# Second plot
			count = plot(cols, rows, count, im = gt.squeeze(0), gt = True, title = "Ground Truth")

		# Third plot
		count = plot(cols, rows, count, im = pred, title = "Predicted Mask")
	plt.show()
