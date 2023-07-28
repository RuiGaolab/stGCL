import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torchvision import models
from stGCL.vit_model import VisionTransformer

class extract_model(Module):
	def __init__(self,net,crop_size, patch_size):
		super(extract_model, self).__init__()

		if net == "resnet50":
			# model = resnet50_model().cuda()
			model = models.resnet50(pretrained=True).requires_grad_(False).cuda()

		elif net == "VGG16":
			model = models.vgg16(pretrained=True).requires_grad_(False).cuda()

		elif net == "densenet":
			model = models.densenet121(pretrained=True).requires_grad_(False).cuda()
		elif net == "ViT":

			model = VisionTransformer(img_size=crop_size, patch_size=patch_size, num_classes=1000,
									  embed_dim=768, depth=6).requires_grad_(False).cuda()
		else:
			raise RuntimeError('The selected model is not supported!')
		self.f = []

		for name, module in model.named_children():
			if name not in ["fc","maxpool","classifier"]:
				self.f.append(module)
		# encoder
		self.f = nn.Sequential(*self.f)

	def forward(self, x):
		x       = self.f(x)
		feature = torch.flatten(x, start_dim=1)
		return F.normalize(feature, dim=-1)
