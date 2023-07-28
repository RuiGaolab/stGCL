import os
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from stGCL.modules import extract_model
from sklearn.metrics.cluster import adjusted_rand_score
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import json
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import glob2
import shutil
from PIL import Image
from tqdm import tqdm
import os
import torch
import random
import numpy as np
from stGCL.process import set_seed
import pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS = 10000000000
from torchvision import transforms
from torch.utils.data import Dataset

class CustomDataset(Dataset):
	def __init__(self, imgs_path = None, sampling_index = None, transform = None):
		file_list = glob2.glob( str(imgs_path) + "/*.jpeg" )
		self.data      = []
		self.barcode   = []
		self.imgs_path=imgs_path

		for file in file_list:
			self.data.append( file )
			self.barcode.append(file.replace(str(self.imgs_path), "")[1:-5])
		self.transform = transform

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		img_path   = self.data[idx]
		img        = Image.open( img_path )
		image_code = self.barcode[idx]
		if self.transform is not None:
			pos = self.transform(img)
		return pos, image_code

def image_transform(size):
	return transforms.Compose([
		# transforms.RandomResizedCrop(size),
		transforms.RandomHorizontalFlip(p=0.5),
		transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
		transforms.RandomGrayscale(p=0.8),
		transforms.ToTensor(),
		transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

# train_transform_64 = transforms.Compose([
# 	transforms.RandomResizedCrop(64),
# 	transforms.RandomHorizontalFlip(p=0.5),
# 	transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
# 	transforms.RandomGrayscale(p=0.8),
# 	transforms.ToTensor(),
# 	transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
#
# train_transform_32 = transforms.Compose([
# 	transforms.RandomResizedCrop(32),
# 	transforms.RandomHorizontalFlip(p=0.5),
# 	transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
# 	transforms.RandomGrayscale(p=0.8),
# 	transforms.ToTensor(),
# 	transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

def tiling(image,scale, positionsdata, crop_size,image_adress) :
	"""
	Tiling H&E images to small tiles based on spot spatial location
	"""

	x_pixel = (positionsdata["x"] * scale).astype(int)
	y_pixel = (positionsdata["y"] * scale).astype(int)
	# Check the exist of out_path
	tillingPath=os.path.join(image_adress, "tem_img")
	if os.path.exists(tillingPath):
		shutil.rmtree(tillingPath)
	os.mkdir(tillingPath)
	x_pixel = x_pixel.tolist()
	y_pixel = y_pixel.tolist()

	img_new = image.copy()
	for i in range(len(x_pixel)):
		x = x_pixel[i]
		y = y_pixel[i]
		img_new[int(x - 2):int(x + 2), int(y - 2):int(y + 2), :] = 0

	cv2.imwrite(os.path.join(image_adress, "map.jpg"), img_new)
	# if image.dtype == np.float32 or image.dtype == np.float64:
	# 	image = (image * 255).astype(np.uint8)
	# img_pillow = Image.fromarray(image)
	tile_names = []

	with tqdm(
		total=len(positionsdata),
		desc="Tiling image",
		bar_format="{l_bar}{bar} [ time left: {remaining} ]",
	) as pbar:
		for barcode, imagerow, imagecol in zip(positionsdata.index, x_pixel , y_pixel ):
			imagerow_down  = int(imagerow - crop_size/ 2)
			imagerow_up    = int(imagerow + crop_size/ 2)
			imagecol_left  = int(imagecol - crop_size/ 2)
			imagecol_right = int(imagecol + crop_size/ 2)
			tile = image[imagerow_down:imagerow_up, imagecol_left:imagecol_right]

			tile_name = str(barcode)
			out_tile = str(tillingPath ) + '/' + str(tile_name + ".jpeg")
			cv2.imwrite(out_tile, tile)

			pbar.update(1)

def image_representation(positionsdata,image_adress,image_file,label=None,score_adress="",pca_num=None,k=7,
						 scale_file=None, image_net="Vit",image_type="full",crop_size=256,batch_size_lw=128,
						 cal_ARI=False,patch_size=64,scale=1,load=False,border=0):
	"""
	Extract image features for each spot.

	Parameters
	----------
	positionsdata
	    Coordinate information of spot.
	image_adress
	    The adress of the image.
	image_file
	    The name of image file.
	label
	    Label for each spot.
	score_adress
	    The location where the extracted image features are stored.
	pca_num
	    The dimension of pca dimensionality reduction.
	k
	    Number of clusters.
	scale_file
	    The storage location of the scale file file.
	image_net
	    Feature Extraction Model.
	image_type
	    The type of image being extracted.
	crop_size
	    The side length of each spot image, in order to consider
	    the surrounding information, here we recommend 256.
	patch_size
		Patch_size of the vit model.
	batch_size_lw
	    Batchsize of feature extraction.
	cal_ARI
		Whether to test the result of feature extraction
	scale
		Image-to-Coordinate scale.
	load
		Whether to use the local model.
	border
		Add the length of the black frame to the image to
		avoid the problem that the edge spot cannot be cut.
	"""
	set_seed(0)
	if scale_file==None :
		if image_type=="full":
			scale=scale
		else:
			raise Exception('Please enter scale_file address')
	else:
		assert (image_type in ['hires', 'lowres'])
		f = open(scale_file, 'r')
		content = f.read()
		a = json.loads(content)
		f.close()
		scale = a["tissue_{}_scalef".format(image_type)]
	print("Loading image data...")

	input_dir = os.path.join(image_adress, image_file)
	image = cv2.imread(input_dir)
	if border !=0:
		image= cv2.copyMakeBorder(image, border, border, border, border, cv2.BORDER_CONSTANT, value=[255, 255, 255])
		positionsdata=positionsdata+border
	# 	print('finish saving')
	tiling(image,scale, positionsdata, crop_size,image_adress=image_adress)

	model = extract_model(image_net, crop_size, patch_size)
	# torch.save(model.state_dict(), "Data/model/extract_ViT.pkl")
	if load:
		print("Load the {} model to extract tissue image features".format(image_net))
		model .load_state_dict(torch.load("Data/model/extract_{}.pkl".format(image_net)))

	model.eval()
	tillingPath = os.path.join(image_adress, "tem_img")

	# data prepare
	total_data = CustomDataset(imgs_path=tillingPath,
							   transform=test_transform)

	total_loader = DataLoader(total_data, batch_size_lw, shuffle=False,
							  pin_memory=True, drop_last=False)

	total_bar = tqdm(total_loader,desc="extract representation by "+image_net)
	feature_dim = []
	barcode = []
	f_num=0

	for image, image_code in total_bar:
		image = image.cuda(non_blocking=True)
		feature = model(image)
		f_num=feature.size()[1]
		feature_dim.append(feature.data.cpu().numpy())
		barcode.append(image_code)

	feature_dim = np.concatenate(feature_dim)
	barcode = np.concatenate(barcode)
	filder=score_adress
	if not os.path.exists(filder):
		os.makedirs(filder)

	if pca_num !=0:
		print("PCA dimensionality reduction")
		pca = PCA(n_components=pca_num)
		pca.fit(feature_dim)
		pca_feature = pca.transform(feature_dim)

		save_fileName = '{}/{}_representation.csv'.format(filder, image_net)
		data_frame = pd.DataFrame(data=feature_dim, index=barcode, columns=list(range(1, f_num + 1)))
		data_frame = data_frame.sort_index(axis=0)
		data_frame.to_csv(save_fileName)
		print("Image features are stored in: ",save_fileName)


	save_fileName = '{}/{}_pca_representation.csv'.format(filder,image_net)
	pca_frame = pd.DataFrame(data=pca_feature , index=barcode, columns=list(range(1, pca_num+1)))
	pca_frame = pca_frame.sort_index(axis=0)
	pca_frame.to_csv(save_fileName)
	print("PCA Image features are stored in: ", save_fileName)

	if (label is not None) & cal_ARI:
		print("Clustering image features using kmeans")
		if pca !=0:
			pca_kmeans = KMeans(n_clusters=k, n_init=20).fit(feature_dim)
			pcaARI = adjusted_rand_score(label, pca_kmeans.labels_)
		else:
			pcaARI=0
		kmeans = KMeans(n_clusters=k, n_init=20).fit(feature_dim)
		ARI = adjusted_rand_score(label, kmeans.labels_)

		print(ARI,pcaARI,k)
	print("Complete the extraction of image representation")
