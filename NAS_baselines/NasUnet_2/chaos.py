import os
import pydicom
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from util.augmentations import *
from torch.utils import data



def make_dataset(root, dirname, type='CT', is_dup=False):
    base_path = os.path.join(root, dirname)
    images_path = os.path.join(root, dirname, 'DICOM_anon')
    mask_path = os.path.join(root, dirname, 'Ground')

    images = os.listdir(images_path)
    images_list = []
    for image_name in images:
        if type == 'CT': # two types batch
            if 'IMG' in image_name:
                image_mask_name = 'liver_GT_' + image_name[:-4].split('-')[-1][2:] + '.png'
            else:
                image_mask_name = 'liver_GT_' + image_name[:-4].split(',')[0][2:] + '.png'
        else:
            M = image_name[:-4].split('-')[-1]
            id = "%03d" % ((int(M)+1) // 2) if is_dup else M[2:]
            image_mask_name = 'liver_' + id + '.png'
        img_path = os.path.join(images_path, image_name)
        img_mask_path = os.path.join(mask_path, image_mask_name)
        images_list.append((img_path, img_mask_path))

    return images_list



class CHAOS_CT(torch.utils.data.Dataset):
	def __init__(self, root, image_size, mode="train", norm={'mu':[0.2389], 'std':[0.2801]}):
		self.mode = mode
		self.datalist = []
		dirnames = os.listdir(root)
		for dir_ in dirnames:
			if dir_ == 'notes.txt':
				continue
			self.datalist += make_dataset(root, dir_)
		if len(self.datalist) == 0:
			raise (RuntimeError("Found 0 images in subfolders of: " + root))

		self.train_transform = Compose([
			RandomTranslate(offset=(0.3, 0.3)),
			RandomVerticallyFlip(),
			RandomHorizontallyFlip(),
			RandomElasticTransform(alpha=1.5, sigma=0.07),
			RandomSizedCrop(size=image_size),
			ToTensor()
			])
		self.test_transform = Compose([
			Resize(size=image_size),
			ToTensor()
			])
		self.img_normalize = transforms.Normalize(norm['mu'], norm['std'])

	def __getitem__(self,index):
		img_path, target_path = self.datalist[index][0], self.datalist[index][1]
		# read image
		CT_info = pydicom.dcmread(img_path)
		target = Image.open(target_path).convert('L')

		if 'PixelData' in CT_info:
			img, itercept = CT_info.RescaleSlope * CT_info.pixel_array + CT_info.RescaleIntercept, CT_info.RescaleIntercept
			img[img >= 4000] = itercept
			img = Image.fromarray(img).convert('L')

		if self.mode == 'train':
			img, target = self.train_transform(img, target)
		else:
			img, target = self.test_transform(img, target)

		img = self.img_normalize(img)
		target[target == 255] = 1

		return img, target

	def __len__(self):
		return len(self.datalist)



if __name__ == '__main__':
	train_root = './Train_Sets/CT/'
	train_dataset = CHAOS_CT(root=train_root,image_size = 512, mode='train')
	kwargs = {'num_workers': 1, 'pin_memory': True}
	train_dataloader = data.DataLoader(train_dataset,batch_size=1,**kwargs)
	print(len(train_dataloader))
	'''
	for batch_idx, (img,target) in enumerate(train_dataloader):
		target = target.view(-1)
	'''

