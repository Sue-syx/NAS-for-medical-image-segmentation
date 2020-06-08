import os
import torch
import time
import numpy as np
from search_config import config
#from loss import SegmentationLosses
from lr_scheduler import LR_Scheduler
import torch.nn.functional as F
from network import AutoNet
from data_loader import ImageFolder
from evaluation import *
from chaos import CHAOS_CT
from metrics import *

#os.environ["CUDA_VISIBLE_DEVICES"] = config['CUDA_VISIBLE_DEVICES']
device = torch.device("cuda:2")

# dataset
# train_dataset = ImageFolder(root = config['train_root'], image_size = 128, mode='train',augmentation_prob=config['augmentation_prob'])
train_dataset = CHAOS_CT(root='./dataset/CHAOS/Train_Sets/CT/', image_size = 512, mode='train')

num_train = len(train_dataset)
indices = list(range(num_train))
split = int(np.floor(0.5 * num_train))

train_queue = torch.utils.data.DataLoader(
		train_dataset, batch_size=config['batch_size'],
		sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
		pin_memory=True, num_workers=4)

valid_queue = torch.utils.data.DataLoader(
		train_dataset, batch_size=config['test_batch_size'],
		sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
		pin_memory=True, num_workers=4)

#criterion = SegmentationLosses(weight=None, cuda=config['cuda']).build_loss(mode=config['loss_type'])
model = AutoNet(num_layers=config['num_layers'],filter_multiplier=config['filter_multiplier'],block_multiplier=config['block_multiplier'],num_class=config['num_class'])

optimizer = torch.optim.SGD(
			model.weight_parameters(),
			config['lr'],
			momentum=config['momentum'],
			weight_decay=config['weight_decay']
			)

architect_optimizer = torch.optim.Adam(model.arch_parameters(),
										lr=config['arch_lr'], betas=(0.9, 0.999),
										weight_decay=config['arch_weight_decay'])

scheduler = LR_Scheduler(config['lr_scheduler'], config['lr'],
							config['epochs'], len(train_queue), min_lr=config['min_lr'])

#model = model.cuda()
model.to(device)

best_pred = 0.0

def train(epoch):
	model.cuda()
	model.train()
	for batch_idx,(images,target) in enumerate(train_queue):
		images,target = images.cuda(), target.cuda()
		scheduler(optimizer, batch_idx, epoch, best_pred)
		optimizer.zero_grad()
		model.zero_grad()
		output = model(images)
		#loss = criterion(output, target)
		#loss.backward(retain_graph=True)
		loss = F.cross_entropy(output,target)
		loss.backward()
		optimizer.step()
		if batch_idx % 200 == 0:
			print("loss:",loss)


		if epoch >= config['alpha_epoch']:
			images_search,target_search = next(iter(valid_queue))
			images_search,target_search = images_search.cuda(),target_search.cuda()
			scheduler(architect_optimizer, batch_idx, epoch, best_pred)
			architect_optimizer.zero_grad()
			output_search = model(images_search)
			arch_loss = F.cross_entropy(output_search,target_search)
			arch_loss.backward()
			architect_optimizer.step()


def validation(epoch):
	metric_val = SegmentationMetric(config['num_class'])
	metric_val.reset()
	model.eval()
	model.to(device)

	for batch_idx, (images,GT) in enumerate(valid_queue):
		#images,GT = images.cuda(),GT.cuda()
		images,GT = images.to(device),GT.to(device)
		output = model(images)
		metric_val.update(GT,output)
		

	pixAcc, mIoU = metric_val.get()
	print("pixAcc:{}, mIoU:{}".format(pixAcc,mIoU))


def save_checkpoint(state):
	torch.save(state, 'checkpoint.pth.tar')

checkpoint = torch.load('checkpoint.pth.tar')

model.load_state_dict(checkpoint['state_dict'])
start_epoch = checkpoint['epoch']
for epoch in range(start_epoch,config['epochs']):
	train(epoch)
	#validation(epoch)
	save_checkpoint({
		'epoch': epoch + 1,
		'state_dict': model.state_dict()
	})	


