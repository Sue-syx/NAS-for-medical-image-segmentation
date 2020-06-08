import os
import torch
import time
import numpy as np
from train_config import config
from lr_scheduler import LR_Scheduler
import torch.nn.functional as F
from decode_network import AutoNet
from data_loader import ImageFolder
from evaluation import *
from chaos import CHAOS_CT
from metrics import *


os.environ["CUDA_VISIBLE_DEVICES"] = config['CUDA_VISIBLE_DEVICES']

# dataset
train_dataset = CHAOS_CT(root='./dataset/CHAOS/Train_Sets/CT/', image_size = 512, mode='train')
valid_dataset = CHAOS_CT(root='./dataset/CHAOS/Train_Sets/CT/', image_size = 512, mode='valid')


num_train = len(train_dataset)
indices = list(range(num_train))
split = int(np.floor(0.8 * num_train))


train_queue = torch.utils.data.DataLoader(
		train_dataset, batch_size=config['batch_size'],
		sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
		pin_memory=True, num_workers=4)

valid_queue = torch.utils.data.DataLoader(
		valid_dataset, batch_size=config['test_batch_size'],
		sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
		pin_memory=True, num_workers=4)



model = AutoNet(num_layers=config['num_layers'],steps=config['steps'],filter_multiplier=config['filter_multiplier'],block_multiplier=config['block_multiplier'],num_class=config['num_class'],
		route_atch=config['route_atch'],up_cell_arch=config['up_cell_arch'],same_cell_arch=config['same_cell_arch'],down_cell_arch=config['down_cell_arch'])


optimizer = torch.optim.SGD(
			model.parameters(),
			config['lr'],
			momentum=config['momentum'],
			weight_decay=config['weight_decay']
			)

scheduler = LR_Scheduler(config['lr_scheduler'], config['lr'],
							config['epochs'], len(train_queue), min_lr=config['min_lr'])


model = model.cuda()

best_pred = 0.0

def train(epoch):
	model.train()
	for batch_idx,(images,target) in enumerate(train_queue):
		images, target = images.cuda(), target.cuda()
		scheduler(optimizer, batch_idx, epoch, best_pred)
		optimizer.zero_grad()
		output = model(images)
		loss = F.cross_entropy(output,target)
		loss.backward()
		optimizer.step()
		if batch_idx % 200 == 0:
			print("train loss:",loss)


def validation(epoch):
	metric_val = SegmentationMetric(config['num_class'])
	metric_val.reset()
	model.eval()

	for batch_idx, (images,target) in enumerate(valid_queue):
		images, target = images.cuda(), target.cuda()
		output = model(images)
		metrics_val.update(target,output)

	pixAcc, mIoU = metric_val.get()
	print("pixAcc:{}, mIoU:{}".format(pixAcc,mIoU))	


def save_checkpoint(state):
	torch.save(state, 'checkpoint0.pth.tar')

for epoch in range(0,config['epochs']):
	train(epoch)
	validation(epoch)
	save_checkpoint({
		'epoch': epoch + 1,
		'state_dict': model.state_dict()
	})