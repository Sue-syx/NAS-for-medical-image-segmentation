import torch.nn.functional as F
from operations import *
import torch.nn as nn
from genotypes import PRIMITIVES
import time


def channel_shuffle(x, K):
	batchsize, num_channels, height, width = x.data.size()

	channels_per_group = num_channels // K

	#reshape
	x = x.reshape(batchsize, K, channels_per_group, height, width)
	
	x = torch.transpose(x, 1, 2).contiguous()
	# flatten
	x = x.reshape(batchsize, -1, height, width)

	return x


class MixedOp(nn.Module):
	def __init__(self, C, stride):
		super(MixedOp, self).__init__()
		self.K = 8
		self._ops = nn.ModuleList()
		for primitive in PRIMITIVES:
			op = OPS[primitive](C//self.K, stride, False)
			if 'pool' in primitive:
				op = nn.Sequential(op, nn.BatchNorm2d(C//self.K, affine=False))
			self._ops.append(op)

	def forward(self, x, weights):
		C = x.shape[1]
		xtemp = x[:, :C//self.K, :, :]
		xtemp2 = x[:, C//self.K:, :, :]
		xtemp1 = sum(w * op(xtemp) for w, op in zip(weights, self._ops))

		y = torch.cat([xtemp1,xtemp2],dim=1)
		y = channel_shuffle(y,self.K)

		return y


def scale_dimension(dim, scale):
	assert isinstance(dim, int)
	return int((float(dim) - 1.0) * scale + 1.0) if dim % 2 else int(dim * scale)


def prev_feature_resize(prev_feature, mode):
    if mode == 'down':
        feature_size_h = scale_dimension(prev_feature.shape[2], 0.5)
        feature_size_w = scale_dimension(prev_feature.shape[3], 0.5)
    elif mode == 'up':
        feature_size_h = scale_dimension(prev_feature.shape[2], 2)
        feature_size_w = scale_dimension(prev_feature.shape[3], 2)

    return F.interpolate(prev_feature, (feature_size_h, feature_size_w), mode='bilinear', align_corners=True)


class SameCell(nn.Module):
	def __init__(self, steps, filter_multiplier, block_multiplier):
		super(SameCell,self).__init__()
		self.C_in = int(filter_multiplier * steps)
		self.C_out = filter_multiplier

		self.preprocess = ReLUConvBN(self.C_in, self.C_out, 1, 1, 0, affine=False)
		self._steps = steps
		self._ops = nn.ModuleList()

		stride = 1
		for i in range(self._steps):
			for j in range(1 + i):
				op = MixedOp(self.C_out,stride)
				self._ops.append(op)

		self._initialize_weights()

	def forward(self, x, n_alphas, n_gamma):
		x = self.preprocess(x)

		states = [x]
		offset = 0
		for i in range(self._steps):
			new_states = []
			for j,h in enumerate(states):
				branch_index = offset + j
				# 以防前向计算时，搜索出来的opration是None
				if self._ops[branch_index] is None:
					continue
				# 下面h是输入的feature map,n_alphas[branch_index]为对应各opration的权重
				new_state = self._ops[branch_index](h,n_alphas[branch_index]) * n_gamma[branch_index]
				new_states.append(new_state)

			# 中间节点
			s = sum(new_states)
			# 索引偏置
			# offset += len(states)
			offset += (i+1)
			states.append(s)

		result = torch.cat(states[-self._steps:],dim=1)

		return result

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				torch.nn.init.kaiming_normal_(m.weight)	
			elif isinstance(m, nn.BatchNorm2d):
				if m.weight is not None:
					m.weight.data.fill_(1)
					m.bias.data.zero_()



class UpCell(nn.Module):
	def __init__(self, steps, filter_multiplier, block_multiplier):
		super(UpCell,self).__init__()
		self.C_in = int(filter_multiplier * steps * 2)
		self.C_out = filter_multiplier

		self.preprocess = ReLUConvBN(self.C_in, self.C_out, 1, 1, 0, affine=False)
		self._steps = steps
		self._ops = nn.ModuleList()

		stride = 1
		for i in range(self._steps):
			for j in range(1 + i):
				op = MixedOp(self.C_out,stride)
				self._ops.append(op)

		self._initialize_weights()


	def forward(self,x,n_alphas,n_gamma):
		x = prev_feature_resize(x,'up')
		x = self.preprocess(x)

		states = [x]
		offset = 0
		for i in range(self._steps):
			new_states = []
			for j,h in enumerate(states):
				branch_index = offset + j
				# 以防前向计算时，搜索出来的opration是None
				if self._ops[branch_index] is None:
					continue
				# 下面h是输入的feature map,n_alphas[branch_index]为对应各opration的权重
				new_state = self._ops[branch_index](h,n_alphas[branch_index]) * n_gamma[branch_index]
				new_states.append(new_state)

			# 中间节点
			s = sum(new_states)
			# 索引偏置
			# offset += len(states)
			offset += (i+1)
			states.append(s)

		result = torch.cat(states[-self._steps:],dim=1)

		return result


	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				torch.nn.init.kaiming_normal_(m.weight)	
			elif isinstance(m, nn.BatchNorm2d):
				if m.weight is not None:
					m.weight.data.fill_(1)
					m.bias.data.zero_()



class DownCell(nn.Module):
	def __init__(self, steps, filter_multiplier, block_multiplier):
		super(DownCell,self).__init__()
		self.C_in = int(filter_multiplier * steps * 0.5)
		self.C_out = filter_multiplier

		self.preprocess = ReLUConvBN(self.C_in, self.C_out, 1, 1, 0, affine=False)
		self._steps = steps
		self._ops = nn.ModuleList()

		stride = 1
		for i in range(self._steps):
			for j in range(1 + i):		
				op = MixedOp(self.C_out,stride)
				self._ops.append(op)

		self._initialize_weights()


	def forward(self,x,n_alphas,n_gamma):
		x = prev_feature_resize(x,'down')
		x = self.preprocess(x)

		states = [x]
		offset = 0
		for i in range(self._steps):
			new_states = []
			for j,h in enumerate(states):
				branch_index = offset + j
				# 以防前向计算时，搜索出来的opration是None
				if self._ops[branch_index] is None:
					continue
				# 下面h是输入的feature map,n_alphas[branch_index]为对应各opration的权重
				new_state = self._ops[branch_index](h,n_alphas[branch_index]) * n_gamma[branch_index]
				new_states.append(new_state)

			# 中间节点
			s = sum(new_states)

			# 索引偏置
			# offset += len(states)
			offset += (i+1)
			states.append(s)

		result = torch.cat(states[-self._steps:],dim=1)

		return result


	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				torch.nn.init.kaiming_normal_(m.weight)	
			elif isinstance(m, nn.BatchNorm2d):
				if m.weight is not None:
					m.weight.data.fill_(1)
					m.bias.data.zero_()



class Cell(nn.Module):
	def __init__(self, steps, filter_multiplier, block_multiplier):
		super(Cell,self).__init__()
		self.up_cell = UpCell(steps, filter_multiplier, block_multiplier)
		self.same_cell = SameCell(steps, filter_multiplier, block_multiplier)
		self.down_cell = DownCell(steps, filter_multiplier, block_multiplier)

	def forward(self,x_up,x_same,x_down,alphas_up,alphas_same,alphas_down,beta0,beta1,beta2,beta3,gamma_up,gamma_same,gamma_down):
		fsum = 0

		if x_up is not None:
			x_up = self.up_cell(x_up,alphas_up,gamma_up)
			fsum += x_up * beta3

		if x_same is not None:
			x = x_same
			x_same = self.same_cell(x_same,alphas_same,gamma_same)
			fsum += (x * beta1 + x_same * beta2)

		if x_down is not None:
			x_down = self.down_cell(x_down,alphas_down,gamma_down)
			fsum += x_down * beta0

		return fsum
