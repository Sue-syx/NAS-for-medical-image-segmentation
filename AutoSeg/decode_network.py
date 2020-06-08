import torch
import torch.nn as nn
import torch.nn.functional as F
from genotypes import PRIMITIVES
from operations import *




class Op(nn.Module):
	def __init__(self, C, stride, idx):
		super(Op, self).__init__()
		self._ops = nn.ModuleList()
		op = OPS[PRIMITIVES[idx]](C, stride, False)
		if 'pool' in PRIMITIVES[idx]:
			op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
		self._ops.append(op)

	def forward(self, x):

		x = self._ops[0](x)

		return x



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
	def __init__(self, steps, filter_multiplier, block_multiplier, cell_arch):
		super(SameCell,self).__init__()
		self.C_in = int(filter_multiplier * steps)
		self.C_out = filter_multiplier

		self.preprocess = ReLUConvBN(self.C_in, self.C_out, 1, 1, 0, affine=False)
		self._steps = steps
		self._ops = nn.ModuleList()
		self.cell_arch = cell_arch

		stride = 1
		for i in range(self._steps):
			op = Op(self.C_out,stride,self.cell_arch[i][1])
			self._ops.append(op)

		self._initialize_weights()

	def forward(self, x):
		x = self.preprocess(x)

		states = [x]
		for i in range(self._steps):
			new_state = self._ops[i](states[self.cell_arch[i][0]])
			states.append(new_state)

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
	def __init__(self, steps, filter_multiplier, block_multiplier, cell_arch):
		super(UpCell, self).__init__()
		self.C_in = int(filter_multiplier * steps * 2)
		self.C_out = filter_multiplier

		self.preprocess = ReLUConvBN(self.C_in, self.C_out, 1, 1, 0, affine=False)
		self._steps = steps
		self._ops = nn.ModuleList()
		self.cell_arch = cell_arch

		stride = 1
		for i in range(self._steps):
			op = Op(self.C_out,stride,self.cell_arch[i][1])
			self._ops.append(op)

		self._initialize_weights()


	def forward(self, x):
		x = prev_feature_resize(x,'up')
		x = self.preprocess(x)

		states = [x]
		for i in range(self._steps):
			new_state = self._ops[i](states[self.cell_arch[i][0]])
			states.append(new_state)

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
	def __init__(self, steps, filter_multiplier, block_multiplier, cell_arch):
		super(DownCell,self).__init__()
		self.C_in = int(filter_multiplier * steps * 0.5)
		self.C_out = filter_multiplier

		self.preprocess = ReLUConvBN(self.C_in, self.C_out, 1, 1, 0, affine=False)
		self._steps = steps
		self._ops = nn.ModuleList()
		self.cell_arch = cell_arch

		stride = 1
		for i in range(self._steps):
			op = Op(self.C_out,stride,self.cell_arch[i][1])
			self._ops.append(op)

		self._initialize_weights()


	def forward(self, x):
		x = prev_feature_resize(x,'down')
		x = self.preprocess(x)

		states = [x]
		for i in range(self._steps):
			new_state = self._ops[i](states[self.cell_arch[i][0]])
			states.append(new_state)

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




class AutoNet(nn.Module):
	def __init__(self,num_layers,steps,filter_multiplier,block_multiplier,num_class,route_atch,up_cell_arch,same_cell_arch,down_cell_arch):
		super(AutoNet,self).__init__()
		self.cells = nn.ModuleList()
		self._steps = steps
		self._block_multiplier = block_multiplier
		self._filter_multiplier = filter_multiplier
		self._num_layers = num_layers
		self.num_class = num_class
		self.route_atch = route_atch

		assert num_layers == len(route_atch), "config num_layers not correspond with route_atch"

		# 根据输入图片的大小改变下面stride的大小
		C_out = self._steps * self._filter_multiplier
		self.stem = nn.Sequential(
				nn.Conv2d(1, C_out, 3, stride=2, padding=1),
				nn.BatchNorm2d(C_out),
			)

		last_block_multiplier = self.route_atch[-1][1]
		pad_d = int(3 * (2**(3-last_block_multiplier)))
		self.aspp = nn.Sequential(
			ASPP(self._filter_multiplier * (2**last_block_multiplier) * self._steps, self.num_class, pad_d, pad_d) 
			)
		
		# 0: \> cell;   1: skip-connect;   2: -> cell;   3: /> cell;
		for i in range(self._num_layers):
			if route_atch[i][-1] == 0:
				self.cells.append(DownCell(self._steps, self._filter_multiplier*(2**route_atch[i][1]), self._block_multiplier, down_cell_arch))
			elif route_atch[i][-1] == 1:
				self.cells.append(Identity())
			elif route_atch[i][-1] == 2:
				self.cells.append(SameCell(self._steps, self._filter_multiplier*(2**route_atch[i][1]), self._block_multiplier, same_cell_arch))
			elif route_atch[i][-1] == 3:
				self.cells.append(UpCell(self._steps, self._filter_multiplier*(2**route_atch[i][1]), self._block_multiplier, up_cell_arch))


	def forward(self, x):
		x0 = self.stem(x)
		for i in range(self._num_layers):
			x0 = self.cells[i](x0)

		aspp_result = self.aspp(x0)
		upsample = nn.Upsample(size=x.size()[2:], mode='bilinear', align_corners=True)
		result = upsample(aspp_result)

		return result 

