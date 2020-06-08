import torch
import torch.nn as nn
from config import config
from network import AutoNet
import torch.nn.functional as F

'''
这个文件用来找到最终的网络结构是什么样子的
'''

#model = AutoNet(num_layers=config['num_layers'],filter_multiplier=config['filter_multiplier'],block_multiplier=config['block_multiplier'],num_class=config['num_class'])

checkpoint = torch.load('checkpoint.pth.tar', map_location=torch.device('cpu'))


# 获得网络结构的三种类型的参数
alphas_up = checkpoint['state_dict']['alphas_up']
alphas_same = checkpoint['state_dict']['alphas_same']
alphas_down = checkpoint['state_dict']['alphas_down']
betas = checkpoint['state_dict']['betas']
gamma_up = checkpoint['state_dict']['gamma_up']
gamma_same = checkpoint['state_dict']['gamma_same']
gamma_down = checkpoint['state_dict']['gamma_down']


# 顶层的beta参数
num_layers = betas.shape[0]
block_multiplier = betas.shape[1]
conncect_type = betas.shape[2]

assert num_layers == config['num_layers'], 'config arg num_layers!= model num_layers'
assert block_multiplier == config['block_multiplier'], 'config arg block_multiplier != model block_multiplier'
assert conncect_type == 4, 'conncect_type num is not 4'
steps = config['steps']

for layer in range(num_layers):
	if layer == 0:
		betas[layer,0,0] = torch.zeros(1)
		betas[layer,0,1:] = F.softmax(betas[layer,0,1:], dim=-1) * (3/4)
		betas[layer,1:,:] = torch.zeros(4,4)
	elif layer == 1:
		betas[layer,0,0] = torch.zeros(1)
		betas[layer,0,1:] = F.softmax(betas[layer,0,1:], dim=-1) * (3/4)
		betas[layer,1,:] = F.softmax(betas[layer,1,:], dim=-1)
		betas[layer,2:,:] = torch.zeros(3,4)
	elif layer == 2:
		betas[layer,0,0] = torch.zeros(1)
		betas[layer,0,1:] = F.softmax(betas[layer,0,1:], dim=-1) * (3/4)
		betas[layer,1,:] = F.softmax(betas[layer,1,:], dim=-1)
		betas[layer,2,:] = F.softmax(betas[layer,2,:], dim=-1)
		betas[layer,3:,:] = torch.zeros(2,4)
	elif layer == 3:
		betas[layer,0,0] = torch.zeros(1)
		betas[layer,0,1:] = F.softmax(betas[layer,0,1:], dim=-1) * (3/4)
		betas[layer,1,:] = F.softmax(betas[layer,1,:], dim=-1)
		betas[layer,2,:] = F.softmax(betas[layer,2,:], dim=-1)
		betas[layer,3,:] = F.softmax(betas[layer,3,:], dim=-1)
		betas[layer,4,:] = torch.zeros(4)
	else:
		betas[layer,0,0] = torch.zeros(1)
		betas[layer,0,1:] = F.softmax(betas[layer,0,1:], dim=-1) * (3/4)
		betas[layer,1,:] = F.softmax(betas[layer,1,:], dim=-1)
		betas[layer,2,:] = F.softmax(betas[layer,2,:], dim=-1)
		betas[layer,3,:] = F.softmax(betas[layer,3,:], dim=-1)
		betas[layer,4,:3] = F.softmax(betas[layer,4,:3],dim=-1) * (3/4)
		betas[layer,4,-1] = torch.zeros(1)



# 0: /> cell
# 1: skip-connect 
# 2: -> cell 
# 3: \> cell
'''
dis = torch.zeros(block_multiplier)
route = [[] for i in  range(block_multiplier)]
for i in range(num_layers):
	dis_n = torch.zeros(block_multiplier)
	new_route = [[] for m in  range(block_multiplier)]
	for j in range(block_multiplier):
		if j == 0:
			d = torch.tensor([0, dis[0]+betas[i,j,1], dis[0]+betas[i,j,2], dis[1]+betas[i,j+1,0]])
			max_id = torch.argmax(d)
			dis_n[j] = d[max_id]
		elif j == 4:
			d = torch.tensor([dis[-2]+betas[i,j-1,3], dis[-1]+betas[i,j,1], dis[-1]+betas[i,j,2], 0])
			max_id = torch.argmax(d)
			dis_n[j] = d[max_id]
		else:
			d = torch.tensor([dis[j-1]+betas[i,j-1,3], dis[j]+betas[i,j,1], dis[j]+betas[i,j,2], dis[j+1]+betas[i,j+1,0]])
			max_id = torch.argmax(d)
			dis_n[j] = d[max_id]

		if (max_id > 2) and (j == block_multiplier - 1):
			new_route[j].append([])
		else:
			new_route[j] += route[j+(max_id.item()-1)//2]
			new_route[j].append([i,j,max_id.item()])
	dis = dis_n
	route = new_route

'''

'''
N = 2
dis = torch.zeros(block_multiplier,N)
route = [[[],[]] for i in  range(block_multiplier)]
for i in range(num_layers):
	dis_n = torch.zeros(block_multiplier,N)
	new_route = [[[],[]] for m in  range(block_multiplier)]
	for j in range(block_multiplier):
		if j == 0:
			d = torch.tensor([0, 0, dis[0,0]+betas[i,j,1], dis[0,1]+betas[i,j,1], 
							dis[0,0]+betas[i,j,2], dis[0,1]+betas[i,j,2], dis[1,0]+betas[i,j+1,0], dis[1,1]+betas[i,j+1,0]])
		elif j == 4:
			d = torch.tensor([dis[-2,0]+betas[i,j-1,3], dis[-2,1]+betas[i,j-1,3], dis[-1,0]+betas[i,j,1], dis[-1,1]+betas[i,j,1],
								dis[-1,0]+betas[i,j,2], dis[-1,1]+betas[i,j,2], 0, 0])
		else:
			d = torch.tensor([dis[j-1,0]+betas[i,j-1,3], dis[j-1,1]+betas[i,j-1,3], dis[j,0]+betas[i,j,1], dis[j,1]+betas[i,j,1],
								dis[j,0]+betas[i,j,2], dis[j,1]+betas[i,j,2], dis[j+1,0]+betas[i,j+1,0], dis[j+1,1]+betas[i,j+1,0]])
		max_id0 = torch.argmax(d)
		dis_n[j,0] = d[max_id0]
		d[d==dis_n[j,0]] = 0
		max_id1 = torch.argmax(d)
		dis_n[j,1] = d[max_id1]
		if (max_id0 > 5) and (j == block_multiplier - 1):
			new_route[j][0].append([])
		if (max_id1 > 5) and (j == block_multiplier - 1):
			new_route[j][1].append([])
		else:
			new_route[j][0] += route[j+(max_id0.item()//2-1)//2][max_id0.item()%2]
			new_route[j][1] += route[j+(max_id1.item()//2-1)//2][max_id1.item()%2]
			new_route[j][0].append([i,j,max_id0.item()//2])
			new_route[j][1].append([i,j,max_id1.item()//2])
			#print("jLine:{},max_id0.item():{},new_route0:{}".format(j,max_id0.item(),new_route[j][0]))
			#print("jline:{},max_id1.item():{},new_route1:{}".format(j,max_id1.item(),new_route[j][1]))
	dis = dis_n
	route = new_route

print(dis)
for i in range(5):
	for j in range(2):
		print(route[i][j])
'''

print(betas)

N = 4
dis = torch.zeros(block_multiplier,N)
route = [[[] for w in range(N)] for i in  range(block_multiplier)]
for i in range(num_layers):
	dis_n = torch.zeros(block_multiplier,N)
	new_route = [[[] for w in range(N)] for i in  range(block_multiplier)]
	for j in range(block_multiplier):
		if j == 0:
			list1 = [0 for w in range(N)]
			list2 = [dis[0,w]+betas[i,j,1] for w in range(N)]
			list3 = [dis[0,w]+betas[i,j,2] for w in range(N)]
			list4 = [dis[1,w]+betas[i,j+1,0] for w in range(N)]
			d = torch.tensor(list1+list2+list3+list4)
		elif j == 4:
			list1 = [dis[-2,w]+betas[i,j-1,3] for w in range(N)]
			list2 = [dis[-1,w]+betas[i,j,1] for w in range(N)]
			list3 = [dis[-1,w]+betas[i,j,2] for w in range(N)]
			list4 = [0 for i in range(N)]
			d = torch.tensor(list1+list2+list3+list4)
		else:
			list1 = [dis[j-1,w]+betas[i,j-1,3] for w in range(N)]
			list2 = [dis[j,w]+betas[i,j,1] for w in range(N)]
			list3 = [dis[j,w]+betas[i,j,2] for w in range(N)]
			list4 = [dis[j+1,w]+betas[i,j+1,0] for w in range(N)]
			d = torch.tensor(list1+list2+list3+list4)
		max_id = []
		for w in range(N):
			idx = torch.argmax(d)
			dis_n[j,w] = d[idx]
			d[d==dis_n[j,w]] = 0
			max_id.append(idx)

		for w in range(N):
			if (max_id[w] > 5) and (j == block_multiplier - 1):
				new_route[j][w].append([])
			else:
				new_route[j][w] += route[j+(max_id[w].item()//N-1)//2][max_id[w].item()%N]
				new_route[j][w].append([i,j,max_id[w].item()//N])
	dis = dis_n
	route = new_route

print(dis)
for i in range(5):
	for j in range(N):
		print(route[i][j])






# 涉及到cell结构的 gamma 和 alpha参数，这两个参数是乘在一起的

print(alphas_up)
alphas_up = F.softmax(alphas_up,dim=-1)
alphas_same = F.softmax(alphas_same,dim=-1)
alphas_down = F.softmax(alphas_down,dim=-1)

offset = 0
for i in range(steps):
	gamma_up[offset:offset+i+1] = F.softmax(gamma_up[offset:offset+i+1],dim=-1)
	gamma_same[offset:offset+i+1] = F.softmax(gamma_same[offset:offset+i+1],dim=-1)
	gamma_down[offset:offset+i+1] = F.softmax(gamma_down[offset:offset+i+1],dim=-1)
	offset += (i+1)

alphas_up = alphas_up.transpose(0,1) * gamma_up
alphas_up = alphas_up.transpose(0,1)

alphas_same = alphas_same.transpose(0,1) * gamma_same
alphas_same = alphas_same.transpose(0,1)

alphas_down = alphas_down.transpose(0,1) * gamma_down
alphas_down = alphas_down.transpose(0,1)



up_cell = []
offset = 0
for i in range(steps):
	max_id = torch.argmax(alphas_up[offset:offset+i+1,:])
	up_cell.append([max_id.item()//8, max_id.item()%8])
	offset += (i+1)
print(up_cell)


same_cell = []
offset = 0
for i in range(steps):
	max_id = torch.argmax(alphas_same[offset:offset+i+1,:])
	same_cell.append([max_id.item()//8, max_id.item()%8])
	offset += (i+1)
print(same_cell)


down_cell = []
offset = 0
for i in range(steps):
	max_id = torch.argmax(alphas_down[offset:offset+i+1,:])
	down_cell.append([max_id.item()//8, max_id.item()%8])
	offset += (i+1)
print(down_cell)
