import torch.nn as nn
import cell
import torch.nn.functional as F
from operations import *
from genotypes import PRIMITIVES
import time
from config import config


class AutoNet(nn.Module):
    def __init__(self, num_layers, steps=4, filter_multiplier=8, block_multiplier=5, num_class=2):
        super(AutoNet, self).__init__()

        self.cells = nn.ModuleList()
        self._steps = steps
        self._block_multiplier = block_multiplier
        self._filter_multiplier = filter_multiplier
        self._num_layers = num_layers
        self.num_class = num_class
        # 网络的结构权重
        self._initialize_alphas_betas()

        # 根据输入图片的大小改变下面stride的大小
        C_out = self._steps * self._filter_multiplier
        self.stem = nn.Sequential(
            nn.Conv2d(1, C_out, 3, stride=2, padding=1),
            nn.BatchNorm2d(C_out),
        )

        # 2 + 3 + 4 + 5 * (9-3)
        for i in range(2, self._block_multiplier):
            for j in range(i):
                self.cells += [cell.Cell(self._steps, self._filter_multiplier * (2 ** j), self._block_multiplier)]
        for i in range(self._num_layers - self._block_multiplier + 2):
            for j in range(self._block_multiplier):
                self.cells += [cell.Cell(self._steps, self._filter_multiplier * (2 ** j), self._block_multiplier)]

        self.aspp_1 = nn.Sequential(
            ASPP(self._filter_multiplier * self._steps, self.num_class, 24, 24)
        )
        self.aspp_2 = nn.Sequential(
            ASPP(self._filter_multiplier * 2 * self._steps, self.num_class, 12, 12)
        )
        self.aspp_3 = nn.Sequential(
            ASPP(self._filter_multiplier * 4 * self._steps, self.num_class, 6, 6)
        )
        self.aspp_4 = nn.Sequential(
            ASPP(self._filter_multiplier * 8 * self._steps, self.num_class, 3, 3)
        )
        self.aspp_5 = nn.Sequential(
            ASPP(self._filter_multiplier * 16 * self._steps, self.num_class, 1, 1)
        )

    def forward(self, x):
        device = x.device

        x0 = self.stem(x)

        normalized_betas = torch.zeros([self._num_layers, self._block_multiplier, 4], device=device)
        k = sum(1 for i in range(self._steps) for n in range(1 + i))
        normalized_gamma_up = torch.zeros(k, device=device)
        normalized_gamma_same = torch.zeros(k, device=device)
        normalized_gamma_down = torch.zeros(k, device=device)

        alphas_up = F.softmax(self.alphas_up, dim=-1)
        alphas_same = F.softmax(self.alphas_same, dim=-1)
        alphas_down = F.softmax(self.alphas_down, dim=-1)
        offset = 0
        for i in range(self._steps):
            normalized_gamma_up[offset:offset + i + 1] = F.softmax(self.gamma_up[offset:offset + i + 1], dim=-1)
            normalized_gamma_same[offset:offset + i + 1] = F.softmax(self.gamma_same[offset:offset + i + 1], dim=-1)
            normalized_gamma_down[offset:offset + i + 1] = F.softmax(self.gamma_down[offset:offset + i + 1], dim=-1)
            offset += (i + 1)

        for layer in range(self._num_layers):
            if layer == 0:
                normalized_betas[layer, 0, 1:] = F.softmax(self.betas[layer, 0, 1:], dim=-1) * (3 / 4)
            elif layer == 1:
                normalized_betas[layer, 0, 1:] = F.softmax(self.betas[layer, 0, 1:], dim=-1) * (3 / 4)
                normalized_betas[layer, 1, :] = F.softmax(self.betas[layer, 1, :], dim=-1)
            elif layer == 2:
                normalized_betas[layer, 0, 1:] = F.softmax(self.betas[layer, 0, 1:], dim=-1) * (3 / 4)
                normalized_betas[layer, 1, :] = F.softmax(self.betas[layer, 1, :], dim=-1)
                normalized_betas[layer, 2, :] = F.softmax(self.betas[layer, 2, :], dim=-1)
            elif layer == 3:
                normalized_betas[layer, 0, 1:] = F.softmax(self.betas[layer, 0, 1:], dim=-1) * (3 / 4)
                normalized_betas[layer, 1, :] = F.softmax(self.betas[layer, 1, :], dim=-1)
                normalized_betas[layer, 2, :] = F.softmax(self.betas[layer, 2, :], dim=-1)
                normalized_betas[layer, 3, :] = F.softmax(self.betas[layer, 3, :], dim=-1)
            else:
                normalized_betas[layer, 0, 1:] = F.softmax(self.betas[layer, 0, 1:], dim=-1) * (3 / 4)
                normalized_betas[layer, 1, :] = F.softmax(self.betas[layer, 1, :], dim=-1)
                normalized_betas[layer, 2, :] = F.softmax(self.betas[layer, 2, :], dim=-1)
                normalized_betas[layer, 3, :] = F.softmax(self.betas[layer, 3, :], dim=-1)
                normalized_betas[layer, 4, :3] = F.softmax(self.betas[layer, 4, :3], dim=-1) * (3 / 4)

        count = 0
        for layer in range(self._num_layers):
            if layer == 0:
                level1 = self.cells[count](None, x0, None,
                                           alphas_up, alphas_same, alphas_down,
                                           0, normalized_betas[layer, 0, 1], normalized_betas[layer, 0, 2], 0,
                                           normalized_gamma_up, normalized_gamma_same, normalized_gamma_down)
                count += 1
                level2 = self.cells[count](None, None, x0,
                                           alphas_up, alphas_same, alphas_down,
                                           normalized_betas[layer, 0, 3], 0, 0, 0,
                                           normalized_gamma_up, normalized_gamma_same, normalized_gamma_down)
                count += 1

            elif layer == 1:
                new_level1 = self.cells[count](None, level1, None,
                                               alphas_up, alphas_same, alphas_down,
                                               0, normalized_betas[layer, 0, 1], normalized_betas[layer, 0, 2], 0,
                                               normalized_gamma_up, normalized_gamma_same, normalized_gamma_down)
                count += 1
                new_level2 = self.cells[count](None, level2, level1,
                                               alphas_up, alphas_same, alphas_down,
                                               normalized_betas[layer, 0, 3], normalized_betas[layer, 1, 1],
                                               normalized_betas[layer, 1, 2], 0,
                                               normalized_gamma_up, normalized_gamma_same, normalized_gamma_down)
                count += 1
                level3 = self.cells[count](None, None, level2,
                                           alphas_up, alphas_same, alphas_down,
                                           normalized_betas[layer, 1, 3], 0, 0, 0,
                                           normalized_gamma_up, normalized_gamma_same, normalized_gamma_down)
                count += 1
                level1 = new_level1
                level2 = new_level2

            elif layer == 2:
                new_level1 = self.cells[count](None, level1, None,
                                               alphas_up, alphas_same, alphas_down,
                                               0, normalized_betas[layer, 0, 1], normalized_betas[layer, 0, 2], 0,
                                               normalized_gamma_up, normalized_gamma_same, normalized_gamma_down)
                count += 1
                new_level2 = self.cells[count](level3, level2, level1,
                                               alphas_up, alphas_same, alphas_down,
                                               normalized_betas[layer, 0, 3], normalized_betas[layer, 1, 1],
                                               normalized_betas[layer, 1, 2], normalized_betas[layer, 2, 0],
                                               normalized_gamma_up, normalized_gamma_same, normalized_gamma_down)
                count += 1
                new_level3 = self.cells[count](None, level3, level2,
                                               alphas_up, alphas_same, alphas_down,
                                               normalized_betas[layer, 1, 3], normalized_betas[layer, 2, 1],
                                               normalized_betas[layer, 2, 2], 0,
                                               normalized_gamma_up, normalized_gamma_same, normalized_gamma_down)
                count += 1
                level4 = self.cells[count](None, None, level3,
                                           alphas_up, alphas_same, alphas_down,
                                           normalized_betas[layer, 2, 3], 0, 0, 0,
                                           normalized_gamma_up, normalized_gamma_same, normalized_gamma_down)
                count += 1
                level1 = new_level1
                level2 = new_level2
                level3 = new_level3

            elif layer == 3:
                new_level1 = self.cells[count](None, level1, None,
                                               alphas_up, alphas_same, alphas_down,
                                               0, normalized_betas[layer, 0, 1], normalized_betas[layer, 0, 2], 0,
                                               normalized_gamma_up, normalized_gamma_same, normalized_gamma_down)
                count += 1
                new_level2 = self.cells[count](level3, level2, level1,
                                               alphas_up, alphas_same, alphas_down,
                                               normalized_betas[layer, 0, 3], normalized_betas[layer, 1, 1],
                                               normalized_betas[layer, 1, 2], normalized_betas[layer, 2, 0],
                                               normalized_gamma_up, normalized_gamma_same, normalized_gamma_down)
                count += 1
                new_level3 = self.cells[count](level4, level3, level2,
                                               alphas_up, alphas_same, alphas_down,
                                               normalized_betas[layer, 1, 3], normalized_betas[layer, 2, 1],
                                               normalized_betas[layer, 2, 2], normalized_betas[layer, 3, 0],
                                               normalized_gamma_up, normalized_gamma_same, normalized_gamma_down)
                count += 1
                new_level4 = self.cells[count](None, level4, level3,
                                               alphas_up, alphas_same, alphas_down,
                                               normalized_betas[layer, 2, 3], normalized_betas[layer, 3, 1],
                                               normalized_betas[layer, 3, 2], 0,
                                               normalized_gamma_up, normalized_gamma_same, normalized_gamma_down)

                count += 1
                level5 = self.cells[count](None, None, level4,
                                           alphas_up, alphas_same, alphas_down,
                                           normalized_betas[layer, 3, 3], 0, 0, 0,
                                           normalized_gamma_up, normalized_gamma_same, normalized_gamma_down)
                count += 1
                level1 = new_level1
                level2 = new_level2
                level3 = new_level3
                level4 = new_level4

            else:
                new_level1 = self.cells[count](None, level1, None,
                                               alphas_up, alphas_same, alphas_down,
                                               0, normalized_betas[layer, 0, 1], normalized_betas[layer, 0, 2], 0,
                                               normalized_gamma_up, normalized_gamma_same, normalized_gamma_down)
                count += 1
                new_level2 = self.cells[count](level3, level2, level1,
                                               alphas_up, alphas_same, alphas_down,
                                               normalized_betas[layer, 0, 3], normalized_betas[layer, 1, 1],
                                               normalized_betas[layer, 1, 2], normalized_betas[layer, 2, 0],
                                               normalized_gamma_up, normalized_gamma_same, normalized_gamma_down)
                count += 1
                new_level3 = self.cells[count](level4, level3, level2,
                                               alphas_up, alphas_same, alphas_down,
                                               normalized_betas[layer, 1, 3], normalized_betas[layer, 2, 1],
                                               normalized_betas[layer, 2, 2], normalized_betas[layer, 3, 0],
                                               normalized_gamma_up, normalized_gamma_same, normalized_gamma_down)
                count += 1
                new_level4 = self.cells[count](level5, level4, level3,
                                               alphas_up, alphas_same, alphas_down,
                                               normalized_betas[layer, 2, 3], normalized_betas[layer, 3, 1],
                                               normalized_betas[layer, 3, 2], normalized_betas[layer, 3, 0],
                                               normalized_gamma_up, normalized_gamma_same, normalized_gamma_down)
                count += 1
                new_level5 = self.cells[count](None, level5, level4,
                                               alphas_up, alphas_same, alphas_down,
                                               normalized_betas[layer, 3, 3], normalized_betas[layer, 4, 1],
                                               normalized_betas[layer, 4, 2], 0,
                                               normalized_gamma_up, normalized_gamma_same, normalized_gamma_down)
                count += 1
                level1 = new_level1
                level2 = new_level2
                level3 = new_level3
                level4 = new_level4
                level5 = new_level5

        aspp_result1 = self.aspp_1(level1)
        aspp_result2 = self.aspp_2(level2)
        aspp_result3 = self.aspp_3(level3)
        aspp_result4 = self.aspp_4(level4)
        aspp_result5 = self.aspp_5(level5)
        upsample = nn.Upsample(size=x.size()[2:], mode='bilinear', align_corners=True)
        aspp_result1 = upsample(aspp_result1)
        aspp_result2 = upsample(aspp_result2)
        aspp_result3 = upsample(aspp_result3)
        aspp_result4 = upsample(aspp_result4)
        aspp_result5 = upsample(aspp_result5)

        sum_result = aspp_result1 + aspp_result2 + aspp_result3 + aspp_result4 + aspp_result5

        return sum_result

    def _initialize_alphas_betas(self):
        k = sum(1 for i in range(self._steps) for n in range(1 + i))

        # OPS定义于operations.py中
        num_ops = len(PRIMITIVES)

        alphas_up = (1e-3 * torch.randn([k, num_ops], requires_grad=True))
        alphas_same = (1e-3 * torch.randn([k, num_ops], requires_grad=True))
        alphas_down = (1e-3 * torch.randn([k, num_ops], requires_grad=True))

        betas = (1e-3 * torch.randn([self._num_layers, self._block_multiplier, 4], requires_grad=True))

        gamma_up = (1e-3 * torch.randn(k, requires_grad=True))
        gamma_same = (1e-3 * torch.randn(k, requires_grad=True))
        gamma_down = (1e-3 * torch.randn(k, requires_grad=True))

        self._arch_parameters = [alphas_up, alphas_same, alphas_down, betas, gamma_up, gamma_same, gamma_down]
        self._arch_param_names = ['alphas_up', 'alphas_same', 'alphas_down', 'betas', 'gamma_up', 'gamma_same',
                                  'gamma_down']

        [self.register_parameter(name, torch.nn.Parameter(param)) for name, param in
         zip(self._arch_param_names, self._arch_parameters)]

    def arch_parameters(self):
        return [param for name, param in self.named_parameters() if name in self._arch_param_names]

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if name not in self._arch_param_names]


if __name__ == '__main__':
    model = AutoNet(filter_multiplier=config['filter_multiplier'], block_multiplier=config['block_multiplier'])
    model = model.cuda()
    # with torch.autograd.set_detect_anomaly(True):

    optimizer = torch.optim.SGD(
        model.weight_parameters(),
        config['lr'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )
    model.train()
    for i in range(5):
        data = torch.randn([8, 3, 32, 32], device="cuda:0")
        target = torch.randn([8, 1, 32, 32], device="cuda:0")
        optimizer.zero_grad()
        result = model(data)
        loss = torch.sum(target - result)
        loss.backward()
        optimizer.step()
