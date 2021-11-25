import torch
from torch import nn
import torch.nn.functional as F
from new_metric_ops import Distribution_ops, dist_ops
from genotypes import PRIMITIVES_ops1, PRIMITIVES_ops2, Genotype
import functools



class MixedLayer(nn.Module):
    """
    a mixtures output of 8 type of units.

    we use weights to aggregate these outputs while training.
    and softmax to select the strongest edges while inference.
    """

    def __init__(self, query_num, class_num):
        """

        :param query_num: class_num * query_num
        :param class_num: 5
        """
        super(MixedLayer, self).__init__()

        self.layers1 = nn.ModuleList()

        self.layers2 = nn.ModuleList()

        """

        PRIMITIVES_ops1 = [
                  'skip',
                  'normalization_channel',
                  'normalization_global',
                  'normalization_sep',

              ]
    
        PRIMITIVES_ops2 = [
         
                  'multiplication',
                  'sub_abs',
                  'sub_squ',
                  'cov',
                         
              ]     
              

        """
        for primitive in PRIMITIVES_ops1:
            # create corresponding layer
            layer1 = Distribution_ops[primitive]()
            self.layers1.append(layer1)

        for primitive in PRIMITIVES_ops2:
            # create corresponding layer
            layer2 = dist_ops[primitive](query_num, class_num)
            self.layers2.append(layer2)



    def forward(self, query, support, weights):
        """

        :param x: data
        :param weights: alpha,[op_num:4], the output = sum of alpha * op(x)
        :return:
        """
        support1 = []
        support_sum_list = []
        query1 = []
        for w, layer in zip(weights[0], self.layers1):
            temp = layer(query, support)
            B, C, hw = temp[1][0].size()
            support_tensor = torch.randn(len(temp[1]), B, C, hw)
            query1 += [w * temp[0].cuda()]
            for i in range(len(temp[1])):
                support_tensor[i] =  w * temp[1][i].cuda()
            support1 += [support_tensor]
        query_sum = sum(query1).cuda()
        support_sum = sum(support1)
        for i in range(support_sum.size(0)):
            support_sum_list += [support_sum[i].cuda()]

        res2 = [w.cuda()* layer(query_sum, support_sum_list).cuda() for w, layer in zip(weights[1], self.layers2)]
        res = sum(res2)
        return res


class Cell(nn.Module):

    def __init__(self, steps, query_num, class_num):
        """

        :param steps: 2, number of layers inside a cell

        """
        super(Cell, self).__init__()

        # steps inside a cell
        self.steps = steps  # 2


        self.layers = nn.ModuleList()

        for i in range(self.steps):
            # for each i inside cell, it connects with all previous output
            # plus previous two cells' output

            layer = MixedLayer(query_num, class_num)
            self.layers.append(layer)


    def forward(self, query, support, q_image, S_image, weights, weight_metric):
        """

        :param query:
        :param support:
        :param weights: [2, 4]
        :return:
        """

        offset = 0

        s = self.layers[offset](query, support, weights[offset])
        offset += 1
        # append one state since s is the elem-wise addition of all output

        s1 = self.layers[offset](q_image, S_image, weights[offset])

        mix = weight_metric[0][0]*s + weight_metric[0][1]*s1 # 两节点相加


        # concat along dim=channel
        return mix  # [75, 5]


class Network(nn.Module):
    """
    stack number:layer of cells and then flatten to fed a linear layer
    """

    def __init__(self, query_num, class_num, criterion, steps=2, norm_layer=nn.BatchNorm2d):
        """


        :param num_classes: 5
        :param criterion:
        :param steps: nodes num inside cell
        :param multiplier: output channel of cell = multiplier * ch

        """
        super(Network, self).__init__()

        self.query_num = query_num
        self.class_num = class_num

        self.criterion = criterion
        self.steps = steps


        self.cells = nn.ModuleList()

        cell = Cell(steps, query_num, class_num)
        self.cells += [cell]

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, inplace=False),
        )
        self.image_features = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        l = 2
        k = 2
        num_ops = len(Distribution_ops)  # 4

        # TODO
        # this kind of implementation will add alpha into self.parameters()
        # it has num k of alpha parameters, and each alpha shape: [num_ops]
        # it requires grad and can be converted to cpu/gpu automatically
        self.alpha_normal = nn.Parameter(torch.randn(l, k, num_ops))
        self.alpha_layer = nn.Parameter(torch.randn(1, 2))



        with torch.no_grad():
            # initialize to smaller value
            self.alpha_normal.mul_(1e-3)
            self.alpha_layer.mul_(1e-3)
        self._arch_parameters = [
            self.alpha_normal,
            self.alpha_layer
        ]

    def new(self):
        """
        create a new model and initialize it with current alpha parameters.
        However, its weights are left untouched.
        :return:
        """
        model_new = Network(self.query_num, self.class_num, self.criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input1, input2):
        """
        input1: torch.Size([75, 3, 84, 84])
        input2: len(list)= 5 torch.size([1/5, 3, 84, 84])

        """

        # s0 & s1 means the last cells' output
        q = self.features(input1)

        # extract features of input2--support set
        S = []
        for i in range(len(input2)):
            S.append(self.features(input2[i]))

        q_image = self.image_features(q)

        S_image = []
        for i in range(len(S)):
            temp_s = self.image_features(S[i])
            S_image.append(temp_s)


        for i, cell in enumerate(self.cells):
            # weights are shared across all reduction cell or normal cell
            # according to current cell's type, it choose which architecture parameters
            # to use
            weights = F.softmax(self.alpha_normal, dim=-1)
            weight_layer = F.softmax(self.alpha_layer, dim=-1)

            s0 = cell(q, S, q_image, S_image, weights, weight_layer)

        return s0

    def loss(self, input1, input2, target):
        """

        :param x:
        :param target:
        :return:
        """
        logits = self(input1, input2)
        return self.criterion(logits, target)

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):
        """
        :return:
        """
        def _parse1(weights):
            gene = []
            for i in range(2):
                W = weights[i].copy()
                j = 0
                k_best = None
                for k in range(len(W[j])): 
                    if k_best is None or W[j][k] > W[j][k_best]:
                        k_best = k
                gene.append((PRIMITIVES_ops1[k_best], j))

                j += 1
                k_best = None
                for k in range(len(W[j])):
                    if k_best is None or W[j][k] > W[j][k_best]:
                        k_best = k
                gene.append((PRIMITIVES_ops2[k_best], j))
            return gene

        gene_normal = _parse1(F.softmax(self.alpha_normal, dim=-1).data.cpu().numpy())
        weight_layer = F.softmax(self.alpha_layer, dim=-1).data
        genotype = Genotype(
            gene=gene_normal, weight=weight_layer

        )

        return genotype


