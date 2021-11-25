import torch
import torch.nn as nn
from torch.nn import functional as F


Distribution_ops = {
    'skip': lambda  : Identity(),
    'normalization_channel' : lambda  : Normalization_channel(),
    'normalization_global' : lambda  : Normalization_global(),
    'normalization_sep' : lambda  : Normalization_sep()
}


dist_ops = {
    'multiplication' : lambda query_num, class_num : Multiplication(query_num, class_num),
    'sub_abs' : lambda query_num, class_num : sub_abs(query_num, class_num),
    'sub_squ' : lambda query_num, class_num : sub_squ(query_num, class_num),
    'cov' : lambda  query_num, class_num : cov(query_num, class_num)
}


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x, y):
        y_list = []
        B, C, h, w = x.size()
        x1 = x.view(B, C, h*w).cuda()
        for i in range(len(y)):
            y_sam = y[i]
            B, C, h, w = y_sam.size()
            y_sam = y_sam.view(B, C, h*w).cuda()
            y_list.append(y_sam)
        return x1, y_list


class Normalization_channel(nn.Module):
    def __init__(self):
        super(Normalization_channel, self).__init__()

    def forward(self,x, y):
        B, C, h, w = x.size()
        x_norm = torch.zeros(B, C, h*w).cuda()

        for i in range(B):
            x_sam = x[i]
            x_sam = x_sam.view(C, -1)
            x_sam_norm = torch.norm(x_sam, 2, 1, True)
            x_sam = x_sam / x_sam_norm
            x_norm[i] = x_sam

        y_list = []

        for i in range(len(y)):
            y_class = y[i]
            B, C, h, w = y_class.size()
            y_norm = torch.zeros(B, C, h*w).cuda()
            for j in range(y_class.size(0)):
                y_sam = y_class[j]
                y_sam = y_sam.view(C,-1)
                y_sam_norm = torch.norm(y_sam, 2, 1, True)
                y_sam = y_sam / y_sam_norm
                y_sam = y_sam.view(C, -1)
                y_norm[j] = y_sam
            y_list.append(y_norm)
        return x_norm, y_list


class Normalization_global(nn.Module):
    def __init__(self):
        super(Normalization_global, self).__init__()

    def forward(self,x, y):
        B, C, h, w = x.size()
        x_norm = torch.zeros(B, C, h* w).cuda()
        y_list = []

        for i in range(B):
            x_sam = x[i]
            x_sam_global = x_sam.view(-1)
            x_sam_norm = torch.norm(x_sam_global, 2, 0, True)
            x_sam = x_sam / x_sam_norm
            x_sam = x_sam.view(C, -1)
            x_norm[i] = x_sam*10

        for i in range(len(y)):
            y_class = y[i]
            B, C, h, w = y_class.size()
            y_norm = torch.zeros(B, C, h*w).cuda()
            for j in range(y_class.size(0)):
                y_sam = y_class[j]
                y_sam_global = y_sam.view(-1)
                y_sam_norm = torch.norm(y_sam_global, 2, 0, True)
                y_sam = y_sam / y_sam_norm
                y_sam = y_sam.view(C, -1)
                y_norm[j] = y_sam*10
            y_list.append(y_norm)
        return x_norm, y_list


class Normalization_sep(nn.Module):
    def __init__(self):
        super(Normalization_sep, self).__init__()

    def forward(self, x, y):
        B, C, h, w = x.size()
        x_norm = torch.zeros(B, C, h * w).cuda()
        for i in range(B):
            query_sam = x[i]
            query_sam = query_sam.view(C, -1)
            query_sam_norm = torch.norm(query_sam, 2, 1, True)
            query_sam = query_sam / query_sam_norm
            x_norm[i] = query_sam

        y_list = []
        for i in range(len(y)):
            support_set_sam = y[i]
            B, C, h, w = support_set_sam.size()  #

            support_set_sam = support_set_sam.permute(1, 0, 2, 3)  # 转换维度 为C B h w
            support_set_sam = support_set_sam.contiguous().view(C, -1)
            mean_support = torch.mean(support_set_sam, 1, True)  # 求类平均向量
            support_set_sam = support_set_sam - mean_support
            support_set_sam = support_set_sam.contiguous().view(C, B, h*w)
            support_set_sam = support_set_sam.permute( 1, 0, 2)
            y_list.append(support_set_sam)
        return x_norm, y_list


class Multiplication(nn.Module):
    def __init__(self, query_num, class_num):
        super(Multiplication, self).__init__()
        self.query_num = query_num
        self.class_num = class_num

    def Multiplier(self, v1, v2):
        length = min(v1.size(1), v2.size(1), )
        if length == 0: return 0
        dp1 = v1 * v2
        dp = dp1.sum(-1)
        return dp

    def forward(self, x, y):
        x = x.view(x.size(0), -1)
        y_proto = []
        for i in range(len(y)):
            temp_s = y[i]
            s_proto = temp_s.contiguous().view(temp_s.size(0), -1).mean(0, keepdim=True)
            y_proto.append(s_proto)
        y_proto = torch.cat(y_proto, 0)  #5, 1600
        d = torch.randn(self.query_num, self.class_num).cuda()
        for i in range(x.size(0)):
            for j in range(y_proto.size(0)):
                d[i][j] = self.Multiplier(x[i].unsqueeze(0), y_proto[j].unsqueeze(0))
        d = F.softmax(d, dim=1)
        return d


class sub_abs(nn.Module):
    def __init__(self, query_num, class_num):
        super(sub_abs, self).__init__()
        self.query_num = query_num
        self.class_num = class_num
    def sub_abs(self, input1, input2):
        q = input1.size(0)  # 75
        C = input2.size(0)  # 5
        d = input1.size(1)  # 1600
        assert d == input2.size(1)
        input1 = input1.unsqueeze(1).expand(q, C, d)  # size(75, 5, 1600)
        input2 = input2.unsqueeze(0).expand(q, C, d)  # size(75, 5, 1600)
        output = torch.abs(input1 - input2).sum(2)
        return output

    def forward(self, x, y):
        x = x.view(x.size(0), -1)
        y_proto = []
        for i in range(len(y)):
            temp_s = y[i]
            s_proto = temp_s.contiguous().view(temp_s.size(0), -1).mean(0, keepdim=True)
            y_proto.append(s_proto)
        y_proto = torch.cat(y_proto, 0)
        d = self.sub_abs(x, y_proto)
        d = F.softmax(-d, dim=1)
        return d


class sub_squ(nn.Module):
    def __init__(self, query_num, class_num):
        super(sub_squ, self).__init__()
        self.query_num = query_num
        self.class_num = class_num
    def sub_squ(self, input1, input2):
        q = input1.size(0)  # 75
        C = input2.size(0)  # 5
        d = input1.size(1)  # 1600
        assert d == input2.size(1)
        input1 = input1.unsqueeze(1).expand(q, C, d)  # size(75, 5, 1600)
        input2 = input2.unsqueeze(0).expand(q, C, d)  # size(75, 5, 1600)
        output = torch.pow(input1 - input2, 2).sum(2)
        return output

    def forward(self, x, y):
        x = x.view(x.size(0), -1)
        y_proto = []
        for i in range(len(y)):
            temp_s = y[i]
            s_proto = temp_s.contiguous().view(temp_s.size(0), -1).mean(0, keepdim=True)
            y_proto.append(s_proto)
        y_proto = torch.cat(y_proto, 0)
        d = self.sub_squ(x, y_proto)
        d = F.softmax(-d, dim=1)
        return d


class cov(nn.Module):
    def __init__(self, query_num, class_num):
        super(cov, self).__init__()
        self.query_num = query_num
        self.class_num = class_num
        self.classifier_25 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout(),
            nn.Conv1d(1, 1, kernel_size=25, stride=25),
        )
        self.classifier_441 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout(),
            nn.Conv1d(1, 1, kernel_size=441, stride=441),
        )

    def cov_matrix(self, input):
        CovaMatrix_list = []
        for i in range(len(input)):
            support_set_sam = input[i]
            B, C, hw = support_set_sam.size()

            support_set_sam = support_set_sam.permute(1, 0, 2)
            support_set_sam = support_set_sam.contiguous().view(C, -1)

            covariance_matrix = support_set_sam @ torch.transpose(support_set_sam, 0, 1)  # 转置点乘求和
            covariance_matrix = torch.div(covariance_matrix, hw * B - 1)
            CovaMatrix_list.append(covariance_matrix)
        return CovaMatrix_list

    def cov_dist(self, input, CovaMatrix_list):

        B, C, hw = input.size()
        Cova_Sim = []

        for i in range(B):
            query_sam = input[i]
            if torch.cuda.is_available():   #cuda
                mea_sim = torch.zeros(1, len(CovaMatrix_list) * hw).cuda()
            # mea_sim = torch.zeros(1, len(CovaMatrix_list)*hw)

            for j in range(len(CovaMatrix_list)):
                temp_dis = torch.transpose(query_sam, 0, 1) @ CovaMatrix_list[j] @ query_sam
                mea_sim[0, j * hw:(j + 1) * hw] = temp_dis.diag()

            Cova_Sim.append(mea_sim.unsqueeze(0))

        Cova_Sim = torch.cat(Cova_Sim, 0)  # get Batch*1*(h*w*num_classes)
        return Cova_Sim

    def forward(self, x, y):
        cov_matrix = self.cov_matrix(y)
        cov_dist = self.cov_dist(x, cov_matrix)
        if x.size(2)==25:
            cov = self.classifier_25(cov_dist)
        else:
            cov = self.classifier_441(cov_dist)
        cov = cov.squeeze(1)
        return cov
