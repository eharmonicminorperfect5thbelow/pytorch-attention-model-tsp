import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


def try_gpu(e):
    if torch.cuda.is_available():
        return e.cuda()
    return e

class Encoder(nn.Module):
    def __init__(self, dx=2, dh=128, dff=512, N=3, M=8):
        super(Encoder, self).__init__()

        self.dx = dx
        self.dh = dh
        self.dk = dh // M
        self.dv = dh // M
        self.dff = dff
        self.N = N
        self.M = M

        self.dense_h = nn.Linear(self.dx, self.dh)

        self.dense_q = nn.ModuleList([nn.Linear(self.dh, self.dk, bias=False) for _ in range(self.N * self.M)])
        self.dense_v = nn.ModuleList([nn.Linear(self.dh, self.dv, bias=False) for _ in range(self.N * self.M)])
        self.dense_k = nn.ModuleList([nn.Linear(self.dh, self.dk, bias=False) for _ in range(self.N * self.M)])
        self.dense_o = nn.ModuleList([nn.Linear(self.dv, self.dh, bias=False) for _ in range(self.N * self.M)])

        self.w_bn = nn.ParameterList([nn.Parameter(torch.randn(self.dh)) for _ in range(self.N)])
        self.b_bn = nn.ParameterList([nn.Parameter(torch.randn(self.dh)) for _ in range(self.N)])

        self.dense_ff_0 = nn.ModuleList([nn.Linear(self.dh, self.dff) for _ in range(self.N)])
        self.dense_ff_1 = nn.ModuleList([nn.Linear(self.dff, self.dh) for _ in range(self.N)])

        self.softmax = nn.Softmax(2)

    def forward(self, x):
        h = self.dense_h(x)
        self.bn = try_gpu(nn.BatchNorm1d(x.size()[1], affine=False))

        for n in range(self.N):
            h_mha = self.multi_head_attention(h, n)
            h_bn = self.w_bn[n] * self.bn(h + h_mha) + self.b_bn[n]
            h_ff = self.feed_forward(h_bn, n)
            h = self.w_bn[n] * self.bn(h_bn + h_ff) + self.b_bn[n]

        return h

    def multi_head_attention(self, x, n):
        h = 0

        for m in range(self.M):
            q = self.dense_q[n * self.M + m](x)
            v = self.dense_v[n * self.M + m](x)
            k = self.dense_k[n * self.M + m](x)
            k_t = torch.transpose(k, 1, 2)
            u = torch.bmm(q, k_t) / math.sqrt(self.dk)
            s = self.softmax(u)
            t = torch.bmm(s, v)
            h += self.dense_o[n * self.M + m](t)

        return h

    def feed_forward(self, x, n):
        h_0 = F.relu(self.dense_ff_0[n](x))
        h_1 = self.dense_ff_1[n](h_0)

        return h_1


class Decoder(nn.Module):
    def __init__(self, dx=2, dh=128, M=8, samples=1280):
        super(Decoder, self).__init__()

        self.dx = dx
        self.dh = dh
        self.dk = dh // M
        self.dv = dh // M
        self.M = M
        self.samples = samples

        self.vl = nn.Parameter(torch.randn(self.dh))
        self.vf = nn.Parameter(torch.randn(self.dh))

        self.dense_q = nn.ModuleList([nn.Linear(3 * self.dh, self.dk, bias=False) for _ in range(self.M)])
        self.dense_v = nn.ModuleList([nn.Linear(self.dh, self.dv, bias=False) for _ in range(self.M)])
        self.dense_k = nn.ModuleList([nn.Linear(self.dh, self.dk, bias=False) for _ in range(self.M)])
        self.dense_o = nn.ModuleList([nn.Linear(self.dv, 3 * self.dh, bias=False) for _ in range(self.M)])

        self.dense_q_a = nn.Linear(3 * self.dh, self.dk, bias=False)
        self.dense_k_a = nn.Linear(self.dh, self.dk, bias=False)

        self.softmax = nn.Softmax(2)

    def forward(self, x, decode):
        log_p = try_gpu(torch.zeros(x.size()[0], 1))
        mask = try_gpu(torch.zeros(x.size()[0], 1, x.size()[1]))
        selected = try_gpu(torch.zeros(x.size()[0], 0).type(torch.int64))
        
        m = x.sum(1) / x.size()[1]
        m = m.view(m.size()[0], 1, m.size()[1])

        vl = self.vl.expand(m.size())
        vf = self.vf.expand(m.size())

        for i in range(x.size()[1]):
            hc = torch.cat((m, vl, vf), 2)
            h = self.multi_head_attention(hc, x, mask)
            a, prob = self.attention(hc, x, mask)

            if decode == 'greedy':
                p = a.max(1)
                log_p = log_p + p.values
                indices = p.indices.unsqueeze(1)
            else:
                indices = prob.multinomial(1)
                log_p = log_p + a.gather(1, indices)
            
            selected = torch.cat((selected, indices), 1)

            vl_numpy = np.zeros(m.size())
            vf_numpy = np.zeros(m.size())
            
            for j in range(len(indices)):
                vl_numpy[j] = x[j][selected[j][-1]].detach().cpu().numpy()
                vf_numpy[j] = x[j][selected[j][0]].detach().cpu().numpy()
                mask[j][0][indices[j][0]] = float('Inf')
            
            vl = try_gpu(torch.from_numpy(vl_numpy).type(torch.float32))
            vf = try_gpu(torch.from_numpy(vf_numpy).type(torch.float32))

        return selected, log_p.squeeze()

    def multi_head_attention(self, x1, x2, mask):
        h = 0

        for m in range(self.M):
            q = self.dense_q[m](x1)
            v = self.dense_v[m](x2)
            k = self.dense_k[m](x2)
            k_t = torch.transpose(k, 1, 2)
            u = torch.bmm(q, k_t) / math.sqrt(self.dk) - mask
            s = self.softmax(u)
            t = torch.bmm(s, v)
            h += self.dense_o[m](t)

        return h

    def attention(self, x1, x2, mask):
        q = self.dense_q_a(x1)
        k = self.dense_k_a(x2)
        k_t = torch.transpose(k, 1, 2)
        u = torch.bmm(q, k_t) / math.sqrt(self.dk) - mask
        s = F.log_softmax(u, 2).squeeze()
        p = self.softmax(u).squeeze()

        return s, p
