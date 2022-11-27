import numpy as np
import random
import torch
import torch.nn as nn

np.random.seed(0)
random.seed(0)
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

''' Universal discriminator to maximize mutual information between local and
 cluster-aware global patches of the heterogeneous graph. '''

class Discriminator(nn.Module):
    def __init__(self, n_h, n_s):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_s, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        ''' We use node specific cluster-based global summary representation with positive and negative local patches,
         for learning consensus based representation via discriminator based on InfoMax Principle. '''
        sc_1 = self.f_k(h_pl, c) # Positive local + global patch consensus: [(3550 x 64), (64 x 64) x (64 x 3550)]
        sc_2 = self.f_k(h_mi, c) # Negative local + global patch consensus: [(3550 x 64), (64 x 64) x (64 x 3550)]

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1) # Concatenation of positive & negative logits

        return logits
