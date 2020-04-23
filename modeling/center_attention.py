import torch.nn as nn
import torch.nn.functional as F
import torch
from modeling.fc import *
from modeling.bilinear_attention import *
from modeling.counting import Counter
from modeling.bc import *

## Center Guided Self Attention
class CGSA(nn.Module):
    def __init__(self):
        super(CGSA, self).__init__()


    def forward(self, v_relation, q_emb, b=[]):