import torch.nn as nn
import torch.nn.functional as F
import torch
from modeling.fc import *
from modeling.bilinear_attention import *
from modeling.counting import Counter
from modeling.bc import *
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class BAN(nn.Module):
    def __init__(self, v_relation_dim, num_hid, gamma,
                 min_num_objects=10, use_counter=False):
        super(BAN, self).__init__()

        self.v_att = BiAttention(v_relation_dim, num_hid, num_hid, gamma)
        self.glimpse = gamma
        self.use_counter = use_counter
        b_net = []
        q_prj = []
        c_prj = []
        q_att = []
        v_prj = []

        for i in range(gamma):
            b_net.append(BCNet(v_relation_dim, num_hid, num_hid, None, k=1))
            q_prj.append(FCNet([num_hid, num_hid], '', .2))
            if self.use_counter:
                c_prj.append(FCNet([min_num_objects + 1, num_hid], 'ReLU', .0))

        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)
        self.q_att = nn.ModuleList(q_att)
        self.v_prj = nn.ModuleList(v_prj)
        if self.use_counter:
            self.c_prj = nn.ModuleList(c_prj)
            self.counter = Counter(min_num_objects)

    def forward(self, v_relation, q_emb, b=[]):
        if self.use_counter:
            boxes = b[:, :, :4].transpose(1, 2)

        b_emb = [0] * self.glimpse
        # b x g x v x q
        att, att_logits = self.v_att.forward_all(v_relation, q_emb)

        for g in range(self.glimpse):
            # b x l x h
            b_emb[g] = self.b_net[g].forward_with_weights(
                                        v_relation, q_emb, att[:, g, :, :])
            # atten used for counting module
            atten, _ = att_logits[:, g, :, :].max(2)
            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb

            if self.use_counter:
                embed = self.counter(boxes, atten)
                q_emb = q_emb + self.c_prj[g](embed).unsqueeze(1)
        joint_emb = q_emb.sum(1)
        return q_emb, att


class Encoder_fusion(nn.Module):
    def __init__(self,sync_bn=True):
        super().__init__()

        self.bert_dim=768
        self.attn_dim=256

        if sync_bn:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.num_pos=12

        self.conv1 = nn.Sequential(nn.Conv2d(512, self.attn_dim, kernel_size=1, stride=1,
                                bias=False),
                                   BatchNorm(self.attn_dim),
                                   nn.ReLU(),
                                   nn.Dropout(0.5),
                                   )


        self.conv2 = nn.Sequential(nn.Conv2d(256, self.attn_dim, kernel_size=1, stride=1,
                                bias=False),
                                   BatchNorm(self.attn_dim),
                                   nn.ReLU(),
                                   nn.Dropout(0.5),
                                   )

        self.conv3 = nn.Sequential(nn.Conv2d(256, self.attn_dim, kernel_size=1, stride=1,
                                bias=False),
                                   BatchNorm(self.attn_dim),
                                   nn.ReLU(),
                                   nn.Dropout(0.5),
                                   )

        self.t_dim=FCNet([self.bert_dim, self.attn_dim], bias=False)
        self.v_dim = FCNet([self.attn_dim, self.attn_dim], bias=False)
        self.ban=BAN(self.attn_dim,self.attn_dim, 2)
        # self.ban_t = BAN(self.attn_dim, self.attn_dim, 1)
        self.ban_self = BAN(self.attn_dim, self.attn_dim, 1)

        self.conv_f = nn.Sequential(nn.Conv2d(self.attn_dim, self.attn_dim, kernel_size=1, stride=1,
                                bias=False),
                                   BatchNorm(self.attn_dim),
                                   nn.ReLU(),
                                   nn.Dropout(0.5),
                                   )
        # self.center_conf = nn.Sequential(FCNet([attn_dim, 64], bias=True),
        #                                  FCNet([64, 1], bias=True))

        # self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, input):

        (feat_high, feat_mid, feat_low, text) = input # [b,256,80,80], [b,512,40,40], [b,256,20,20]
        feat_mid = self.conv1(feat_mid)
        feat_high = self.conv2(feat_high)
        feat_low = self.conv3(feat_low)

        batch = feat_high.shape[0]

        ## creating position matrix

        # feat_high_pos = extract_position_matrix(feat_high.shape)
        # feat_mid_pos = extract_position_matrix(feat_mid.shape)
        # feat_low_pos = extract_position_matrix(feat_low.shape)

        feat_cat = torch.cat([feat_high.view(batch,-1,self.attn_dim),feat_mid.view(batch, -1, self.attn_dim),feat_low.view(batch, -1, self.attn_dim)],dim=1)
        # pos_center = torch.cat(
        #     [feat_high_pos.view(batch, -1, 2), feat_mid_pos.view(batch, -1, 2), feat_low_pos.view(batch, -1, 2)], dim=1)

        # text.shape= [b,1,20,768], feat_center.shape=[b,8400,256]
        text_emb = self.t_dim(text.squeeze(1))
        feat_cat= self.v_dim(feat_cat)

        ## bilinear attention with bert embedding

        (joint_emb, _) = self.ban(text_emb, feat_cat)
        # (joint_emb_t, _) = self.ban_t(feat_cat, text_emb)
        # joint_emb = torch.cat([joint_emb,joint_emb_t],dim=1)
        (joint_emb, _) = self.ban_self(joint_emb, joint_emb)

        feat_high_emb=joint_emb[:,:feat_high.shape[2]*feat_high.shape[3],:].view(batch,feat_high.shape[2],feat_high.shape[3],-1).permute(0,3,1,2)
        feat_mid_emb=joint_emb[:,feat_high.shape[2]*feat_high.shape[3]:feat_high.shape[2]*feat_high.shape[3]+feat_mid.shape[2]*feat_mid.shape[3],:].view(batch,feat_mid.shape[2],feat_mid.shape[3],-1).permute(0,3,1,2)
        feat_low_emb=joint_emb[:,feat_high.shape[2]*feat_high.shape[3]+feat_mid.shape[2]*feat_mid.shape[3]:,:].view(batch,feat_low.shape[2],feat_low.shape[3],-1).permute(0,3,1,2)

        feat_mid_emb = F.interpolate(feat_mid_emb, size=feat_high_emb.size()[2:], mode='bilinear', align_corners=True)
        feat_low_emb = F.interpolate(feat_low_emb, size=feat_high_emb.size()[2:], mode='bilinear', align_corners=True)

        feat_final=feat_high_emb*feat_mid_emb*feat_low_emb

        feat_final=self.conv_f(feat_final)



        # ## center prediction
        # centerness = self.center_conf(joint_emb)
        #
        # _,order = torch.sort(torch.norm(torch.cat([(pos_center[:, :, 0] - center[:, 0].unsqueeze(1)).unsqueeze(2),(pos_center[:, :, 1] - center[:, 1].unsqueeze(1)).unsqueeze(2)],2),dim=2),dim=1)
        #
        # logits = center.new(batch, self.num_pos, 1).zero_()
        # for i in range(batch):
        #     centerness_single = centerness[i]
        #     order_single = order[i]
        #     order_single = order_single[:self.num_pos]
        #
        #     centerness_single = centerness_single[order_single,:]
        #     logits[i]=centerness_single



        return feat_final



def extract_position_matrix(shape):
    device = torch.device("cuda")

    batch=shape[0]
    channel=shape[1]
    height_dim=shape[2]
    width_dim=shape[3]


    height_range = torch.repeat_interleave(torch.linspace(0, 1, steps=height_dim).unsqueeze(0), width_dim, dim=0)
    width_range = torch.repeat_interleave(torch.linspace(0, 1, steps=width_dim).unsqueeze(1), height_dim, dim=1)

    return torch.cat([height_range.unsqueeze(0),width_range.unsqueeze(0)]).unsqueeze(0).repeat(batch,1,1,1).cuda()

# def bidirect_attention(v,t):
