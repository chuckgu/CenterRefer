from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from .darknet import *
from model.ReferCam import *
from pydensecrf import densecrf
from model.pamr import PAMR
import argparse
import collections
import logging
import json
import re
import time
## can be commented if only use LSTM encoder
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

def max_norm(p, version='torch', e=1e-5):
	if version is 'torch':
		if p.dim() == 3:
			C, H, W = p.size()
			p = F.relu(p)
			max_v = torch.max(p.view(C,-1),dim=-1)[0].view(C,1,1)
			min_v = torch.min(p.view(C,-1),dim=-1)[0].view(C,1,1)
			p = F.relu(p-min_v-e)/(max_v-min_v+e)
		elif p.dim() == 4:
			N, C, H, W = p.size()
			p = F.relu(p)
			max_v = torch.max(p.view(N,C,-1),dim=-1)[0].view(N,C,1,1)
			min_v = torch.min(p.view(N,C,-1),dim=-1)[0].view(N,C,1,1)
			p = F.relu(p-min_v-e)/(max_v-min_v+e)
	elif version is 'numpy' or version is 'np':
		if p.ndim == 3:
			C, H, W = p.shape
			p[p<0] = 0
			max_v = np.max(p,(1,2),keepdims=True)
			min_v = np.min(p,(1,2),keepdims=True)
			p[p<min_v+e] = 0
			p = (p-min_v-e)/(max_v+e)
		elif p.ndim == 4:
			N, C, H, W = p.shape
			p[p<0] = 0
			max_v = np.max(p,(2,3),keepdims=True)
			min_v = np.min(p,(2,3),keepdims=True)
			p[p<min_v+e] = 0
			p = (p-min_v-e)/(max_v+e)
	return p


class Self_Attn_Layer(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn_Layer, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        if type(x) is list or type(x) is tuple:
            x=x[0]
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out,attention

class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, num_layer, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.num_layer = num_layer

        self.layers=nn.Sequential()

        for i in range(num_layer):
            self.layers.add_module(
                "attn_%d" % i,
                Self_Attn_Layer(self.chanel_in, 'relu')
            )

    def forward(self, x):
        return self.layers(x)



def generate_coord(batch, height, width):
    # coord = Variable(torch.zeros(batch,8,height,width).cuda())
    xv, yv = torch.meshgrid([torch.arange(0,height), torch.arange(0,width)])
    xv_min = (xv.float()*2 - width)/width
    yv_min = (yv.float()*2 - height)/height
    xv_max = ((xv+1).float()*2 - width)/width
    yv_max = ((yv+1).float()*2 - height)/height
    xv_ctr = (xv_min+xv_max)/2
    yv_ctr = (yv_min+yv_max)/2
    hmap = torch.ones(height,width)*(1./height)
    wmap = torch.ones(height,width)*(1./width)
    coord = torch.autograd.Variable(torch.cat([xv_min.unsqueeze(0), yv_min.unsqueeze(0),\
        xv_max.unsqueeze(0), yv_max.unsqueeze(0),\
        xv_ctr.unsqueeze(0), yv_ctr.unsqueeze(0),\
        hmap.unsqueeze(0), wmap.unsqueeze(0)], dim=0).cuda())
    coord = coord.unsqueeze(0).repeat(batch,1,1,1)
    return coord

class RNNEncoder(nn.Module):
    def __init__(self, vocab_size, word_embedding_size, word_vec_size, hidden_size, bidirectional=False,
               input_dropout_p=0, dropout_p=0, n_layers=1, rnn_type='lstm', variable_lengths=True):
        super(RNNEncoder, self).__init__()
        self.variable_lengths = variable_lengths
        self.embedding = nn.Embedding(vocab_size, word_embedding_size)
        self.input_dropout = nn.Dropout(input_dropout_p)
        self.mlp = nn.Sequential(nn.Linear(word_embedding_size, word_vec_size), 
                                 nn.ReLU())
        self.rnn_type = rnn_type
        self.rnn = getattr(nn, rnn_type.upper())(word_vec_size, hidden_size, n_layers,
                                                 batch_first=True,
                                                 bidirectional=bidirectional,
                                                 dropout=dropout_p)
        self.num_dirs = 2 if bidirectional else 1

    def forward(self, input_labels):
        """
        Inputs:
        - input_labels: Variable long (batch, seq_len)
        Outputs:
        - output  : Variable float (batch, max_len, hidden_size * num_dirs)
        - hidden  : Variable float (batch, num_layers * num_dirs * hidden_size)
        - embedded: Variable float (batch, max_len, word_vec_size)
        """
        if self.variable_lengths:
            input_lengths = (input_labels!=0).sum(1)  # Variable (batch, )

            # make ixs
            input_lengths_list = input_lengths.data.cpu().numpy().tolist()
            sorted_input_lengths_list = np.sort(input_lengths_list)[::-1].tolist() # list of sorted input_lengths
            sort_ixs = np.argsort(input_lengths_list)[::-1].tolist() # list of int sort_ixs, descending
            s2r = {s: r for r, s in enumerate(sort_ixs)} # O(n)
            recover_ixs = [s2r[s] for s in range(len(input_lengths_list))]  # list of int recover ixs
            assert max(input_lengths_list) == input_labels.size(1)

            # move to long tensor
            sort_ixs = input_labels.data.new(sort_ixs).long()  # Variable long
            recover_ixs = input_labels.data.new(recover_ixs).long()  # Variable long

            # sort input_labels by descending order
            input_labels = input_labels[sort_ixs]

        # embed
        embedded = self.embedding(input_labels)  # (n, seq_len, word_embedding_size)
        embedded = self.input_dropout(embedded)  # (n, seq_len, word_embedding_size)
        embedded = self.mlp(embedded)            # (n, seq_len, word_vec_size)
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_input_lengths_list, batch_first=True)
        # forward rnn
        output, hidden = self.rnn(embedded)
        # recover
        if self.variable_lengths:
            # recover rnn
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True) # (batch, max_len, hidden)
            output = output[recover_ixs]
        sent_output = []
        for ii in range(output.shape[0]):
            sent_output.append(output[ii,int(input_lengths_list[ii]-1),:])
        return torch.stack(sent_output, dim=0)

class grounding_model(nn.Module):
    def __init__(self, corpus=None, emb_size=256, jemb_drop_out=0.1, bert_model='bert-base-uncased', \
     coordmap=True, leaky=False, dataset=None, light=False,seg=False,att=False):
        super(grounding_model, self).__init__()
        self.coordmap = coordmap
        self.light = light
        self.seg=seg
        self.att=att
        self.lstm = (corpus is not None)
        self.emb_size = emb_size
        if bert_model=='bert-base-uncased':
            self.textdim=768
        else:
            self.textdim=1024
        ## Visual model
        self.visumodel = Darknet(config_path='./model/yolov3.cfg')
        self.visumodel.load_weights('./saved_models/yolov3.weights')
        self.intmd_fea=[]
        ## Text model
        if self.lstm:
            self.textdim, self.embdim=1024, 512
            self.textmodel = RNNEncoder(vocab_size=len(corpus),
                                          word_embedding_size=self.embdim,
                                          word_vec_size=self.textdim//2,
                                          hidden_size=self.textdim//2,
                                          bidirectional=True,
                                          input_dropout_p=0.2,
                                          variable_lengths=True)
        else:
            self.textmodel = BertModel.from_pretrained(bert_model)

        ## Mapping module
        self.mapping_visu = nn.Sequential(OrderedDict([
            ('0', ConvBatchNormReLU(1024, emb_size, 1, 1, 0, 1, leaky=leaky)),
            ('1', ConvBatchNormReLU(512, emb_size, 1, 1, 0, 1, leaky=leaky)),
            ('2', ConvBatchNormReLU(256, emb_size, 1, 1, 0, 1, leaky=leaky))
        ]))
        self.mapping_lang = torch.nn.Sequential(
          nn.Linear(self.textdim, emb_size),
          nn.BatchNorm1d(emb_size),
          nn.ReLU(),
          nn.Dropout(jemb_drop_out),
          nn.Linear(emb_size, emb_size),
          nn.BatchNorm1d(emb_size),
          nn.ReLU(),
        )
        embin_size = emb_size*2
        if self.coordmap:
            embin_size+=8
        if self.light:
            self.fcn_emb = nn.Sequential(OrderedDict([
                ('0', torch.nn.Sequential(
                    ConvBatchNormReLU(embin_size, emb_size, 1, 1, 0, 1, leaky=leaky),)),
                ('1', torch.nn.Sequential(
                    ConvBatchNormReLU(embin_size, emb_size, 1, 1, 0, 1, leaky=leaky),)),
                ('2', torch.nn.Sequential(
                    ConvBatchNormReLU(embin_size, emb_size, 1, 1, 0, 1, leaky=leaky),)),
            ]))
            self.fcn_out = nn.Sequential(OrderedDict([
                ('0', torch.nn.Sequential(
                    nn.Conv2d(emb_size, 3*5, kernel_size=1),)),
                ('1', torch.nn.Sequential(
                    nn.Conv2d(emb_size, 3*5, kernel_size=1),)),
                ('2', torch.nn.Sequential(
                    nn.Conv2d(emb_size, 3*5, kernel_size=1),)),
            ]))
        else:
            self.fcn_emb = nn.Sequential(OrderedDict([
                ('0', torch.nn.Sequential(
                    ConvBatchNormReLU(embin_size, emb_size, 1, 1, 0, 1, leaky=leaky),
                    # Self_Attn(emb_size,'relu'),
                    ConvBatchNormReLU(emb_size, emb_size, 3, 1, 1, 1, leaky=leaky),
                    # Self_Attn(emb_size, 'relu'),
                    ConvBatchNormReLU(emb_size, emb_size, 1, 1, 0, 1, leaky=leaky),)),
                ('1', torch.nn.Sequential(
                    ConvBatchNormReLU(embin_size, emb_size, 1, 1, 0, 1, leaky=leaky),
                    # Self_Attn(emb_size, 'relu'),
                    ConvBatchNormReLU(emb_size, emb_size, 3, 1, 1, 1, leaky=leaky),
                    # Self_Attn(emb_size, 'relu'),
                    ConvBatchNormReLU(emb_size, emb_size, 1, 1, 0, 1, leaky=leaky),)),
                ('2', torch.nn.Sequential(
                    ConvBatchNormReLU(embin_size, emb_size, 1, 1, 0, 1, leaky=leaky),
                    # Self_Attn(emb_size, 'relu'),
                    ConvBatchNormReLU(emb_size, emb_size, 3, 1, 1, 1, leaky=leaky),
                    # Self_Attn(emb_size, 'relu'),
                    ConvBatchNormReLU(emb_size, emb_size, 1, 1, 0, 1, leaky=leaky),)),
            ]))
            self.fcn_out = nn.Sequential(OrderedDict([
                ('0', torch.nn.Sequential(
                    ConvBatchNormReLU(emb_size, emb_size//2, 1, 1, 0, 1, leaky=leaky),
                    nn.Conv2d(emb_size//2, 3, kernel_size=1),)),
                ('1', torch.nn.Sequential(
                    ConvBatchNormReLU(emb_size, emb_size//2, 1, 1, 0, 1, leaky=leaky),
                    nn.Conv2d(emb_size//2, 3, kernel_size=1),)),
                ('2', torch.nn.Sequential(
                    ConvBatchNormReLU(emb_size, emb_size//2, 1, 1, 0, 1, leaky=leaky),
                    nn.Conv2d(emb_size//2, 3, kernel_size=1),)),
            ]))
            if self.att:
                self.attn_emb=Self_Attn(4, emb_size, 'relu')
            # self.fcn_emb=torch.nn.Sequential(
            #         ConvBatchNormReLU(embin_size, emb_size, 1, 1, 0, 1, leaky=leaky),
            #         # Self_Attn(emb_size,'relu'),
            #         ConvBatchNormReLU(emb_size, emb_size, 3, 1, 1, 1, leaky=leaky),
            #         # Self_Attn(emb_size, 'relu'),
            #         ConvBatchNormReLU(emb_size, emb_size, 1, 1, 0, 1, leaky=leaky),)

            self.fcn_out_offset = nn.Sequential(OrderedDict([
                ('0', torch.nn.Sequential(
                    ConvBatchNormReLU(emb_size, emb_size//2, 1, 1, 0, 1, leaky=leaky),
                    nn.Conv2d(emb_size//2, 2, kernel_size=1),)),
                ('1', torch.nn.Sequential(
                    ConvBatchNormReLU(emb_size, emb_size//2, 1, 1, 0, 1, leaky=leaky),
                    nn.Conv2d(emb_size//2, 2, kernel_size=1),)),
                ('2', torch.nn.Sequential(
                    ConvBatchNormReLU(emb_size, emb_size//2, 1, 1, 0, 1, leaky=leaky),
                    nn.Conv2d(emb_size//2, 2, kernel_size=1),)),
            ]))
            self.fcn_out_center = nn.Sequential(OrderedDict([
                ('0', torch.nn.Sequential(
                    ConvBatchNormReLU(emb_size, emb_size, 1, 1, 0, 1, leaky=leaky),
                    ConvBatchNormReLU(emb_size, emb_size, 3, 1, 1, 1, leaky=leaky),
                    ConvBatchNormReLU(emb_size, emb_size//2, 1, 1, 0, 1, leaky=leaky),
                    nn.Conv2d(emb_size//2, 8*8, kernel_size=1),)),
                ('1', torch.nn.Sequential(
                    ConvBatchNormReLU(emb_size, emb_size, 1, 1, 0, 1, leaky=leaky),
                    ConvBatchNormReLU(emb_size, emb_size, 3, 1, 1, 1, leaky=leaky),
                    ConvBatchNormReLU(emb_size, emb_size//2, 1, 1, 0, 1, leaky=leaky),
                    nn.Conv2d(emb_size//2, 16*16, kernel_size=1),)),
                ('2', torch.nn.Sequential(
                    ConvBatchNormReLU(emb_size, emb_size, 1, 1, 0, 1, leaky=leaky),
                    ConvBatchNormReLU(emb_size, emb_size, 3, 1, 1, 1, leaky=leaky),
                    ConvBatchNormReLU(emb_size, emb_size, 1, 1, 0, 1, leaky=leaky),
                    nn.Conv2d(emb_size, 32*32, kernel_size=1),)),
            ]))

        # if self.seg:
        # self.segmentation=ReferCam()
        if self.seg:
            seg_emb_size=embin_size+3 #embin_size+3 #emb_size+3
            self.refine=nn.Sequential(OrderedDict([
                ('0', torch.nn.Sequential(
                    ConvBatchNormReLU(seg_emb_size, emb_size, 1, 1, 0, 1, leaky=leaky),
                    nn.Conv2d(emb_size, emb_size, kernel_size=1),)),
                ('1', torch.nn.Sequential(
                    ConvBatchNormReLU(seg_emb_size, emb_size, 1, 1, 0, 1, leaky=leaky),
                    nn.Conv2d(emb_size, emb_size, kernel_size=1),)),
                ('2', torch.nn.Sequential(
                    ConvBatchNormReLU(seg_emb_size, emb_size, 1, 1, 0, 1, leaky=leaky),
                    nn.Conv2d(emb_size, emb_size, kernel_size=1),)),
            ]))

        # self.PAMR_KERNEL = [1, 2, 4, 8, 12, 24]
        # self.PAMR_ITER = 10
        #
        # self._aff = PAMR(self.PAMR_ITER, self.PAMR_KERNEL)

    def forward(self, image, word_id, word_mask):
        ## Visual Module
        ## [1024, 13, 13], [512, 26, 26], [256, 52, 52]
        batch_size = image.size(0)
        raw_fvisu = self.visumodel(image)
        # raw_fvisu = [raw_fvisu[-1]]
        add_num=0
        fvisu = []
        for ii in range(len(raw_fvisu)):
            fvisu.append(self.mapping_visu._modules[str(ii+add_num)](raw_fvisu[ii]))
            fvisu[ii] = F.normalize(fvisu[ii], p=2, dim=1)

        ## Language Module
        if self.lstm:
            # max_len = (word_id != 0).sum(1).max().data[0]
            max_len = (word_id != 0).sum(1).max().item()
            word_id = word_id[:, :max_len]
            raw_flang = self.textmodel(word_id)
        else:
            all_encoder_layers, _ = self.textmodel(word_id, \
                token_type_ids=None, attention_mask=word_mask)
            ## Sentence feature at the first position [cls]
            raw_flang = (all_encoder_layers[-1][:,0,:] + all_encoder_layers[-2][:,0,:]\
                 + all_encoder_layers[-3][:,0,:] + all_encoder_layers[-4][:,0,:])/4
            ## fix bert during training
            raw_flang = raw_flang.detach()
        flang = self.mapping_lang(raw_flang)
        flang = F.normalize(flang, p=2, dim=1)

        flangvisu = []
        for ii in range(len(fvisu)):
            flang_tile = flang.view(flang.size(0), flang.size(1), 1, 1).\
                repeat(1, 1, fvisu[ii].size(2), fvisu[ii].size(3))
            if self.coordmap:
                coord = generate_coord(batch_size, fvisu[ii].size(2), fvisu[ii].size(3))
                flangvisu.append(torch.cat([fvisu[ii], flang_tile, coord], dim=1))
            else:
                flangvisu.append(torch.cat([fvisu[ii], flang_tile], dim=1))
        ## fcn
        supervised=False
        intmd_fea, outbox, cambox,cam_rv, attn_list = [], [] , [], [], []
        if supervised:
            for ii in range(len(fvisu)):
                intmd_fea.append(self.fcn_emb._modules[str(ii)](flangvisu[ii]))
                outbox.append(self.fcn_out._modules[str(ii)](intmd_fea[ii]))
        else:
            for ii in range(len(fvisu)):
                # if self.att:
                #     intmd, attn = self.attn_emb(self.fcn_emb._modules[str(ii+add_num)](flangvisu[ii]))
                #     intmd_fea.append(intmd)
                #     attn_list.append(attn)
                # else:
                intmd_fea.append(self.fcn_emb._modules[str(ii)](flangvisu[ii]))

                cam=self.fcn_out_center._modules[str(ii+add_num)](intmd_fea[ii])
                cambox.append(cam)
                if self.seg:
                    cam_rv.append(self.PCM(cam, flangvisu[ii],  image,ii+add_num))
                    # cam_rv.append(self.run_pamr(image, cam))

                out_center=F.adaptive_avg_pool2d(cam, (1, 1))
                out_offset=self.fcn_out_offset._modules[str(ii)](intmd_fea[ii])

                outbox.append(torch.cat([out_offset,out_center.view(out_center.size(0),1,int(out_center.size(1)**(1/2)),int(out_center.size(1)**(1/2)))],dim=1))




        # if self.seg:
        #     cam_rv=self.aggregation(cambox,flangvisu,image)
        # self.intmd_fea=intmd_fea
        return outbox,fvisu,flang,cambox,cam_rv,attn_list

    def aggregation(self,cambox,flangvisu,image):
        cam_inter=0.
        for ii in range(len(cambox)):
            cam=cambox[ii]
            f=flangvisu[ii]
            cam_rv=self.PCM(cam,f,image,ii)+cam_inter
            if ii<2:
                cam_inter=F.interpolate(cam_rv, (np.array(cam_rv.size()[-2:])*2).tolist(), mode="bilinear", align_corners=True)
                cam_inter=torch.repeat_interleave(cam_inter,4,dim=1)
        return cam_rv



    def run_pamr(self, im, mask):
        im = F.interpolate(im, mask.size()[-2:], mode="bilinear", align_corners=True)
        masks_dec = self._aff(im, max_norm(mask))
        return masks_dec

    def crf_inference(sigm_val, H, W, proc_im):

        sigm_val = np.squeeze(sigm_val)
        d = densecrf.DenseCRF2D(W, H, 2)
        U = np.expand_dims(-np.log(sigm_val + 1e-8), axis=0)
        U_ = np.expand_dims(-np.log(1 - sigm_val + 1e-8), axis=0)
        unary = np.concatenate((U_, U), axis=0)
        unary = unary.reshape((2, -1))
        d.setUnaryEnergy(unary)
        d.addPairwiseGaussian(sxy=3, compat=3)
        d.addPairwiseBilateral(sxy=20, srgb=3, rgbim=proc_im, compat=10)
        Q = d.inference(5)
        pred_raw_dcrf = np.argmax(Q, axis=0).reshape((H, W)).astype(np.float32)
        # predicts_dcrf = im_processing.resize_and_crop(pred_raw_dcrf, mask.shape[0], mask.shape[1])

        return pred_raw_dcrf

    def PCM(self, cam, f, x, scale_ii):

        n, c, h, w = cam.size()

        with torch.no_grad():
            cam_d = F.relu(cam.detach())
            cam_d_max = torch.max(cam_d.view(n,c,-1), dim=-1)[0].view(n,c,1,1)+1e-5
            cam_d_min = torch.min(cam_d.view(n, c, -1), dim=-1)[0].view(n, c, 1, 1)
            cam_d_norm = F.relu(cam_d-cam_d_min)/(cam_d_max-cam_d_min)
            # cam_d_norm[:,0,:,:] = 1-torch.max(cam_d_norm[:,1:,:,:], dim=1)[0]
            # cam_max = torch.max(cam_d_norm, dim=1, keepdim=True)[0]
            # cam_d_norm[cam_d_norm < cam_max] = 0


        # n, c, h, w = f.size()
        # cam = F.interpolate(cam, (h, w), mode='bilinear', align_corners=True).view(n, -1, h * w)
        # f = self.bilinear_att(f,lang)
        x_s = F.interpolate(x, (h, w), mode='bilinear', align_corners=True)
        f = torch.cat([x_s, f.detach()], dim=1)
        f = self.refine._modules[str(scale_ii)](f)
        f = f.view(n, -1, h * w)
        f = f / (torch.norm(f, dim=1, keepdim=True) + 1e-5)

        aff = F.relu(torch.matmul(f.transpose(1, 2), f), inplace=True)
        aff = aff / (torch.sum(aff, dim=1, keepdim=True) + 1e-5)

        # cam_d_norm = cam_d_norm.view(n,-1,h*w)
        # cam_r = cam_d_norm.transpose(1,2)
        # cam_r=torch.matmul(cam_r, aff)
        # # cam_rv = cam_r.transpose(1, 2).view(n, -1, h, w)
        #
        # cam_rv = torch.matmul(cam_r.transpose(1,2), aff).view(n, -1, h, w)

        cam_rv = torch.matmul(cam_d_norm.view(n,-1,h*w), aff).view(n, -1, h, w)

        return cam_rv

    # def bilinear_att(self, f, lang):
    #
    #     n, c, h, w = f.size()
    #     f = f.view(n,-1,h*w)
    #     f = f / (torch.norm(f, dim=1, keepdim=True) + 1e-5) # n x c x hw
    #
    #     lang=self.compress_lang(lang.view(n,-1)).view(n,1,-1) # n x 1 x c
    #
    #     aff = F.relu(torch.matmul(lang, f), inplace=True) # n x 1 x hw
    #     aff = aff / (torch.sum(aff, dim=2, keepdim=True) + 1e-5)
    #     cam_rv = f*aff # +f #torch.matmul(f, aff)
    #
    #
    #     return cam_rv

    # def forward_seg(self,bbox,pred_anchor,gi, gj, best_n_list):
    #     self.segmentation(self.intmd_fea,bbox,pred_anchor,gi, gj, best_n_list)

if __name__ == "__main__":
    import sys
    import argparse
    sys.path.append('.')
    from dataset.referit_loader import *
    from torch.autograd import Variable
    from torch.utils.data import DataLoader
    from torchvision.transforms import Compose, ToTensor, Normalize
    from utils.transforms import ResizeImage, ResizeAnnotation
    parser = argparse.ArgumentParser(
        description='Dataloader test')
    parser.add_argument('--size', default=416, type=int,
                        help='image size')
    parser.add_argument('--data', type=str, default='./ln_data/',
                        help='path to ReferIt splits data folder')
    parser.add_argument('--dataset', default='referit', type=str,
                        help='referit/flickr/unc/unc+/gref')
    parser.add_argument('--split', default='train', type=str,
                        help='name of the dataset split used to train')
    parser.add_argument('--time', default=20, type=int,
                        help='maximum time steps (lang length) per batch')
    parser.add_argument('--emb_size', default=256, type=int,
                        help='word embedding dimensions')
    # parser.add_argument('--lang_layers', default=3, type=int,
    #                     help='number of SRU/LSTM stacked layers')

    args = parser.parse_args()

    torch.manual_seed(13)
    np.random.seed(13)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    input_transform = Compose([
        ToTensor(),
        # ResizeImage(args.size),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    refer = ReferDataset(data_root=args.data,
                         dataset=args.dataset,
                         split=args.split,
                         imsize = args.size,
                         transform=input_transform,
                         max_query_len=args.time)

    train_loader = DataLoader(refer, batch_size=2, shuffle=True,
                              pin_memory=True, num_workers=1)

    model = textcam_yolo_light(emb_size=args.emb_size)

    for batch_idx, (imgs, word_id, word_mask, bbox) in enumerate(train_loader):
        image = Variable(imgs)
        word_id = Variable(word_id)
        word_mask = Variable(word_mask)
        bbox = Variable(bbox)
        bbox = torch.clamp(bbox,min=0,max=args.size-1)

        pred_anchor_list = model(image, word_id, word_mask)
        for pred_anchor in pred_anchor_list:
            print(pred_anchor)
            print(pred_anchor.shape)
