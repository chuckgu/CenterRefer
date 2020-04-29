import sys
# sys.path.append('./external/coco/PythonAPI')

base_dir="/shared/CenterRefer/"
# base_dir="/home/tips/Desktop/project/CenterRefer/"

sys.path.append(base_dir+'external/coco/PythonAPI')
sys.path.append(base_dir+'external/coco/PythonAPI/pycocotools')
sys.path.append(base_dir+'external/refer')

# sys.path.append('/shared/CenterRefer/external/coco/PythonAPI')
# sys.path.append('/shared/CenterRefer/external/coco/PythonAPI/pycocotools')
# sys.path.append('/shared/CenterRefer/external/refer')
import torch
import numpy as np
import torch
from torch.utils.data import Dataset
# from mypath import Path
from tqdm import trange
import os
from refer import REFER
from pycocotools.coco import COCO
from pycocotools import mask
from torchvision import transforms
from dataloaders import custom_transforms as tr
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from scipy import ndimage
from transformers import *
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import skimage
import cv2


Path=base_dir+'data/coco/'
im_type = 'train2014'

def generate_heatmap(_target,center):
    sigma = 10
    spread = 4
    extent = int(spread * sigma)
    gaussian_heatmap = np.zeros(_target.shape, dtype=np.float32)
    height = gaussian_heatmap.shape[0]
    width = gaussian_heatmap.shape[1]

    for i in range(2 * extent):
        for j in range(2 * extent):
            point_y = center[0] - extent + i
            point_x = center[1] - extent + j
            # print(point_y,point_x)
            if point_y >= height or point_y < 0 or point_x >= width or point_x < 0:
                continue

            gaussian_heatmap[int(point_y), int(point_x)] = 1 / 2 / np.pi / (sigma ** 2) * np.exp(
                -1 / 2 * ((i - spread * sigma - 0.5) ** 2 + (j - spread * sigma - 0.5) ** 2) / (sigma ** 2))

    _gaussian_heatmap = (gaussian_heatmap / np.max(gaussian_heatmap))

    return _gaussian_heatmap


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    y, x = center
    y=int(y)
    x=int(x)

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)



class COCOSegmentation(Dataset):
    NUM_CLASSES = 1
    CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4,
        1, 64, 20, 63, 7, 72]

    def __init__(self,
                 args,
                 # base_dir=Path,
                 split='train',
                 year='2014',
                 dataset='Gref',
                 b_test=None):
        super().__init__()
        # ann_file = os.path.join(base_dir, 'annotations/instances_{}{}.json'.format(split, year))
        bert_file = os.path.join(Path, 'annotations/{}_bert_{}.pth'.format(split, year))
        self.img_dir = os.path.join(Path, 'images/{}{}'.format(split, year))
        self.split = split
        self.h=320
        self.w=320
        self.b_test=b_test

        # im_dir = './data/coco/images'
        # im_type = 'train2014'
        vocab_file = base_dir+'data/vocabulary_Gref.txt'

        if dataset == 'Gref':
            refer = REFER(base_dir+'external/refer/data', dataset='refcocog', splitBy='google')
        elif dataset == 'unc':
            refer = REFER(base_dir+'external/refer/data', dataset='refcoco', splitBy='unc')
        elif dataset == 'unc+':
            refer = REFER(base_dir+'external/refer/data', dataset='refcoco+', splitBy='unc')
        else:
            raise ValueError('Unknown dataset %s' % dataset)

        refs = [refer.Refs[ref_id] for ref_id in refer.Refs if refer.Refs[ref_id]['split'] == split]
        # vocab_dict = text_processing.load_vocab_dict_from_file(vocab_file)

        # self.coco = COCO(ann_file)
        # self.coco_mask = mask
        self.refer=refer
        self.refs=refs
        self.bert_folder = os.path.join(Path, 'annotations/{}_bert_{}'.format(split, year))

        if os.path.exists(bert_file):
            file=torch.load(bert_file)
            self.bert_emb = file["bert_emb"]
            self.embToref = file["embToref"]
        else:
            self.bert_emb, self.embToref = self._preprocess(refs, bert_file)

        self.args = args

    def __getitem__(self, index):
        _img, _target, _text_emb, _center = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target, 'text': _text_emb, 'center': _center}

        if self.split == "train":
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)

    def _make_img_gt_point_pair(self, index):
        ref_index=self.embToref[index]
        refer = self.refer
        ref = self.refs[ref_index]
        _text_emb=np.load(os.path.join(self.bert_folder,self.bert_emb[index]))
        # img_metadata = coco.loadImgs(img_id)[0]
        im_name = 'COCO_' + im_type + '_' + str(ref['image_id']).zfill(12)

        _img = Image.open('%s/%s/%s/%s.jpg' % (Path,"images", im_type, im_name)).convert('RGB')
        # cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        # _target = Image.fromarray(self._gen_seg_mask(
        #     cocotarget, img_metadata['height'], img_metadata['width']))
        seg = refer.Anns[ref['ann_id']]['segmentation']
        rle = mask.frPyObjects(seg, _img.height, _img.width)
        _target= np.max(mask.decode(rle), axis=2).astype(np.float32)
        center = ndimage.measurements.center_of_mass(_target) #(y,x)

        scale = min(80 / _target.shape[0], 80 / _target.shape[1])

        center=np.asarray([int(center[0]*scale),int(center[1]*scale)]).astype(np.float32)
        # center=np.asarray([int(center[0]),int(center[1])]).astype(np.float32)

        # _gaussian_heatmap = generate_heatmap(_target,center)

        _gaussian_heatmap = np.zeros([80,80], dtype=np.float32)
        draw_gaussian(_gaussian_heatmap, center, 5)

        # _img = np.asarray(_img)

        # _gaussian_heatmap = Image.fromarray(gaussian_heatmap)
        # _target = Image.fromarray(_target)
        # print(center)
        vis=False

        if vis:
            target=(_target*255)[:,:,None].repeat(3,2)
            # dst = np.asarray(_img) * 0.5 + _target*0.5
            gaussian_heatmap=_gaussian_heatmap[:,:,None]*255
            gaussian_heatmap= np.concatenate((gaussian_heatmap, np.zeros((gaussian_heatmap.shape[0],gaussian_heatmap.shape[1],2))), axis=2)

            dst=np.asarray(_img) * 0.5 + target*0.5+0.5*gaussian_heatmap

            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (30, 30)
            fontScale = 0.7
            fontColor = (255, 255, 255)
            lineType = 2

            cv2.putText(dst, ref['sentences'][0]['sent'],
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)

            Image.fromarray(dst.astype(np.uint8)).save(base_dir+'output/'+im_name+".jpg")


        return _img, _target, _text_emb, _gaussian_heatmap

    def _preprocess(self, refs, ids_file):
        print("Preprocessing to extract embedding from the BERT, this will take a while. " + \
              "But don't worry, it only run once for each split.")
        bert_folder=self.bert_folder
        model_class = BertModel
        tokenizer_class = BertTokenizer
        pretrained_weights = "bert-base-uncased"

        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights)

        if not os.path.isdir(bert_folder):
            os.makedirs(bert_folder)

        tbar = trange(len(refs))
        new_ids = []
        embToref={}
        emb_id=0
        for i in tbar:
            ref=refs[i]
            for sentence in ref['sentences']:
                sent = sentence['sent']
                input_ids = torch.tensor([tokenizer.encode(sent, add_special_tokens=True, max_length=20 ,pad_to_max_length=True)])
                with torch.no_grad():
                    last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples
                bert_file=os.path.join(bert_folder, '{}.npy'.format(emb_id))
                new_ids.append('{}.npy'.format(emb_id))
                np.save(bert_file, last_hidden_states.numpy())

                embToref[emb_id]=i
                emb_id=emb_id+1
        save_dict={}
        save_dict["embToref"]=embToref
        save_dict["bert_emb"] = new_ids
        torch.save(save_dict, ids_file)
        print("saved")
                # tbar = trange(len(ids))
        # new_ids = []
        # for i in tbar:
        #     img_id = ids[i]
        #     cocotarget = self.refer.loadAnns(self.refer.getAnnIds(imgIds=img_id))
        #     img_metadata = self.refer.loadImgs(img_id)[0]
        #     mask = self._gen_seg_mask(cocotarget, img_metadata['height'],
        #                               img_metadata['width'])
        #     # more than 1k pixels
        #     if (mask > 0).sum() > 1000:
        #         new_ids.append(img_id)
        #     tbar.set_description('Doing: {}/{}, got {} qualified images'. \
        #                          format(i, len(ids), len(new_ids)))
        # print('Found number of qualified images: ', len(new_ids))
        # torch.save(new_ids, ids_file)
        return new_ids,embToref

    def _gen_seg_mask(self, target, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        coco_mask = self.coco_mask
        for instance in target:
            rle = coco_mask.frPyObjects(instance['segmentation'], h, w)
            m = coco_mask.decode(rle)
            cat = instance['category_id']
            if cat in self.CAT_LIST:
                c = self.CAT_LIST.index(cat)
            else:
                continue
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
        return mask

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            # tr.RandomHorizontalFlip(),
            # tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            # tr.RandomGaussianBlur(),
            tr.FixScalePad(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            # tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.FixScalePad(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)


    def __len__(self):
        if self.b_test is None:
            return len(self.bert_emb)
        else:
            return int(len(self.bert_emb)*self.b_test)





if __name__ == "__main__":
    from dataloaders import custom_transforms as tr
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    coco_val = COCOSegmentation(args, split='val', year='2017')

    dataloader = DataLoader(coco_val, batch_size=4, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='coco')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)