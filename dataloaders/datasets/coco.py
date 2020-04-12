import sys
# sys.path.append('./external/coco/PythonAPI')

sys.path.append('/home/tips/Desktop/project/CenterRefer/external/coco/PythonAPI')
sys.path.append('/home/tips/Desktop/project/CenterRefer/external/coco/PythonAPI/pycocotools')
sys.path.append('/home/tips/Desktop/project/CenterRefer/external/refer')
# sys.path.append('/shared/CenterRefer/external/coco/PythonAPI')
# sys.path.append('/shared/CenterRefer/external/coco/PythonAPI/pycocotools')
# sys.path.append('/shared/CenterRefer/external/refer')

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

# base_dir="/shared/CenterRefer/"
base_dir="/home/tips/Desktop/project/CenterRefer/"
Path=base_dir+'data/coco/'
im_type = 'train2014'

class COCOSegmentation(Dataset):
    NUM_CLASSES = 21
    CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4,
        1, 64, 20, 63, 7, 72]

    def __init__(self,
                 args,
                 # base_dir=Path,
                 split='train',
                 year='2017',
                 dataset='Gref'):
        super().__init__()
        # ann_file = os.path.join(base_dir, 'annotations/instances_{}{}.json'.format(split, year))
        # ids_file = os.path.join(base_dir, 'annotations/{}_ids_{}.pth'.format(split, year))
        self.img_dir = os.path.join(Path, 'images/{}{}'.format(split, year))
        self.split = split

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
        # if os.path.exists(ids_file):
        #     self.ids = torch.load(ids_file)
        # else:
        #     ids = list(self.coco.imgs.keys())
        #     self.ids = self._preprocess(ids, ids_file)

        self.args = args

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        if self.split == "train":
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)

    def _make_img_gt_point_pair(self, index):
        refer = self.refer
        ref = self.refs[index]
        # img_metadata = coco.loadImgs(img_id)[0]
        im_name = 'COCO_' + im_type + '_' + str(ref['image_id']).zfill(12)

        _img = Image.open('%s/%s/%s/%s.jpg' % (Path,"images", im_type, im_name)).convert('RGB')
        # cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        # _target = Image.fromarray(self._gen_seg_mask(
        #     cocotarget, img_metadata['height'], img_metadata['width']))
        seg = refer.Anns[ref['ann_id']]['segmentation']
        rle = mask.frPyObjects(seg, _img.height, _img.width)
        _target= np.max(mask.decode(rle), axis=2).astype(np.float32)
        center=ndimage.measurements.center_of_mass(_target)
        _target = Image.fromarray(_target)
        # print(center)
        return _img, _target

    def _preprocess(self, ids, ids_file):
        print("Preprocessing mask, this will take a while. " + \
              "But don't worry, it only run once for each split.")
        tbar = trange(len(ids))
        new_ids = []
        for i in tbar:
            img_id = ids[i]
            cocotarget = self.refer.loadAnns(self.refer.getAnnIds(imgIds=img_id))
            img_metadata = self.refer.loadImgs(img_id)[0]
            mask = self._gen_seg_mask(cocotarget, img_metadata['height'],
                                      img_metadata['width'])
            # more than 1k pixels
            if (mask > 0).sum() > 1000:
                new_ids.append(img_id)
            tbar.set_description('Doing: {}/{}, got {} qualified images'. \
                                 format(i, len(ids), len(new_ids)))
        print('Found number of qualified images: ', len(new_ids))
        torch.save(new_ids, ids_file)
        return new_ids

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
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)


    def __len__(self):
        return len(self.refs)



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