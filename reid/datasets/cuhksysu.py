from __future__ import print_function, absolute_import
import os.path as osp

from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json


class CUHKSYSU(Dataset):

    md5 = 'f7eeb1669d77d5b73e6dfba7f087570b'

    def __init__(self, root, split_id=0, num_val=0, download=True):
        super(CUHKSYSU, self).__init__(root, split_id=split_id)
        
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. " +
                               "You can use download=True to download it.")

        self.load(num_val)

    def download(self):
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        import hashlib
        import numpy as np
        from zipfile import ZipFile
        from scipy.io import loadmat
        from scipy.misc import imread
        from scipy.misc import imsave

        raw_dir = osp.join(self.root, 'raw')
        mkdir_if_missing(raw_dir)

        # Download the raw zip file
        fpath = osp.join(raw_dir, 'dataset.zip')
        if osp.isfile(fpath) and \
          hashlib.md5(open(fpath, 'rb').read()).hexdigest() == self.md5:
            print("Using downloaded file: " + fpath)
        else:
            raise RuntimeError("Please request the dataset from sli@ee.cuhk.edu.hk "
                               "or xiaotong@ee.cuhk.edu.hk (academic only).")
        
        # Extract the file
        exdir = osp.join(raw_dir, 'dataset')
        if not osp.isdir(exdir):
            print("Extracting zip file")
            with ZipFile(fpath) as z:
                z.extractall(path=raw_dir)
        
        # Format
        images_dir = osp.join(self.root, 'images')
        mkdir_if_missing(images_dir)

        # Train: 5532  Test: 2900
        # Randomly choose half of the images as cam_0, others as cam_1
        identities = [[[] for _ in range(2)] for _ in range(8432)]

        trainval_pids = set()
        train = loadmat(osp.join(exdir, 'annotation/test/train_test', 'Train.mat'))
        train = train['Train'].squeeze()
        for pid, item in enumerate(train):
            scenes = item[0, 0][2].squeeze()
            images = []
            for im_name, box, _ in scenes:
                im_name = str(im_name[0])
                image = imread(osp.join(exdir, 'Image/SSM', im_name))
                box = box.squeeze().astype(np.int32)
                crop_image = image[int(box[1]):int(box[1]) + int(box[3]), int(box[0]):int(box[0]) + int(box[2])]
                images.append(crop_image)
            num = len(images)
            np.random.shuffle(images)
            # cam 0
            for image in images[(num // 2):]:
                cam = 0
                fname = ('{:08d}_{:02d}_{:05d}.png'
                         .format(pid, cam, len(identities[pid][cam])))
                identities[pid][cam].append(fname)
                imsave(osp.join(images_dir, fname), image)
            # cam 1
            for image in images[:(num // 2)]:
                cam = 1
                fname = ('{:08d}_{:02d}_{:05d}.png'
                         .format(pid, cam, len(identities[pid][cam])))
                identities[pid][cam].append(fname)
                imsave(osp.join(images_dir, fname), image)
            trainval_pids.add(pid)

        assert len(trainval_pids) == 5532
        test_pids = set()

        # Save meta information into a json file
        meta = {'name': 'CUHKSYSU', 'shot': 'multiple', 'num_cameras': 2,
                'identities': identities}
        write_json(meta, osp.join(self.root, 'meta.json'))

        # Save the only training / test split
        splits = [{
            'trainval': sorted(list(trainval_pids)),
            'query': sorted(list(test_pids)),
            'gallery': sorted(list(test_pids))}]
        write_json(splits, osp.join(self.root, 'splits.json'))

