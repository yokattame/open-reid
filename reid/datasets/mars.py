from __future__ import print_function, absolute_import
import os.path as osp

from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json


class MARS(Dataset):
    url_train = 'https://drive.google.com/file/d/0B6tjyrV1YrHeY0hsVExLOTk3eVU/view'
    md5_train = 'f4a3c5967a1b440ccf770f388c609602'
    
    url_test = 'https://drive.google.com/file/d/0B6tjyrV1YrHeTEE2c2hFMTdpRFU/view'
    md5_test = '950d3bfbd792103d12729ac4f57199cf'

    relabel = {0: 0}
    new_label = 1

    def __init__(self, root, split_id=0, num_val=0, download=True):
        super(MARS, self).__init__(root, split_id=split_id)

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
        import shutil
        import copy
        from glob import glob
        from zipfile import ZipFile

        raw_dir = osp.join(self.root, 'raw')
        mkdir_if_missing(raw_dir)

        # Download the raw zip file
        fpath_train = osp.join(raw_dir, 'bbox_train.zip')
        if osp.isfile(fpath_train) and \
          hashlib.md5(open(fpath_train, 'rb').read()).hexdigest() == self.md5_train:
            print("Using downloaded file: " + fpath_train)
        else:
            raise RuntimeError("Please download the dataset manually from {} "
                               "to {}".format(self.url_train, fpath_train))
        
        fpath_test = osp.join(raw_dir, 'bbox_test.zip')
        if osp.isfile(fpath_test) and \
          hashlib.md5(open(fpath_test, 'rb').read()).hexdigest() == self.md5_test:
            print("Using downloaded file: " + fpath_test)
        else:
            raise RuntimeError("Please download the dataset manually from {} "
                               "to {}".format(self.url_test, fpath_test))

        # Extract the file
        exdir_train = osp.join(raw_dir, 'bbox_train')
        if not osp.isdir(exdir_train):
            print("Extracting zip file")
            with ZipFile(fpath_train) as z:
                z.extractall(path=raw_dir)
        
        exdir_test = osp.join(raw_dir, 'bbox_test')
        if not osp.isdir(exdir_test):
            print("Extracting zip file")
            with ZipFile(fpath_test) as z:
                z.extractall(path=raw_dir)

        # Format
        images_dir = osp.join(self.root, 'images')
        mkdir_if_missing(images_dir)

        # 1259 identities (+1 for background) with 6 camera views each
        identities = [[[] for _ in range(6)] for _ in range(1260)]

        def register(exdir):
            fpaths = sorted(glob(osp.join(exdir, '*/*.jpg')))
            pids = set()
            for fpath in fpaths:
                fname = osp.basename(fpath)
                pid = fname[:4]
                if "-" in pid: continue # junk images are just ignored
                pid = int(pid)
                cam = int(fname[5])
                if pid not in self.relabel:
                    self.relabel[pid] = self.new_label
                    self.new_label += 1
                pid = self.relabel[pid]
                assert 0 <= pid <= 1259  # pid == 0 means background
                assert 1 <= cam <= 6
                cam -= 1
                pids.add(pid)
                fname = ('{:08d}_{:02d}_{:05d}.jpg'
                         .format(pid, cam, len(identities[pid][cam])))
                identities[pid][cam].append(fname)
                shutil.copy(fpath, osp.join(images_dir, fname))
            return pids
        
        trainval_pids = register(exdir_train)
        test_pids = register(exdir_test)
        assert self.new_label == 1260
        gallery_pids = copy.copy(test_pids)
        test_pids.remove(0)
        query_pids = test_pids
        assert trainval_pids.isdisjoint(test_pids)

        # Save meta information into a json file
        meta = {'name': 'MARS', 'shot': 'multiple', 'num_cameras': 6,
                'identities': identities}
        write_json(meta, osp.join(self.root, 'meta.json'))

        # Save the only training / test split
        splits = [{
            'trainval': sorted(list(trainval_pids)),
            'query': sorted(list(query_pids)),
            'gallery': sorted(list(gallery_pids))}]
        write_json(splits, osp.join(self.root, 'splits.json'))

