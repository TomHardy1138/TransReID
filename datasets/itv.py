import os.path as osp

from .bases import BaseImageDataset
from torchvision.datasets import ImageFolder


class Common(BaseImageDataset):
    dataset_dir = "common"
    test_dataset_dir = "common_test"

    def __init__(self, root='', verbose=False, **kwargs):
        super(Common, self).__init__()
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.test_dataset_dir = osp.join(self.root, self.test_dataset_dir)

        images = ImageFolder(self.dataset_dir)
        test_images = ImageFolder(self.test_dataset_dir)

        self._check_before_run()

        train, query, gallery, gpids = [], [], [], []
        label = {}
        label_count = -1

        for image, _ in images.imgs:
            label_tmp = osp.dirname(image)
            if label_tmp not in label.keys():
                label_count += 1
                label[label_tmp] = label_count
                lb = label_count
            else:
                lb = label[label_tmp]
            train.append((image, lb, len(train) + 1, 1))

        for image, pid in test_images.imgs:
            if pid not in gpids:
                gpids.append(pid)
                gallery.append((image, pid, 0, 1))
            else:
                query.append((image, pid, len(query) + 2, 1))

        self.train = train
        self.gallery = gallery
        self.query = query

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

        if verbose:
            self.print_dataset_statistics(train, query, gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.test_dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dataset_dir))
