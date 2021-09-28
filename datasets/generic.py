import os
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize, InterpolationMode

from datasets.transform import MyCoTransform, Colorize
import utils.visualize


def root_check(path):
    if not os.path.exists(path):
        print("ERROR: Path does not exist: {}".format(path))
        print("Please make sure that the root path is correctly set for your dataset. "
              "For instance, CITYSCAPES_ROOT is set in datasets/cityscapes.py")
        return False

    return True


class GenericDataset(Dataset):

    def __init__(self, args, images_root, labels_root, imgseq_root, filenames, filenames_gt, classes_dict,
                 orig_res, work_res, target_transform, normalize_tensor, colormap,
                 co_transform, shallow_dec=False, augment=False, interval=None, print_all_logs=True):

        if interval is None:
            interval = [0, 0]
        assert(len(interval) == 2 and interval[0] >= 0 and interval[1] >= 0)

        if not root_check(images_root) or \
           not root_check(labels_root) or \
           not root_check(imgseq_root):
            exit(1)

        self.images_root = images_root
        self.labels_root = labels_root
        self.imgseq_root = imgseq_root
        self.filenames = filenames
        self.filenames_gt = filenames_gt
        self.classes_dict = classes_dict
        self.orig_res = orig_res
        self.work_res = work_res
        self.target_transform = target_transform
        self.normalize_tensor = normalize_tensor

        self.num_classes = len(classes_dict)
        self.colorize = Colorize(colormap, self.num_classes)

        self.backbone = args.backbone
        self.random_crop = args.random_crop
        self.use_orig_res = args.use_orig_res
        self.interval = interval

        self.filenames.sort()
        self.filenames_gt.sort()

        self.input_images_size = len(self.filenames)
        self.gt_images_size = len(self.filenames_gt)

        self.co_transform = None
        if co_transform:
            self.co_transform = MyCoTransform(self.target_transform, shallow_dec, augment=augment,
                                              work_res=self.work_res, random_crop=self.random_crop)
        if print_all_logs:
            utils.visualize.print_summary(self.__dict__, self.__class__.__name__)

    def __getitem__(self, index):
        file_path = self.filenames[index]
        file_path_gt = self.filenames_gt[index]

        labels = []
        images = []
        orig_labels = []
        orig_images = []

        images_filenames = []
        labels_filenames = []

        # Extracting images
        for i in reversed(range(-self.interval[1], self.interval[0] + 1)):
            # Labels
            new_file_path_gt = self.filename_from_index(file_path_gt, i)
            abs_file_path_gt = os.path.join(self.labels_root, new_file_path_gt)

            label = None
            orig_label = None
            new_file_path_gt = ""

            if not os.path.exists(abs_file_path_gt):
                if i == self.interval[1]:
                    print(abs_file_path_gt, "does not exist !")
                    exit(1)
            else:
                with open(abs_file_path_gt, 'rb') as f:
                    label = Image.open(f).convert('P')
                    label = Resize(self.work_res, InterpolationMode.NEAREST)(label)
                    if self.use_orig_res:
                        orig_label = Image.open(f).convert('P')

            labels.append(label)
            labels_filenames.append(new_file_path_gt)
            if self.use_orig_res:
                orig_labels.append(orig_label)

            # Images
            new_file_path = self.filename_from_index(file_path, i)
            abs_file_path = os.path.join(self.imgseq_root, new_file_path)
            if not os.path.exists(abs_file_path):
                print(abs_file_path, "does not exist !")
                exit(1)
            with open(abs_file_path, 'rb') as f:
                image = Image.open(f).convert('RGB')
                if self.use_orig_res:
                    orig_img = Image.open(f).convert('RGB')

            image = Resize(self.work_res, InterpolationMode.BILINEAR)(image)
            images.append(image)
            images_filenames.append(new_file_path)
            if self.use_orig_res:
                orig_images.append(ToTensor()(orig_img))

        # Transforming images and labels
        if self.co_transform is not None:
            images, labels = self.co_transform(images, labels)

        for i, image in enumerate(images):
            # Converting to labels
            if labels[i] is not None:
                labels[i] = self.target_transform(labels[i])
            if self.use_orig_res and orig_labels[i] is not None:
                orig_labels[i] = self.target_transform(orig_labels[i])

            if self.co_transform is None:
                images[i] = ToTensor()(image)

            if self.backbone == "psp101":
                images[i] = self.target_transform(images[i])

        images = torch.stack(images)

        labels = labels[self.interval[0]].unsqueeze(0)
        if self.use_orig_res:
            orig_labels = orig_labels[self.interval[0]].unsqueeze(0)
            orig_images = torch.stack(orig_images)

        return images, labels, orig_images, orig_labels, \
               file_path, file_path_gt, images_filenames, labels_filenames

    def __len__(self):
        return len(self.filenames)

    def filename_from_index(self, base_file_path, index):
        assert(False and "Not implemented for base class")
        return ""
