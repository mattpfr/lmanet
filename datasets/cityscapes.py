import os
import numpy as np

from torchvision.transforms import Compose, Normalize

from datasets.generic import GenericDataset
from datasets.transform import Relabel, ToLabel

###############################################################################################
# Sets the path to the root Cityscapes folder containing the folder "leftImg8bit"
CITYSCAPES_ROOT = "/please/set/cityscapes/root/path/here/"
# Sets the path to the root Cityscapes folder containing the folder "leftImg8bit_sequence"
CITYSCAPES_SEQ_ROOT = "/please/set/cityscapes/root/path/here/"

###############################################################################################

normalize_tensor_cityscapes = Compose([Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

target_transform_cityscapes = Compose([ToLabel(), Relabel(255, 19)])

CITYSCAPES_CLASSES_DICT = {
    "Road"         : {"weight_enc": 2.3653597831726, "weight": 2.8149201869965},   # 0
    "sidewalk"     : {"weight_enc": 4.4237880706787, "weight": 6.9850029945374},   # 1
    "building"     : {"weight_enc": 2.9691488742828, "weight": 3.7890393733978},   # 2
    "wall"         : {"weight_enc": 5.3442072868347, "weight": 9.9428062438965},   # 3
    "fence"        : {"weight_enc": 5.2983593940735, "weight": 9.7702074050903},   # 4
    "pole"         : {"weight_enc": 5.2275490760803, "weight": 9.5110931396484},   # 5
    "traffic light": {"weight_enc": 5.4394111633301, "weight": 10.311357498169},   # 6
    "traffic sign" : {"weight_enc": 5.3659925460815, "weight": 10.026463508606},   # 7
    "vegetation"   : {"weight_enc": 3.4170460700989, "weight": 4.6323022842407},   # 8
    "terrain"      : {"weight_enc": 5.2414722442627, "weight": 9.5608062744141},   # 9
    "sky"          : {"weight_enc": 4.7376127243042, "weight": 7.8698215484619},   # 10
    "person"       : {"weight_enc": 5.2286224365234, "weight": 9.5168733596802},   # 11
    "rider"        : {"weight_enc": 5.455126285553,  "weight": 10.373730659485},   # 12
    "car"          : {"weight_enc": 4.3019247055054, "weight": 6.6616044044495},   # 13
    "truck"        : {"weight_enc": 5.4264230728149, "weight": 10.260489463806},   # 14
    "bus"          : {"weight_enc": 5.4331531524658, "weight": 10.287888526917},   # 15
    "train"        : {"weight_enc": 5.433765411377,  "weight": 10.289801597595},   # 16
    "motorcycle"   : {"weight_enc": 5.4631009101868, "weight": 10.405355453491},   # 17
    "bicycle"      : {"weight_enc": 5.3947434425354, "weight": 10.138095855713},   # 18
    "IGNORE"       : {"weight_enc": 0.0,             "weight": 0.0},               # 19
}

def colormap_cityscapes():
    cmap = np.zeros([256, 3]).astype(np.uint8)
    cmap[0,:] = np.array([128, 64,128])
    cmap[1,:] = np.array([244, 35,232])
    cmap[2,:] = np.array([ 70, 70, 70])
    cmap[3,:] = np.array([ 102,102,156])
    cmap[4,:] = np.array([ 190,153,153])
    cmap[5,:] = np.array([ 153,153,153])

    cmap[6,:] = np.array([ 250,170, 30])
    cmap[7,:] = np.array([ 220,220,  0])
    cmap[8,:] = np.array([ 107,142, 35])
    cmap[9,:] = np.array([ 152,251,152])
    cmap[10,:] = np.array([ 70,130,180])

    cmap[11,:] = np.array([ 220, 20, 60])
    cmap[12,:] = np.array([ 255,  0,  0])
    cmap[13,:] = np.array([ 0,  0,142])
    cmap[14,:] = np.array([  0,  0, 70])
    cmap[15,:] = np.array([  0, 60,100])

    cmap[16,:] = np.array([  0, 80,100])
    cmap[17,:] = np.array([  0,  0,230])
    cmap[18,:] = np.array([ 119, 11, 32])
    cmap[19,:] = np.array([ 0,  0,  0])

    return cmap


class Cityscapes(GenericDataset):

    def __init__(self, args, subset, co_transform, shallow_dec=False, augment=False, interval=None, load_train_ids=True, print_all_logs=True):

        images_root = os.path.join(CITYSCAPES_ROOT, 'leftImg8bit', subset)
        labels_root = os.path.join(CITYSCAPES_ROOT, 'gtFine', subset)
        imgseq_root = os.path.join(CITYSCAPES_SEQ_ROOT, 'leftImg8bit_sequence', subset)

        filenames = [os.path.join(dp, f) for dp, dn, fn in
                          os.walk(os.path.expanduser(images_root))
                          for f in fn if any(f.endswith(ext) for ext in ['.jpg', '.png'])]

        filenames_gt = [os.path.join(dp, f) for dp, dn, fn in
                            os.walk(os.path.expanduser(labels_root))
                            for f in fn if f.endswith("_labelTrainIds.png")]

        classes_dict = CITYSCAPES_CLASSES_DICT

        orig_res = (1024, 2048)
        if args.backbone == "erf":
            work_res = (512, 1024)
        elif args.backbone == "psp101":
            work_res = (1025, 2049)
        else:
            work_res = (-1, -1)
            print("unknown backbone")
            exit(0)

        target_transform = target_transform_cityscapes
        normalize_tensor = normalize_tensor_cityscapes

        super(Cityscapes, self).__init__(
            args, images_root, labels_root, imgseq_root, filenames, filenames_gt, classes_dict,
            orig_res, work_res, target_transform, normalize_tensor, colormap_cityscapes(),
            co_transform, shallow_dec, augment, interval, print_all_logs)

    def filename_from_index(self, base_file_path, index):
        dir_name = os.path.basename(os.path.dirname(base_file_path))
        file_name = os.path.basename(base_file_path)
        name_elts = file_name.split('_')
        old_num = int(name_elts[2])
        new_num = old_num - index
        name_elts[2] = "{:06d}".format(new_num)
        file_name = "_".join(name_elts)
        return os.path.join(dir_name, file_name)
