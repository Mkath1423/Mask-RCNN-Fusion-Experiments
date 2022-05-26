import torch

import torchvision
import torchvision.models.detection as models
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import ToTensor, transforms

from models.backbones import ResNetEarlyFusion, ResNetLateFusion, ResNetMidFusion

from dataloaders import PennFudanDataset, ApesAndShapesDataset

from benchmarking import speed_test


def get_model_instance_segmentation(num_classes,
                                    backbone : nn.Module = ResNetEarlyFusion(4, 4)):
    # load an instance segmentation model pre-trained on COCO
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128),),
        aspect_ratios=((0.5, 1.0, 2.0),))

    mask_rcnn = torchvision.models.detection.MaskRCNN(
        backbone=backbone,  # backbone
        num_classes=3,
        rpn_anchor_generator=anchor_generator,
        image_mean=[0.485, 0.456, 0.406, 0.406],
        image_std=[0.229, 0.224, 0.225, 0.225]
    )

    # get number of input features for the classifier
    in_features = mask_rcnn.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    mask_rcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = mask_rcnn.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    mask_rcnn.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                           hidden_layer,
                                                           num_classes)

    return mask_rcnn


def get_dataset_instance_segmentation():
    return ApesAndShapesDataset({
    # "data_path": "./data/ApesAndShapes",
    "class_weight": [1159680,  122625,  173226,   80469],

    # "rgb_mean": [0.23500395, 0.16341497, 0.08677972],
    # "rgb_scale": [0.05813568, 0.04724738, 0.01910823],
    # "z_mean": 0.6483100016117096,
    # "z_scale": 0.25,

    "rgb_mean": [3.5970e-01, 3.3571e-01, 3.1466e-01],
    "rgb_scale": [1.6452e-01, 1.6263e-01, 1.7999e-01],
    "z_mean": 0.6483100016117096,
    "z_scale": 0.25,

    "z_only": False,
    "percent_train": 80,
})


def display_model_prediction(model, x):
    with torch.no_grad():
        model.cuda()
        x = x.to(device="cuda:0")

        model.eval()
        image = model([x])

        for k, v in image[0].items():
            print(f"{k}: {v.size()}")

        dataset.plot_pred(x[:-1].cpu(),
                          x[-1:].cpu(),
                          image[0]["boxes"].cpu(),
                          image[0]['labels'].cpu(),
                          image[0]['scores'].cpu(),
                          image[0]["masks"].cpu())


def train_test_split(dataset, percent_train=80):
    percent_train = int(percent_train)

    n_train = percent_train*len(dataset)//100
    print(n_train)
    if n_train < 2 or n_train > (len(dataset)-2):
        return False, None, None

    dataset_train, dataset_test = torch.utils.data.random_split(dataset, [n_train,len(dataset)-n_train])
    print("train fraction: ", len(dataset_train)/len(dataset)*100)
    print("test fraction: ", len(dataset_test)/len(dataset)*100)

    return True, dataset_train, dataset_test

dataset = get_dataset_instance_segmentation()

datum = dataset[0]["x"]
print(datum.size())
print(datum.mean((1, 2)))
print(datum.std((1, 2)))

dataset.plot_sample(50)

model = get_model_instance_segmentation(4, backbone=ResNetEarlyFusion(4, 3))

# prep data
is_valid, train_data, test_data = train_test_split(dataset, percent_train=60)
print(is_valid)

display_model_prediction(model, dataset[0]['x'])

if is_valid:
    print(len(train_data), len(test_data))
    # make training objects
    train_dataloader = DataLoader(train_data, batch_size=16)
    test_dataloader  = DataLoader(test_data, batch_size=16)

    for i, batch in enumerate(train_dataloader):
        print(batch)


for i in (dataset[0].items()):
    print(f"{i[0]}: {i[1].shape}")


#
#
# for i, batch in enumerate(train_dataloader):
#     images = []
#     targets = []
#
#     print(batch['x'].shape[0])
#     for image_number in range(batch['x'].shape[0]):
#         images.append(batch['x'][image_number].squeeze())
#
#         targets.append({
#             "labels": batch["labels"][image_number],
#             "boxes": batch["boxes"][image_number],
#             "masks": batch["masks"][image_number]
#         })
#
#     model.train()
#     print(model(images, targets))

# datum = dataset[0]
# print(datum['x'].shape)
# print(datum['boxes'].shape)
# print(datum['masks'].shape)
# print(datum['labels'].shape)
#
# model.train()
#
# im = model([datum["x"]], targets=[{
#             "labels": datum["labels"],
#             "boxes": datum["boxes"],
#             "masks": datum["masks"]
# }])
#
#
#
# print(im)
# for backbone in [
#     ResNetEarlyFusion(4, 3),
#     ResNetMidFusion(3, 1, 3),
#     ResNetLateFusion(3, 1, 3)
# ]:
#     model = get_model_instance_segmentation(2, backbone=backbone)
#     model.eval()
#     speed_test(model, torch.randn(1, 4, 200, 200), iterations=500, auto_cast=False)
