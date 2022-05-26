import os
import h5py
import torch
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
# from torchvision import datasets
from torchvision.io import read_image
from torchvision.transforms import ToTensor, Lambda, Compose, CenterCrop, Resize
from torchvision.transforms.functional import resize, crop, center_crop
import numpy as np
from matplotlib import pyplot as plt

# class HDF5Dataset(Dataset):
#
#     def __init__(self, config, data_path="./out",
#                  transform=None,
#                  image_transform=None,
#                  depth_transform=None,
#                  target_transform=None,
#                  swap_color_channels=False,
#                  model_height=480):
#
#         self.data_path = os.path.expanduser(data_path)
#
#         self.file_names = os.listdir(self.data_path)
#         print(self.file_names)
#
#         self.transform = transform
#         self.target_transform = target_transform
#         self.swap_color_channels = swap_color_channels
#
#         self.model_height = model_height
#         self.swap_color_channels = swap_color_channels
#         self.demo_mode = False
#
#         self.rgb_mean = np.array(config["rgb_mean"])
#         self.rgb_scale = np.array(config["rgb_scale"])
#
#         self.z_mean = config["z_mean"]
#         self.z_scale = config["z_scale"]
#
#         # self.z_only = config["z_only"]
#         self.z_only = False
#
#         self.transform = Compose([
#             Resize(self.model_height),
#             CenterCrop(self.model_height),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomVerticalFlip(),
#             transforms.RandomRotation(degrees=(-90, 90)),
#         ])
#
#         if transform is not None:
#             self.transform = Compose([self.transform, transform])
#
#         self.image_transform = transforms.Normalize(self.rgb_mean, self.rgb_scale)
#         self.depth_transform = transforms.Normalize(self.z_mean, self.z_scale)
#         self.target_transform = None
#
#         if image_transform is not None:
#             self.image_transform = Compose([self.image_transform, image_transform])
#
#         if depth_transform is not None:
#             self.depth_transform = Compose([self.depth_transform, depth_transform])
#
#         if target_transform is not None:
#             self.target_transform = target_transform
#
#         self.class_colors = ['black', 'r', 'b', 'g']
#         self.class_names = ['background', 'cube', 'cylinder', 'monkey']
#
#         print("Transforms Configured:")
#         print("RGB mean: ", self.rgb_mean)
#         print("RGB scale: ", self.rgb_scale)
#
#         print("Z mean: ", self.z_mean)
#         print("Z scale: ", self.z_scale)
#
#     def __len__(self):
#         return len(self.file_names)
#
#     def __getitem__(self, idx):
#         fname = self.data_path + '/' + self.file_names[idx]
#
#         print(fname)
#         if not h5py.is_hdf5(fname): return None
#
#         with h5py.File(fname, "r") as f:
#             print(f.keys())
#             if not f.__contains__("colors"): return None
#             colors = torch.from_numpy(np.array(f["colors"])).permute(2, 0, 1)/255
#
#             if not f.__contains__("depth"): return None
#             depth = torch.from_numpy(np.array(f["depth"])).unsqueeze(0)
#
#             if not f.__contains__("instance_segmaps"): return None
#             instance_segmap = torch.from_numpy(np.array(f["instance_segmaps"])).double()
#
#             if not f.__contains__("class_segmaps"): return None
#             class_segmap = torch.from_numpy(np.array(f["class_segmaps"])).double()
#
#             # print(colors.size())
#             # print(depth.size())
#             # print(instance_segmap.size())
#             # print(class_segmap.size())
#
#             # if self.swap_color_channels:
#             #     image = torch.from_numpy(image).float().permute(2, 0, 1)
#
#             ### generate labels and boxes ###
#             from torchvision.ops import masks_to_boxes
#
#             n_instances = instance_segmap.unique().__len__() - 1
#
#             width = int(colors.size()[0])
#             height = int(colors.size()[1])
#
#             labels = []
#             boxes = []
#             masks = []
#
#             instance_ids = torch.unique(instance_segmap).double()
#
#             for i, id in enumerate(instance_ids):
#                 mask = torch.where(torch.tensor(instance_segmap == id.item()), True, False)
#                 instance_class = class_segmap[mask].mode().values.int()
#
#                 if instance_class == 0: continue
#
#                 instance_boxes = masks_to_boxes(mask.unsqueeze(0))
#
#                 boxes.append(instance_boxes)
#                 masks.append(mask.int().unsqueeze(0))
#                 labels.append(instance_class)
#
#             boxes = torch.stack(boxes)
#             masks = torch.stack(masks)
#             labels = torch.tensor(labels)
#
#
#             ### transforms ###
#             # state = torch.get_rng_state()
#             # colors = self.transform(colors)
#             # viz_image = colors.clone()
#             #
#             # torch.set_rng_state(state)
#             # depth = self.transform(depth)
#             # viz_depth = depth.clone()
#             #
#             # torch.set_rng_state(state)
#             # label = self.transform(label)
#             #
#             # image = self.image_transform(image)
#             #
#             # torch.set_rng_state(state)
#             # depth = self.depth_transform(depth)
#             #
#             # if self.target_transform is not None:
#             #     torch.set_rng_state(state)
#             #     label = self.target_transform(label)
#             ### transforms ###
#
#             rgbd = torch.cat((colors, depth), 0)
#             return {"x": rgbd, "boxes": boxes, "labels": labels, "masks": masks, "color":colors, "depth":depth}
#
#     def plot_sample(self, idx):
#         sample = self.__getitem__(idx)
#
#         color  = sample["color"]
#         depth  = sample["depth"]
#         boxes  = sample["boxes"]
#         labels = sample["labels"]
#         masks  = sample["masks"]
#
#         fig = plt.figure(figsize=(10, 30))
#
#         sub = fig.add_subplot(1, 3, 1)
#         plt.axis("off")
#         plt.imshow(color.permute(1, 2, 0))
#         sub.title.set_text("color")
#
#         sub = fig.add_subplot(1, 3, 2)
#         plt.axis("off")
#         plt.imshow(depth.squeeze())
#         sub.title.set_text("depth")
#
#         sub = fig.add_subplot(1, 3, 3)
#         plt.axis("off")
#         plt.imshow(color.permute(1, 2, 0))
#         sub.title.set_text("masks")
#
#         n_instances = masks.__len__()
#         print(n_instances)
#         for i in range(n_instances):
#             box = boxes[i].squeeze()
#             inst_class = labels[i]
#
#             rect = Rectangle((box[0].item(), box[1].item()), abs(box[2].item() - box[0].item()),
#                              abs(box[3].item() - box[1].item()),
#                                   linewidth=1,edgecolor=self.class_colors[inst_class],facecolor='none')
#             plt.gca().add_patch(rect)
#             plt.text(box[0].item(), box[1].item(),
#                      self.class_names[inst_class],
#                      fontsize=10,
#                      color=self.class_colors[inst_class])
#
#         plt.show()
#
#     def plot_triplet(self, m, z, l, num_classes=3):
#         figure = plt.figure(figsize=(10, 30))
#         figure.add_subplot(1, 3, 1)
#         plt.axis("off")
#         plt.imshow(m.squeeze().permute(1, 2, 0))
#
#         figure.add_subplot(1, 3, 2)
#         plt.axis("off")
#         plt.imshow(z.squeeze(), vmin=-1, vmax=1)
#
#         figure.add_subplot(1, 3, 3)
#         plt.axis("off")
#         plt.imshow(l.squeeze(), vmin=0, vmax=(num_classes-1))
#
#         plt.show()
#
# if __name__ == "__main__":
#     dataset = HDF5Dataset({
#             "name": "dataset_us_d435_480_20210826B",
#             "directory": "/content/drive/Shareddrives/IRIS/Data/Projects/Pepper",
#             "class_weight": [1.0, 1.0, 10.0],
#             "rgb_mean": [0.23500395, 0.16341497, 0.08677972],
#             "rgb_scale": [0.05813568, 0.04724738, 0.01910823],
#             "z_mean": 0.6483100016117096,
#             "z_scale": 0.25,
#             "z_only": False,
#             "val_split": 80,
#         })
#     dataset.plot_sample(0)

with h5py.File("./data/ApesAndShapes/0.hdf5", 'r') as f:
    print(f.keys())

    colors = torch.from_numpy(np.array(f["colors"])).double()
    instances = torch.from_numpy(np.array(f["instance_segmaps"])).double()
    classes = torch.from_numpy(np.array(f["class_segmaps"])).double()

    print(torch.unique(classes))
    fig = plt.figure(figsize=(10, 30))

    sub = fig.add_subplot(1, 3, 1)
    plt.axis("off")
    plt.imshow(classes)
    plt.show()
    # from torchvision.ops import masks_to_boxes
    #
    # classs_colors = ['orange', 'r', 'b', 'g']
    # class_names = ['background', 'cube', 'cylinder', 'monkey']
    # background_class = 0
    #
    # n_instances = instances.unique().__len__() - 1
    #
    # width = int(colors.size()[0])
    # height = int(colors.size()[1])
    #
    # labels = []
    # boxes = []
    # masks = []
    #
    # fig = plt.figure(figsize=(10, 30))
    #
    # # # show classes separated
    # # class_types = torch.unique(classes).double()
    # # num_classes = class_types.__len__()
    # #
    # # for i, clas in enumerate(class_types):
    # #     sub = fig.add_subplot(2, 2, i + 1)
    # #     plt.axis("off")
    # #
    # #     sub.title.set_text(i)
    # #     plt.imshow(torch.where(classes == i, 1., 0.))
    # #
    # # plt.show()
    #
    # # show inst separated
    # instance_ids = torch.unique(instances).double()
    #
    # for i, id in enumerate(instance_ids):
    #     mask = torch.where(torch.tensor(instances == id.item()), True, False)
    #     instance_class = classes[mask].mode().values.int()
    #
    #     if instance_class == background_class: continue
    #
    #     instance_boxes = masks_to_boxes(mask.unsqueeze(0))
    #
    #     boxes.append(instance_boxes)
    #     masks.append(mask.int().unsqueeze(0))
    #     labels.append(instance_class)
    #
    # print(torch.stack(boxes))
    # print(torch.stack(masks))
    # print(torch.tensor(labels))
    #
    # sub = fig.add_subplot(1, 1, 1)
    # plt.axis("off")
    # plt.imshow(colors/255)
    #
    # for i in range(n_instances):
    #     box = boxes[i].squeeze()
    #     inst_class = labels[i]
    #
    #     rect = Rectangle((box[0].item(), box[1].item()), abs(box[2].item() - box[0].item()),
    #                      abs(box[3].item() - box[1].item()),
    #                           linewidth=1,edgecolor=classs_colors[inst_class],facecolor='none')
    #     plt.gca().add_patch(rect)
    #     plt.text(box[0].item(), box[1].item(), class_names[inst_class], fontsize=10, color=classs_colors[inst_class])
    #
    # plt.show()






