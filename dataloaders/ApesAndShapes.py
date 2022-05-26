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


class ApesAndShapesDataset(Dataset):

    def __init__(self, config, data_path="data/ApesAndShapes",
                 transform=None,
                 image_transform=None,
                 depth_transform=None,
                 target_transform=None,
                 swap_color_channels=False,
                 model_height=480):

        self.data_path = os.getcwd() + "/"+ data_path

        self.file_names = os.listdir(data_path)

        print(f"found {self.file_names.__len__()} data files")

        self.transform = transform
        self.target_transform = target_transform
        self.swap_color_channels = swap_color_channels

        self.model_height = model_height
        self.swap_color_channels = swap_color_channels
        self.demo_mode = False

        self.rgb_mean = np.array(config["rgb_mean"])
        self.rgb_scale = np.array(config["rgb_scale"])

        self.z_mean = config["z_mean"]
        self.z_scale = config["z_scale"]

        # self.z_only = config["z_only"]
        self.z_only = False

        self.transform = Compose([
            Resize(self.model_height),
            CenterCrop(self.model_height),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.RandomRotation(degrees=(-90, 90)),
        ])

        if transform is not None:
            self.transform = Compose([self.transform, transform])

        self.image_transform = transforms.Normalize(self.rgb_mean, self.rgb_scale)
        self.depth_transform = transforms.Normalize(self.z_mean, self.z_scale)
        self.target_transform = None

        if image_transform is not None:
            self.image_transform = Compose([self.image_transform, image_transform])

        if depth_transform is not None:
            self.depth_transform = Compose([self.depth_transform, depth_transform])

        if target_transform is not None:
            self.target_transform = target_transform

        self.class_colors = ['black', 'r', 'g', 'b']
        self.class_color_bases = [
            torch.vstack([torch.full((1, 480, 480), 0),   torch.full((1, 480, 480), 0),   torch.full((1, 480, 480), 0)]),
            torch.vstack([torch.full((1, 480, 480), 1.0), torch.full((1, 480, 480), 0),   torch.full((1, 480, 480), 0)]),
            torch.vstack([torch.full((1, 480, 480), 0),   torch.full((1, 480, 480), 1.0), torch.full((1, 480, 480), 0)]),
            torch.vstack([torch.full((1, 480, 480), 0),   torch.full((1, 480, 480), 0),   torch.full((1, 480, 480), 1.0)]),
        ]

        self.class_names = ['background', 'cube', 'cylinder', 'monkey']


        print("Transforms Configured:")
        print("RGB mean: ", self.rgb_mean)
        print("RGB scale: ", self.rgb_scale)

        print("Z mean: ", self.z_mean)
        print("Z scale: ", self.z_scale)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        fname = self.data_path + '/' + self.file_names[idx]
        if not h5py.is_hdf5(fname): return None

        with h5py.File(fname, "r") as f:
            if not f.__contains__("colors"): return None
            colors = torch.from_numpy(np.array(f["colors"])).permute(2, 0, 1)/255
            # print(colors.shape)
            state = torch.get_rng_state()
            colors = self.transform(colors)
            colors = self.image_transform(colors)

            if not f.__contains__("depth"): return None
            depth = torch.from_numpy(np.array(f["depth"])).unsqueeze(0)
            # print(depth.shape)
            torch.set_rng_state(state)
            depth = self.transform(depth)
            depth = self.depth_transform(depth)

            if not f.__contains__("instance_segmaps"): return None
            instance_segmap = torch.from_numpy(np.array(f["instance_segmaps"])).double().unsqueeze(0)
            # print(instance_segmap.shape)
            torch.set_rng_state(state)
            instance_segmap = self.transform(instance_segmap).squeeze()

            if not f.__contains__("class_segmaps"): return None
            class_segmap = torch.from_numpy(np.array(f["class_segmaps"])).double().unsqueeze(0)
            # print(class_segmap.shape)
            torch.set_rng_state(state)
            class_segmap = self.transform(class_segmap).squeeze()

            # if self.swap_color_channels:
            #     image = torch.from_numpy(image).float().permute(2, 0, 1)

            ### transforms ###
            torch.set_rng_state(state)


            ### generate labels and boxes ###
            from torchvision.ops import masks_to_boxes


            labels = []
            boxes = []
            masks = []

            instance_ids = torch.unique(instance_segmap).double()

            for i, id in enumerate(instance_ids):
                mask = torch.where(torch.tensor(instance_segmap == id.item()), True, False)
                instance_class = class_segmap[mask].mode().values.int()

                if instance_class == 0: continue

                instance_boxes = masks_to_boxes(mask.unsqueeze(0))

                if torch.any(instance_boxes[:, 2:] <= instance_boxes[:, :2]).item():
                    continue

                boxes.append(instance_boxes)
                masks.append(mask.int().unsqueeze(0))
                labels.append(instance_class)

            boxes = torch.vstack(boxes)
            masks = torch.vstack(masks)
            labels = torch.tensor(labels, dtype=torch.long)

            rgbd = torch.cat((colors, depth), 0)
            return {"x": rgbd,
                    "boxes": boxes, "labels": labels, "masks": masks,
                    "color":colors, "depth":torch.zeros(size=(1, 480, 480)),
                    "instance_segmap":instance_segmap, "class_segmap":class_segmap}

    def plot_sample(self, idx):
        sample = self.__getitem__(idx)

        color  = sample["color"]
        depth  = sample["depth"]
        boxes  = sample["boxes"]
        labels = sample["labels"]
        masks  = sample["masks"]

        fig = plt.figure(figsize=(10, 30))

        sub = fig.add_subplot(1, 3, 1)
        plt.axis("off")
        plt.imshow(color.permute(1, 2, 0))
        sub.title.set_text("color")

        sub = fig.add_subplot(1, 3, 2)
        plt.axis("off")
        plt.imshow(depth.squeeze())
        sub.title.set_text("depth")

        sub = fig.add_subplot(1, 3, 3)
        plt.axis("off")
        plt.imshow(color.permute(1, 2, 0))
        sub.title.set_text("targets")

        for label, mask in zip(labels, masks):
            overlay = torch.vstack([self.class_color_bases[label], mask.unsqueeze(0)])
            plt.imshow(overlay.permute(1, 2, 0).detach().cpu().numpy())

        n_instances = masks.__len__()
        for i in range(n_instances):
            box = boxes[i].squeeze()
            inst_class = labels[i]

            rect = Rectangle((box[0].item(), box[1].item()), abs(box[2].item() - box[0].item()),
                             abs(box[3].item() - box[1].item()),
                                  linewidth=1,edgecolor=self.class_colors[inst_class],facecolor='none')
            plt.gca().add_patch(rect)
            plt.text(box[0].item(), box[1].item(),
                     self.class_names[inst_class],
                     fontsize=10,
                     color=self.class_colors[inst_class])

        plt.show()

    def plot_pred(self, color, depth, boxes, labels, scores, masks):
        fig = plt.figure(figsize=(10, 30))
        sub = fig.add_subplot(1, 3, 1)
        plt.axis("off")
        sub.title.set_text("color")
        plt.imshow(color.permute(1, 2, 0))

        sub = fig.add_subplot(1, 3, 2)
        plt.axis("off")
        sub.title.set_text("depth")
        plt.imshow(depth.permute(1, 2, 0).squeeze())

        sub = fig.add_subplot(1, 3, 3)
        plt.axis("off")
        sub.title.set_text("targets")
        plt.imshow(color.permute(1, 2, 0))

        for box, label, score, mask in zip(boxes, labels, scores, masks):

            overlay = torch.vstack([self.class_color_bases[label], mask])
            plt.imshow(overlay.permute(1, 2, 0).detach().cpu().numpy())

            rect = Rectangle((box[0].item(), box[1].item()), abs(box[2].item() - box[0].item()),
                             abs(box[3].item() - box[1].item()),
                             linewidth=1, edgecolor=self.class_colors[label.item()], facecolor='none')
            plt.gca().add_patch(rect)
            plt.text(box[0].item(), box[1].item(),
                     self.class_names[label.item()],
                     fontsize=5,
                     color=self.class_colors[label.item()])

            plt.text(box[0].item(), box[1].item() + 7,
                     f"{score.item():.3f}",
                     fontsize=5,
                     color=self.class_colors[label.item()])

        plt.show()