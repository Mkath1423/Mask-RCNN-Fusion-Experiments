import matplotlib.pyplot as plt
import numpy as np
import torch

from dataloaders import ApesAndShapesDataset


def func3(x, y):
    return (1 - x / 2 + x**5 + y**3) * np.exp(-(x**2 + y**2))


# make these smaller to increase the resolution
dx, dy = 0.0125, 0.0125

x = np.arange(-3.0, 3.0, dx)
y = np.arange(-3.0, 3.0, dy)
X, Y = np.meshgrid(x, y)

# when layering multiple images, the images need to have the same
# extent.  This does not mean they need to have the same shape, but
# they both need to render to the same coordinate system determined by
# xmin, xmax, ymin, ymax.  Note if you use different interpolations
# for the images their apparent extent could be different due to
# interpolation edge effects

colors = [
    torch.vstack([torch.full((1, 480, 480), 0), torch.full((1, 480, 480), 0), torch.full((1, 480, 480), 0)]),
    torch.vstack([torch.full((1, 480, 480), 1.0), torch.full((1, 480, 480), 0), torch.full((1, 480, 480), 0)]),
    torch.vstack([torch.full((1, 480, 480), 0), torch.full((1, 480, 480), 1.0), torch.full((1, 480, 480), 0)]),
    torch.vstack([torch.full((1, 480, 480), 0), torch.full((1, 480, 480), 0), torch.full((1, 480, 480), 1.0)]),
]

fig = plt.figure(frameon=False)
dataset = ApesAndShapesDataset({
            "rgb_mean": [0.23500395, 0.16341497, 0.08677972],
            "rgb_scale": [0.05813568, 0.04724738, 0.01910823],
            "z_mean": 0.6483100016117096,
            "z_scale": 0.25
        })
Z1 = np.add.outer(range(8), range(8)) % 2  # chessboard
im1 = plt.imshow(Z1, cmap=plt.cm.gray, interpolation='nearest')

Z2 = torch.from_numpy(func3(X, Y)).unsqueeze(0)
print(Z2.size())

datum = dataset[0]
overlay = torch.vstack([datum["x"][:-1], datum["masks"][0].unsqueeze(0)])

print(datum["x"][:-1])

for i in datum.items():
    print(i[0], i[1].size())

# plt.imshow(overlay.permute(1, 2, 0))

for label, mask in zip(datum["labels"], datum["masks"]):
    overlay = torch.vstack([colors[label], mask.unsqueeze(0)])
    print(overlay)
    plt.imshow(overlay.permute(1, 2, 0))

plt.show()