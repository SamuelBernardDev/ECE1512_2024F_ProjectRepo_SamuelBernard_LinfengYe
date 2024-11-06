import torch
from matplotlib import pyplot as plt
import os
FileName = "images_best.pt"
PT = torch.load(os.path.join("logged_files/MNIST/10/ConvNet/RANDOM/Normal/", FileName))

fig, axs = plt.subplots(10, 10)
CNT = 0
for i in range (10):
    for j in range(10):
        axs[i, j].imshow(PT[CNT][0], cmap='Greys')
        axs[i, j].axis('off')
        # axs[i, j].tight_layout()

        CNT = CNT+1
plt.tight_layout()
plt.show()
