from torchvision.datasets import MNIST
from torchvision import transforms

class MnistDataset(MNIST):
    def __init__(self, root, train=True):
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
        super(MnistDataset, self).__init__(root, train=train, download=True, transform=transform)
