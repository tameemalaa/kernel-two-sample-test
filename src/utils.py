import numpy as np
from torchvision import datasets, transforms

def load_mnist_digits(digit, n_samples=100):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    mnist = datasets.MNIST(root='data/mnist', train=True, download=True, transform=transform)
    idx = (mnist.targets == digit).nonzero().squeeze()
    selected_idx = np.random.choice(idx, size=n_samples, replace=False)
    images = mnist.data[selected_idx].float().view(n_samples, -1).numpy() / 255.0
    return images

def load_all_mnist_digits(n_samples=100):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    mnist = datasets.MNIST(root='data/mnist', train=True, download=True, transform=transform)
    images = mnist.data.float().view(len(mnist), -1).numpy() / 255.0
    return images[:n_samples]


def median_heuristic(X, Y):
    Z = np.vstack([X, Y])
    dists = np.sqrt(np.sum((Z[:, None, :] - Z[None, :, :])**2, axis=2))
    return np.median(dists)