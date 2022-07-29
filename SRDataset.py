from torch.utils.data import Dataset
import torchvision.datasets
from torchvision import transforms
from torchvision.transforms import InterpolationMode

class SRDataset(Dataset):
    def __init__(self, orginal_dataset, input_size= (14,14), output_size= (28,28) ):
        self.original_dataset = orginal_dataset
        self.inputresize = transforms.Compose([transforms.Resize(input_size), transforms.Resize(output_size, InterpolationMode.BILINEAR)])
        self.outputresize = transforms.Resize(output_size, InterpolationMode.BILINEAR)


    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, index):
        temp = self.original_dataset.__getitem__(index)
        lr = self.inputresize(temp[0])
        hr = self.outputresize(temp[0])
        label = temp[1]
        return lr, hr, label


class SRMNIST(Dataset):
  def __init__(self, input_size= (14,14), output_size= (28,28) ):
        mnist_data = torchvision.datasets.MNIST(
        './MNIST', 
        train=True, download=True, transform=transforms.ToTensor())

        inputresize = transforms.Compose([transforms.Resize(input_size), transforms.Resize(output_size, InterpolationMode.BILINEAR)])
        outputresize = transforms.Resize(output_size, InterpolationMode.BILINEAR)
        self.inputs  = []
        self.targets = []
        self.labels  = []
        for i in range(len(mnist_data)):
            img1 = inputresize(mnist_data[i][0])
            img2 = outputresize(mnist_data[i][0])
            self.inputs.append(img1)
            self.targets.append(img2)
            self.labels.append(mnist_data[i][1])

  def __len__(self):
    return len(self.inputs)

  def __getitem__(self,index):
    return self.inputs[index], self.targets[index], self.labels[index]

class fastMNIST(Dataset):
    def __init__(self, input_size= (14,14), output_size= (28,28) ):
        mnist_data = torchvision.datasets.MNIST(
        './MNIST', 
        train=True, download=True, transform=transforms.ToTensor())
        
        self.datalist = list(mnist_data)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.datalist[index]
