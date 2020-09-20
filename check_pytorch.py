import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
from torch.utils.data import Dataset


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=4),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.fc1 = nn.Linear(384 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        # print out.size()
        out = F.dropout(F.relu(self.fc1(out)))
        out = F.dropout(F.relu(self.fc2(out)))
        out = self.fc3(out)

        return out


class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = sorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image


# make a path from argument
checkpath = os.path.join(os.getcwd(), sys.argv[1])

# check path existing
if not os.path.exists(checkpath):
    # it means that it is the absolute path
    checkpath = sys.argv[1]
    # check abs path existing
    if not os.path.exists(checkpath):
        print("Path doesn't exist")

# get all filenames of folder
namelist = os.listdir(checkpath)

use_gpu = torch.cuda.is_available()
cnn = CNN()
if use_gpu:
    cnn.cuda()

optimizer = torch.optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)

checkpoint = torch.load('model_pytorch.pth.tar')
cnn.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(227),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(227),
    transforms.ToTensor()])

my_dataset = CustomDataSet(checkpath, transform=test_transform)
check_loader = torch.utils.data.DataLoader(my_dataset, batch_size=1, shuffle=False, num_workers=0)

predictlist = []

for idx, img in enumerate(check_loader):
    images = Variable(img)
    labels = Variable(torch.Tensor([0]))

    if use_gpu:
        images = images.cuda()

    pred_labels = cnn(images)

    if pred_labels[0][0] > pred_labels[0][1]:
        predictlist.append('female')
    else:
        predictlist.append('male')

# make a final dict {filename: prediction, ...}
result = {k: v for k, v in zip(sorted(namelist), predictlist)}

# write to .json file
json_object = json.dumps(result, indent=4)
with open("process_results.json", "w") as outfile:
    outfile.write(json_object)
    
print('Done. json file is ready')