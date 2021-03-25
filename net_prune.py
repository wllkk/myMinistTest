import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np

train_batch_size = 64
test_batch_size = 1000
img_size = 28

transform = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.5),(0.5))
])

trainset = torchvision.datasets.MNIST(root='./data',train=True,transform=transform,download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size= train_batch_size,shuffle=True,num_workers=2)

def imshow(img):
    img = img /2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

imshow(torchvision.utils.make_grid(iter(trainloader).next()[0]))


class MyTestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=28,kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(in_channels=28,out_channels=28*2,kernel_size=5)
        self.fc1 = nn.Linear(28*2*4*4,out_features=1024)
        self.fc2 = nn.Linear(in_features=1024,out_features=10)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.pool(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.pool(x)
        x = F.relu(x)

        x = x.view(inputs.size()[0], -1)
        print(x.size())

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        #x= F.log_softmax(x, dim=-1)

        return x


def train():
    model = MyTestNet()
    momentum = 0.5
    opt = torch.optim.Adam(params=model.parameters(),lr=0.001)
    #opt = torch.optim.SGD(params=model.parameters(), lr= 0.001,momentum=momentum)
    device = torch.device("cuda:0")
    model.to(device)
    for epoch in range(10):
        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)
            
            pre_label = model(images)
            
            loss = F.cross_entropy(input=pre_label, target=labels).mean()
            #loss = F.nll_loss(pre_label,labels)
            #print(F.softmax(pre_label))
            pre_label = torch.argmax(F.softmax(pre_label), dim = 1)
            #print("pre",pre_label)
            #print("gt",labels)
            acc = (pre_label == labels).sum() / torch.tensor(labels.size()[0], dtype=torch.float32)
            model.zero_grad()
            loss.backward()
            opt.step()
            #print(model.children())
            #for m in model.modules():
                #if(isinstance(m, nn.Conv2d)):
                #m.weight.grad.data.add_(0.0001* torch.sign(m.weight.data))
        """
        for m in model.modules():
            if(isinstance(m, nn.Conv2d)):
                for name in m.state_dict():
                    print(m.state_dict()[name])
        """
        print(acc.detach().cpu().numpy(), loss.detach().cpu().numpy())

    torch.save(model.state_dict(), './model/MyNet.pth')    
    torch.save(model, './model/MyNet_All.pth')

def test():
    test_model=torch.load('./model/MyNet_All.pth')
    #test_model = MyTestNet()
    #test_model.load_state_dict(torch.load('model/MyNet.pth'))
    device = torch.device("cuda:0")
    test_model.to(device)
    #test_model.eval()
    testset = torchvision.datasets.MNIST(root='data/',train=False,transform=transform,download=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)
    #images, labels = iter(testloader).next()
    #images, labels = iter(testloader).next()
    images,labels = next(iter(testloader))
    images = images.to(device)
    labels = labels.to(device)
    print(labels)
    with torch.no_grad():
        pre_label = test_model(images)
        pre_label = torch.argmax(pre_label,dim=1)
        acc = (pre_label == labels).sum() / torch.tensor(labels.size()[0], dtype=torch.float32)
        print(acc.detach().cpu().numpy())

print("----------------train---------------")
train()
print("----------------test---------------")

test()


