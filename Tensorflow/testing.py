class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)           # input (3, 32, 32) output (6, 30, 30)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)      # output (6, 15, 15)

        x = self.conv2(x)           # output (16, 13, 13)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)      # output (16, 6, 6)

        x = x.view(-1, self.num_flat_features(x))   # reshape to a flat tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # This function can be used to show the shape of the output, so we don't need to count.
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = LeNet()
print(net)