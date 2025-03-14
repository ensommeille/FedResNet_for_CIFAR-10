import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import json

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='Federated Learning with CIFAR-10')
    parser.add_argument('--num_clients', type=int, default=50, help='Number of clients')
    parser.add_argument('--num_rounds', type=int, default=20, help='Number of global communication rounds')
    parser.add_argument('--local_epochs', type=int, default=3, help='Number of local training epochs for each client')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate')
    return parser.parse_args()

# 指定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 获取命令行参数
args = parse_args()

# 超参数设置
num_clients = args.num_clients
num_rounds = args.num_rounds
local_epochs = args.local_epochs
batch_size = args.batch_size
learning_rate = args.learning_rate

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载 CIFAR-10 数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

# 将训练集数据划分给不同客户端
client_data_indices = np.array_split(np.arange(len(trainset)), num_clients)

# 定义简单的卷积神经网络模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size()[0], -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 客户端类
class Client:
    def __init__(self, client_id, data_indices):
        self.client_id = client_id
        self.data = Subset(trainset, data_indices)
        self.model = SimpleCNN().to(device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, global_state):
        # 加载服务器下发的全局模型参数作为初始状态
        self.model.load_state_dict(global_state)
        trainloader = DataLoader(self.data, batch_size=batch_size, shuffle=True)
        self.model.train()
        running_loss = 0.0
        for epoch in range(local_epochs):
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(f'Client {self.client_id}, Epoch {epoch + 1}, Loss: {running_loss / len(trainloader):.3f}')
        return self.model.state_dict()

# 服务器类
class Server:
    def __init__(self):
        self.model = SimpleCNN().to(device)

    def aggregate(self, client_weights):
        aggregated_weights = {}
        for key in client_weights[0].keys():
            aggregated_weights[key] = sum([client_weight[key] for client_weight in client_weights]) / num_clients
        self.model.load_state_dict(aggregated_weights)
        return self.model.state_dict()

    def evaluate(self):
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Accuracy on 10000 test images: {accuracy:.2f}%')
        return accuracy

# 训练主流程
def main():
    server = Server()
    clients = [Client(i, client_data_indices[i]) for i in range(num_clients)]
    accuracies = []  # 用于记录每一轮的准确率
    for round_ in range(num_rounds):
        print(f'===== Round {round_ + 1} =====')
        # 获取当前全局模型状态
        global_state = server.model.state_dict()
        client_weights = []
        for client in clients:
            # 客户端进行本地训练并返回模型权重
            client_weight = client.train(global_state)
            client_weights.append(client_weight)
        # 聚合客户端权重
        server_weights = server.aggregate(client_weights)
        # 评估全局模型性能
        accuracy = server.evaluate()
        accuracies.append(accuracy)

    # 绘制准确率随训练轮数变化的折线图
    plt.plot(range(1, num_rounds + 1), accuracies)
    plt.xlabel('Training Rounds')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy vs Training Rounds')
    plt.grid(True)

    # 构建图像保存路径和文件名
    save_folder = "./tables"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    file_name = f"client_{num_clients}_epoch_{local_epochs}_round_{num_rounds}_lr_{learning_rate}_basic.png"
    save_path = os.path.join(save_folder, file_name)
    plt.savefig(save_path)
    plt.show()

    # 保存结果为 JSON 文件
    result_folder = "./res"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    result_file_name = f"client_{num_clients}_epoch_{local_epochs}_round_{num_rounds}_lr_{learning_rate}_basic.json"
    result_file_path = os.path.join(result_folder, result_file_name)
    result = {
        "num_clients": num_clients,
        "local_epochs": local_epochs,
        "num_rounds": num_rounds,
        "learning_rate": learning_rate,
        "accuracies": accuracies
    }
    with open(result_file_path, 'w') as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    main()