import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import random
from torch.optim.lr_scheduler import ExponentialLR

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='Federated Learning with CIFAR-10')
    parser.add_argument('--num_clients', type=int, default=50, help='Number of clients')
    parser.add_argument('--num_rounds', type=int, default=150, help='Number of global communication rounds')
    parser.add_argument('--local_epochs', type=int, default=5, help='Number of local training epochs for each client')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')  # 初始学习率稍大
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for SGD optimizer')
    parser.add_argument('--fedprox_mu', type=float, default=0.0001, help='FedProx proximal term coefficient')
    parser.add_argument('--dp_noise_scale', type=float, default=0.001, help='Differential privacy noise scale')
    parser.add_argument('--compression_threshold', type=float, default=0.001, help='Model update compression threshold')
    parser.add_argument('--num_selected_clients', type=int, default=10, help='Number of selected clients per round')
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'], help='Optimizer type (SGD or Adam)')
    parser.add_argument('--lr_gamma', type=float, default=0.9, help='Learning rate decay factor for exponential decay')
    parser.add_argument('--sample_strategy', type=str, default='mixed', choices=['random', 'performance', 'mixed'], help='Sampling strategy: random, performance, or mixed')
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
momentum = args.momentum
weight_decay = args.weight_decay
fedprox_mu = args.fedprox_mu
dp_noise_scale = args.dp_noise_scale
compression_threshold = args.compression_threshold
num_selected_clients = args.num_selected_clients
optimizer_type = args.optimizer
lr_gamma = args.lr_gamma
sample_strategy = args.sample_strategy

# 数据预处理：训练集数据增强 + 归一化
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# 测试集只做归一化
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载 CIFAR-10 数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# 将训练集数据划分给不同客户端
client_data_indices = np.array_split(np.arange(len(trainset)), num_clients)

# 定义 SqueezeNet 模型（修改全连接层适配 CIFAR-10）
class SqueezeNetFed(nn.Module):
    def __init__(self, pretrained=True):
        super(SqueezeNetFed, self).__init__()
        self.model = models.squeezenet1_1(pretrained=pretrained)
        # 修改输出层为 10 类
        self.model.classifier[1] = nn.Conv2d(512, 10, kernel_size=(1, 1), stride=(1, 1))
        self.model.num_classes = 10

    def forward(self, x):
        return self.model(x)

# 客户端定义
class Client:
    def __init__(self, client_id, data_indices):
        self.client_id = client_id
        self.data = Subset(trainset, data_indices)
        self.model = SqueezeNetFed(pretrained=False).to(device)
        # 根据用户指定的优化器类型创建优化器
        if optimizer_type == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_type == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.last_loss = float('inf')
        # 创建学习率调度器（指数衰减）
        self.scheduler = ExponentialLR(self.optimizer, gamma=lr_gamma)

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
                # 加入 FedProx 近端项，限制本地模型偏离全局模型过远
                proximal_term = 0.0
                for name, param in self.model.named_parameters():
                    global_param = global_state[name].to(device)
                    proximal_term += ((param - global_param) ** 2).sum()
                loss += (fedprox_mu / 2) * proximal_term
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

                # 释放不需要的中间变量
                del outputs, proximal_term
                torch.cuda.empty_cache()

            print(f'Client {self.client_id}, Epoch {epoch + 1}, Loss: {running_loss / len(trainloader):.3f}')
        self.last_loss = running_loss / len(trainloader)
        # 更新学习率
        self.scheduler.step()
        # 训练结束后计算模型更新（delta）
        local_state = self.model.state_dict()
        delta = {}
        for key in global_state.keys():
            update = local_state[key] - global_state[key].to(device)
            if torch.is_floating_point(update):
                # 压缩更新：将小于阈值的更新置零
                mask = update.abs() < compression_threshold
                update[mask] = 0.0
                # 添加差分隐私噪声
                noise = torch.randn_like(update) * dp_noise_scale
                delta[key] = update + noise
            else:
                # 对于非浮点参数（例如 BatchNorm 的 num_batches_tracked），直接传输
                delta[key] = update
        return delta


# 服务器定义
class Server:
    def __init__(self):
        self.model = SqueezeNetFed(pretrained=False).to(device)

    def aggregate(self, client_deltas):
        """
        使用安全聚合技术对客户端更新进行均值聚合
        注意：此处仅为模拟安全聚合，实际应用中应采用安全多方计算协议保障隐私
        """
        global_state = self.model.state_dict()
        aggregated_delta = {}
        for key in global_state.keys():
            aggregated_delta[key] = sum(delta[key] for delta in client_deltas) / len(client_deltas)
        # 更新全局模型参数：全局模型 = 旧全局模型 + 聚合更新
        new_global_state = {}
        for key in global_state.keys():
            new_global_state[key] = global_state[key] + aggregated_delta[key]
        self.model.load_state_dict(new_global_state)
        return new_global_state

    def evaluate(self):
        """测试服务器端模型"""
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
        self.model.eval()
        correct, total = 0, 0
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

# 随机采样
def sample_clients_randomly(clients, num_selected_clients):
    return random.sample(clients, num_selected_clients)

# 基于客户端历史性能的采样
def sample_clients_by_performance(clients, num_selected_clients):
    losses = [client.last_loss for client in clients]
    sorted_indices = np.argsort(losses)
    selected_indices = sorted_indices[:num_selected_clients]
    return [clients[i] for i in selected_indices]

# 训练主流程
def main():
    server = Server()
    clients = [Client(i, client_data_indices[i]) for i in range(num_clients)]
    accuracies = []  # 用于记录每一轮的准确率
    for round_ in range(num_rounds):
        print(f'===== Round {round_ + 1} =====')
        # 获取当前全局模型状态
        global_state = server.model.state_dict()
        # 选择采样方法
        if sample_strategy == 'random':
            selected_clients = sample_clients_randomly(clients, num_selected_clients)
            print("Sampling randomly")
        elif sample_strategy == 'performance':
            selected_clients = sample_clients_by_performance(clients, num_selected_clients)
            print("Sampling by performance")
        elif sample_strategy == 'mixed':
            if round_ % 2 == 0:
                selected_clients = sample_clients_randomly(clients, num_selected_clients)
                print("Sampling randomly")
            else:
                selected_clients = sample_clients_by_performance(clients, num_selected_clients)
                print("Sampling by performance")
        client_deltas = []
        for client in selected_clients:
            # 客户端进行本地训练并返回模型更新（差分隐私 + 压缩 + FedProx）
            delta = client.train(global_state)
            client_deltas.append(delta)
        # 聚合客户端更新（模拟安全聚合）
        server.aggregate(client_deltas)
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
    file_name = f"client_{num_clients}_rounds_{num_rounds}_epochs_{local_epochs}_optimizer_{optimizer_type}_strategy_{sample_strategy}.png"
    save_path = os.path.join(save_folder, file_name)
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    main()