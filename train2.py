import torch, argparse, os
import net, config, loaddataset
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from collections import defaultdict
import numpy
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.svm import OneClassSVM
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, self.num_flat_features(x))
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()
        self.labels = 10  # 标签数量

        # 编码器层
        self.fc1 = nn.Linear(input_size + self.labels, 512)  # 编码器输入层
        self.fc2 = nn.Linear(512, latent_size)
        self.fc3 = nn.Linear(512, latent_size)

        # 解码器层
        self.fc4 = nn.Linear(latent_size + self.labels, 512)  # 解码器输入层
        self.fc5 = nn.Linear(512, input_size)  # 解码器输出层

    # 编码器部分
    def encode(self, x):
        x = F.relu(self.fc1(x))  # 编码器的隐藏表示
        mu = self.fc2(x)  # 潜在空间均值
        log_var = self.fc3(x)  # 潜在空间对数方差
        return mu, log_var

    # 重参数化技巧
    def reparameterize(self, mu, log_var):  # 从编码器输出的均值和对数方差中采样得到潜在变量z
        std = torch.exp(0.5 * log_var)  # 计算标准差
        eps = torch.randn_like(std)  # 从标准正态分布中采样得到随机噪声
        return mu + eps * std  # 根据重参数化公式计算潜在变量z

    # 解码器部分
    def decode(self, z):
        z = F.relu(self.fc4(z))  # 将潜在变量 z 解码为重构图像
        return torch.sigmoid(self.fc5(z))  # 将隐藏表示映射回输入图像大小，并应用 sigmoid 激活函数，以产生重构图像

    # 前向传播
    def forward(self, x, y):  # 输入图像 x，标签 y 通过编码器和解码器，得到重构图像和潜在变量的均值和对数方差
        x = torch.cat([x, y], dim=1)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        z = torch.cat([z, y], dim=1)
        return self.decode(z), mu, log_var


input_size = 784  # 输入大小
latent_size = 64  # 潜在变量大小


def generate_images_by_label(model, label, num_images=1, latent_size=64):
    model.eval()
    with torch.no_grad():
        # 创建指定标签的one-hot编码
        labels = torch.full((num_images,), label).long().to(DEVICE)
        labels_onehot = F.one_hot(labels, num_classes=10).float()

        # 生成随机噪声
        noise = torch.randn(num_images, latent_size).to(DEVICE)

        # 拼接噪声和标签
        z = torch.cat([noise, labels_onehot], dim=1)

        # 生成图像
        generated_images = model.decode(z).cpu()

        # 调整图像形状 (适用于MNIST的28x28)
        generated_images = generated_images.view(num_images, 1, 28, 28)

        return generated_images

def compute_cosine_similarity(features1, features2):
    """Compute cosine similarity between two sets of features"""
    features1 = F.normalize(features1, p=2, dim=1)
    features2 = F.normalize(features2, p=2, dim=1)
    return torch.sum(features1 * features2, dim=1).cpu().numpy()

def train(args):
    #先弄出原始图像和重构图像来
    target_network = LeNet().to(DEVICE)
    target_network.load_state_dict(torch.load('D:\\Python\\pycharm\\adv_code\\paper2\\My-lab\\paper2\\Stage1-CVAE\\save_path\\model\\LeNet_model.pth'))
    target_network.eval()

    cvae = CVAE().to(DEVICE)
    cvae.load_state_dict(torch.load('D:\\Python\\pycharm\\adv_code\\paper2\\My-lab\\paper2\\Stage1-CVAE\\save_path\\model\\CVAE_mnist.pth'))
    cvae.eval()

    train_dataset = loaddataset.PreDataset(
        root='D:\\Python\\pycharm\\adv_code\\paper2\\My-lab\\paper2\\dataset\\mnist',
        train=True,
        transform=config.test_transform,
        download=True
    )
    # 减少训练数据 每个类别选2000张 实际训练是全部的训练样本
    labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    num_samples_per_class = 6000
    selected_indices = []
    for label in label_to_indices:
        indices = label_to_indices[label]
        numpy.random.shuffle(indices)
        selected_indices.extend(indices[:num_samples_per_class])

    subset_dataset = torch.utils.data.Subset(train_dataset, selected_indices)

    train_data = torch.utils.data.DataLoader(
        subset_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=10,  # 建议增加workers提升加载速度
        pin_memory=True,
        drop_last=True
    )

    #加载编码器
    model = net.SimCLRStage2().to(DEVICE)
    model.load_state_dict(torch.load("D:\\Python\\pycharm\\adv_code\\paper2\\My-lab\\paper2\\Stage2-contrastrival_learning\\MNIST\\model\\model_stage1_epoch400.pth"), strict=False)
    model.eval()

    all_feature_cos = []
    for batch_idx, (image, label) in enumerate(train_data):
        image, label = image.to(DEVICE), label.to(DEVICE)

        with torch.no_grad():
            output = target_network(image)
            predicted_class = torch.argmax(output, dim=1)

            # 遍历批次
            generated_images_batch = []
            for i in range(image.size(0)):
                # 根据预测标签生成对应的重构图像
                label_to_generate = predicted_class[i].item()
                generated_images = generate_images_by_label(cvae, label=label_to_generate, num_images=1)
                generated_images_batch.append(generated_images)

        generated_images_batch = torch.cat(generated_images_batch, dim=0).to(DEVICE)

        feature_normal = model(image)
        feature_reconstructed = model(generated_images_batch)

        batch_cos = compute_cosine_similarity(feature_normal, feature_reconstructed)
        all_feature_cos.extend(batch_cos)

        print(f"Processed batch {batch_idx + 1}/{len(train_data)}, avg cosine sim: {numpy.mean(batch_cos):.4f}")


    features = numpy.vstack(all_feature_cos)  # 形状: (10000, 2)

# 使用孤立森林进行训练
    #iso_forest = IsolationForest(n_estimators=1000, contamination=0.1, random_state=42)
    #iso_forest.fit(features)

    from sklearn.neighbors import KernelDensity
    import matplotlib.pyplot as plt
    kde = KernelDensity(
        bandwidth=0.2,  # 核密度带宽，可优化
        kernel='gaussian'  # 常用核函数
    )

    kde.fit(features)
    train_scores = kde.score_samples(features)

    threshold = numpy.percentile(train_scores, 5)
    os.makedirs(args.save_path, exist_ok=True)

    joblib.dump(kde, os.path.join(args.save_path, 'D:\\Python\\pycharm\\adv_code\\paper2\\My-lab\\paper2\\Stage2-contrastrival_learning\\MNIST\\model\\kde_model.pkl'))
    joblib.dump(threshold, os.path.join(args.save_path, "D:\\Python\\pycharm\\adv_code\\paper2\\My-lab\\paper2\\Stage2-contrastrival_learning\\MNIST\\model\\kde_threshold.pkl"))


    # Save models
    #os.makedirs(args.save_path, exist_ok=True)
    #joblib.dump(iso_forest, os.path.join(args.save_path, 'D:\\Python\\pycharm\\adv_code\\paper2\\My-lab\\paper2\\Stage2-contrastrival_learning\\MNIST\\model\\isolation_forest.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train IsolationForest')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
    parser.add_argument('--save_path', default = 'D:\\Python\\pycharm\\adv_code\\paper2\\My-lab\\paper2\\Stage2-contrastrival_learning\\MNIST\\model', type=str, help='path to save model')
    args = parser.parse_args()
    train(args)



