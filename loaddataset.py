from torchvision.datasets import MNIST
from torchvision import transforms
from PIL import Image


class PreDataset(MNIST):
    def __getitem__(self, index):
        # 继承父类的取 target
        target = int(self.targets[index])

        # 用 MNIST 自带的数据 self.data 取出 tensor，再转 numpy，再转 PIL Image
        img = self.data[index].numpy()  # tensor → numpy
        img = Image.fromarray(img, mode='L')  # numpy → PIL (L=灰度图)

        # 再做 transform（不报错）
        if self.transform:
            img = self.transform(img)

        return img, target

