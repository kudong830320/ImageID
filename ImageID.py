import requests
import zipfile
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

import torchvision

#取得GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

#準確率
def accuracy_fn(y_pred, y_true):
  correct_num = (y_pred==y_true).sum()
  acc = correct_num / len(y_true) * 100
  return acc

#訓練
def train_step(dataloader, model, cost_fn, optimizer, accuracy_fn, device):
  train_cost = 0
  train_acc = 0
  for batch, (x, y) in enumerate(dataloader):
    x = x.to(device)
    y = y.to(device)
    model.to(device)

    model.train()#訓練

    y_pred = model(x)

    cost = cost_fn(y_pred, y)

    train_cost += cost
    train_acc += accuracy_fn(y_pred.argmax(dim=1), y)

    optimizer.zero_grad()

    cost.backward()

    optimizer.step()

  train_cost /= len(train_dataloader)
  train_acc /= len(train_dataloader)

  print(f"\nTrain Cost: {train_cost:.4f}, Train Acc: {train_acc:.2f}")


def test_step(dataloader, model, cost_fn, accuracy_fn, device):
  test_cost = 0
  test_acc = 0
  model.eval()
  with torch.inference_mode():
    for x, y in dataloader:
      x = x.to(device)
      y = y.to(device)
      model.to(device)

      test_pred = model(x)

      test_cost += cost_fn(test_pred, y)
      test_acc += accuracy_fn(test_pred.argmax(dim=1), y)

    test_cost /= len(test_dataloader)
    test_acc /= len(test_dataloader)

  print(f"Test Cost: {test_cost:.4f}, Test Acc: {test_acc:.2f} \n")

#把資料包裝成Dataset
class ImageDataset(Dataset):
  def __init__(self, root, train, transform=None):

    #訓練還是測試資料
    if train:
      image_root = Path(root) / "train"
    else:
      image_root = Path(root) / "test"

    #取得類別
    with open(Path(root) / "classnames.txt", "r") as f:
      lines = f.readlines()
      self.classes = [line.strip() for line in lines] #取消\n

    #取得檔案路徑
    self.paths = [i for i in image_root.rglob("*") if i.is_file()]
    self.transform = transform

  #取得圖片標籤
  def __getitem__(self, index):
    img = Image.open(self.paths[index]).convert("RGB")
    class_name = self.paths[index].parent.name #取得回傳人物
    class_idx = self.classes.index(class_name) #取得回傳

    #判斷是否需要轉換
    if self.transform:
      return self.transform(img), class_idx
    else:
      return img, class_idx

  #取得有多少訓練資料
  def __len__(self):
    return len(self.paths)


from torchinfo import summary #模型架構

#取得pytorch別人訓練好的模型
weights = torchvision.models.EfficientNet_B1_Weights.DEFAULT
model = torchvision.models.efficientnet_b1(weights=weights)

#取得EfficientNet模型的轉換
#print(weights.transforms())

effcientnet_b1_transforms = weights.transforms()

#取得模型最後一層的資料linear的資料
#print(model.classifier[1])

#修改最後出輸的數量
from torch import nn
model.classifier[1] = nn.Linear(in_features=1280, out_features=18, bias=True)

#把梯度追蹤取消，就可以直接使用別人訓練好的模型
for param in model.features.parameters():
    param.requires_grad = False


#取得模型資料
#設定模型大小
#每一層輸入大小/輸出大小/餐數量/可否被訓練
#變數名稱
""" summary(model=model,
    input_size=(16, 3, 64, 64),
    col_names=["input_size", "output_size", "num_params", "trainable"],
    row_settings=["var_names"]) """
#訓練
train_dataset = ImageDataset(root="Image",
              train=True,
              transform=effcientnet_b1_transforms
)
#測試
test_dataset = ImageDataset(root="Image",
              train=False,
              transform=effcientnet_b1_transforms
)

from torch.utils.data import DataLoader

BATCH_SIZE = 16
#訓練資料
train_dataloader = DataLoader(dataset=train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True
)
#測試資料
test_dataloader = DataLoader(dataset=test_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False
)

#print(len(train_dataloader), len(test_dataloader))

cost_fn = nn.CrossEntropyLoss()
optimizer = optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

from tqdm.auto import tqdm

epochs = 10

#跑訓練模型
for epoch in tqdm(range(epochs)):
  print(f"Epoch: {epoch}\n-------")

  train_step(train_dataloader, model, cost_fn, optimizer, accuracy_fn, device)

  test_step(test_dataloader, model, cost_fn, accuracy_fn, device)

img = Image.open("nami.jpg").convert("RGB")
img = effcientnet_b1_transforms(img)
#print(img.shape)
img = img.reshape(-1, 3, 240, 240)
model.eval()
with torch.inference_mode():
  y_pred = model(img.to(device))

y_pred = torch.softmax(y_pred, dim=1)
class_idx =  y_pred.argmax(dim=1)
print('Answer:' + train_dataset.classes[class_idx])

img_traget = Image.open("nami.jpg")
img_traget.show()
