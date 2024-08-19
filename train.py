import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
from torch.utils.data import DataLoader
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 하이퍼파라미터
learning_rate = 0.001
training_epochs = 1000  # 에폭 수를 1000으로 설정
batch_size = 32

# 데이터셋 경로 설정
data_dir = './cat_dog'

# 이미지 전처리 설정
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 이미지 크기 조정
    transforms.ToTensor(),  # 텐서로 변환
])

# 데이터셋 로드
train_dataset = dsets.ImageFolder(root=f'{data_dir}/training_set', transform=transform)
test_dataset = dsets.ImageFolder(root=f'{data_dir}/test_set', transform=transform)

# 데이터 로더 설정
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# CNN 모델 정의
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = torch.nn.Linear(16 * 16 * 64, 2)  # 고양이와 강아지 2가지 클래스

        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

model = CNN().to(device)

# 손실 함수 및 옵티마이저 설정
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 학습 단계
total_batch = len(train_loader)
print('총 배치의 수 : {}'.format(total_batch))

for epoch in range(training_epochs):
    avg_cost = 0
    for X, Y in train_loader:
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    avg_cost_value = avg_cost.item()
    print(f'[Epoch: {epoch + 1}] cost = {avg_cost_value:.9f}')

    # cost 값이 0.03 이하로 내려가면 학습 중지
    if avg_cost_value < 0.03:
        print(f'Cost가 0.03 이하로 내려갔습니다. Epoch: {epoch + 1}, cost = {avg_cost_value:.9f}')
        torch.save(model.state_dict(), 'cat_dog_model.pth')  # 모델 저장
        break

# 테스트 단계
model.eval()  # 드롭아웃, 배치 정규화 등을 평가 모드로 전환

with torch.no_grad():
    correct = 0
    total = 0
    for X_test, Y_test in test_loader:
        X_test = X_test.to(device)
        Y_test = Y_test.to(device)

        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        total += Y_test.size(0)
        correct += (predicted == Y_test).sum().item()

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
