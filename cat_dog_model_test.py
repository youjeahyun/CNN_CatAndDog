import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# CNN 모델 정의 (구조가 같아야 함)
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
        self.fc = torch.nn.Linear(16 * 16 * 64, 2)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# 모델 로드
model = CNN().to(device)
model.load_state_dict(torch.load('cat_dog_model.pth'))
model.eval()

# 이미지 전처리 설정
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# 임의의 이미지 로드 및 전처리
image_path = './한예슬.jpg'  # 테스트할 이미지의 경로를 입력하세요
image = Image.open(image_path)
if image.mode != 'RGB':
    image = image.convert('RGB')
image_tensor = transform(image).unsqueeze(0).to(device)  # 배치 차원 추가 및 장치로 이동

# 모델 예측
with torch.no_grad():
    output = model(image_tensor)
    _, predicted = torch.max(output.data, 1)

# 예측 결과 출력
class_names = ['cat', 'dog']
predicted_class = class_names[predicted.item()]

print(f'The image is classified as: {predicted_class}')

# 이미지 출력
plt.imshow(image)
plt.title(f'The image is classified as: {predicted_class}')
plt.axis('off')
plt.show()
