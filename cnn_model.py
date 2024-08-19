import torch
import torch.nn as nn

inputs = torch.Tensor(1, 1, 28, 28)
#print('텐서의 크기 : {}'.format(inputs.shape))
#print(inputs)

conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
#print(conv1)

conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,stride=1,  padding=1)
#print(conv2)

pool = nn.MaxPool2d(2)
#print(pool)


#Convolution 연산 후의 출력 크기는 (입력 크기 - 커널 크기 + 2 × 패딩) / 스트라이드 + 1로 계산
out = conv1(inputs)
#print(out.shape)

#pooling 연산 후의 출력 크기는 (입력 크기 - 풀링 커널 크기 + 2 × 패딩) / 스트라이드 + 1로 계산
out = pool(out)
#print(out.shape)

out = conv2(out)
#print(out.shape)

out = pool(out)
#print(out.shape)
#print(out.size(3))
# 첫번째 차원인 배치 차원은 그대로 두고 나머지는 펼쳐라
out = out.view(out.size(0), -1)
print(out.shape)

fc = nn.Linear(3136, 10) # input_dim = 3,136, output_dim = 10
out = fc(out)
print(out.shape)