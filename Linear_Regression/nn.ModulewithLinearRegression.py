import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

#y=2x가정

x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])

#모델 선언 및 초기화. 단순 선형 회귀이므로 input_dim = 1, output_dim = 1
model = nn.Linear(1,1)

print(list(model.parameters())) #첫 번째 값 가중치w, 두 번째 값 편향치b
optimizer = torch.optim.SGD(model.parameters(),lr =0.01)
nb_epochs = 2000
for epoch in range(nb_epochs + 1):
    #H(x)
    prediction = model(x_train)

    #cost
    cost = F.mse_loss(prediction, y_train) # 파이토치에서 제공하는 MSE(평균제곱오차함수)

    #cost로 H(x)개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))

new_var = torch.FloatTensor([4.0])
pred_y = model(new_var)
print('훈련 끝난 후 입력이 4 일때의 예측값:',pred_y)


#다중 선형 회귀

x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

model = nn.Linear(3,1)
print(list(model.parameters()))#처음 3개 가중치 뒤에 하나 편향치

optimizer = torch.optim.SGD(model.parameters(), lr=1e-5) # 가중치가 0.01이면 기울기가 발산함. 0.00001로 lr설정 = 1e-5
nb_epochs = 2000
for epoch in range(nb_epochs+1):
    prediction = model(x_train)

    cost = F.mse_loss(prediction, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
    # 100번마다 로그 출력
      print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, nb_epochs, cost.item()
      ))

new_var = torch.FloatTensor([[73,80,75]])
pred_y = model(new_var)
print("훈련 후 입력이 73, 80, 75일 때의 예측값:",pred_y)