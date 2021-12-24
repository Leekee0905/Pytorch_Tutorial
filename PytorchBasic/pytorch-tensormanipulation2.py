import numpy as np 
import torch
t = np.array([[[0,1,2],
[3,4,5]],
[[6,7,8],
[9,10,11]]])

#View
ft = torch.FloatTensor(t)
print(ft)
print(ft.shape)

print(ft.view([-1,3])) # ft텐서를 (?,3)으로 크기변수
print(ft.view([-1,3]).shape)
print(ft.view([-1,1,3]))
print(ft.view([-1,1,3]).shape)

#Squeeze
ft = torch.FloatTensor([[0], [1], [2]])
print(ft)
print(ft.shape)
print(ft.squeeze())
print(ft.squeeze().shape)

#Unsqueeze
ft = torch.Tensor([0, 1, 2])
print(ft.shape)
print(ft.unsqueeze(0))
print(ft.unsqueeze(0).shape)
print('\n')
print(ft.view(1, -1))
print(ft.view(1, -1).shape)
print(ft.unsqueeze(1))
print(ft.unsqueeze(1).shape)
print(ft.unsqueeze(-1))
print(ft.unsqueeze(-1).shape)
print('\n')

#타입 캐스팅
lt = torch.LongTensor([1,2,3,4])
print(lt)
print(lt.float())
print('\n')

bt = torch.ByteTensor([True, False, False, True])
print(bt)
print(bt.long())
print(bt.float())
print('\n')

#연결하기
x = torch.FloatTensor([[1,2],[3,4]])
y = torch.FloatTensor([[5,6],[7,8]])

print(torch.cat([x,y], dim=0))
print(torch.cat([x,y], dim=1))
print('\n')

#스탴킹
x = torch.FloatTensor([1, 4])
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])

print(torch.stack([x, y, z]))
print(torch.cat([x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)], dim=0)) #윗줄과 같은 동작
print(torch.stack([x, y, z], dim=1))
print('\n')

#ones_like zeros_like - 0으로 채워진 텐서와 1로 채워진 텐서
x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])
print(x)
print(torch.ones_like(x)) # 입력 텐서와 크기를 동일하게 하면서 값을 1로 채우기
print(torch.zeros_like(x)) # 입력 텐서와 크기를 동일하게 하면서 값을 0으로 채우기
print('\n')

#In-Place Operation 덮어쓰기 연산
x = torch.FloatTensor([[1,2],[3,4]])
print(x.mul(2.)) #곱하기 2 결과
print(x) #기존값
print(x.mul_(2.)) # 곱하기 2를 수행한 결과를 변수 x에 저장하면서 결과출력
print(x) #기존값