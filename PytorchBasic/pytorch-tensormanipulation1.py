import numpy as np
import torch
t = np.array([0., 1., 2. ,3., 4., 5., 6.])
print(t)

print('Rank of t: ',t.ndim)
print('Shape of t: ',t.shape)

print('t[0] t[1] t[-1] = ', t[0], t[1], t[-1]) #인덱스를 통한 원소 접근
print('t[2:5] t[4:1] = ', t[2:5], t[4:-1]) #[시작번호 : 끝번호]로 범위 지정을 통해 가져옴


#1D with PyTorch
t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])
print(t)
print(t.dim()) #dim = 차원
print(t.shape) #shape
print(t.size()) #shape

#2D with PyTorch
t= torch.FloatTensor([[1., 2., 3.],
[4., 5., 6.],
[7., 8., 9.],
[10., 11., 12.]])
print(t)
print(t.dim())
print(t.size())

#슬라이싱
print(t[:,1]) #첫 번째 차원을 전체 선택한 상황에서 두 번째 차원의 두 번째 것만 가져오기
print(t[:,1].size())
print(t[:,:-1]) #첫 번째 차원을 전체 선택한 상황에서 두 번째 차원의 맨 마지막에서 첫 번째 제외하고 다 가져오기

#브로드캐스팅
#크기가 다른 행렬 또는 텐서에 대해 사칙 연산을 수행할 필요가 있는데 이를 자동으로 크기를 맞춰서 연산을 수행하게 만드는 것

m1 = torch.FloatTensor([3, 3])
m2 = torch.FloatTensor([2, 2])
print(m1+m2)

m1 = torch.FloatTensor([1, 2])
m2 = torch.FloatTensor([3]) #[3]->[3,3]
print(m1+m2)

m1 = torch.FloatTensor([1, 2])
m2 = torch.FloatTensor([[3], [4]])
print(m1+m2)

#곱셈
m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print('Shape of Matrix 1: ', m1.shape) # 2 x 2
print('Shape of Matrix 2: ', m2.shape) # 2 x 1
print(m1.matmul(m2)) # 2 x 1

m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print('Shape of Matrix 1: ', m1.shape) # 2 x 2
print('Shape of Matrix 2: ', m2.shape) # 2 x 1
print(m1 * m2) # 2 x 2
print(m1.mul(m2))

#평균
t = torch.FloatTensor([1, 2])
print(t.mean())

t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)
print(t.mean())
print(t.mean(dim=0))
print(t.mean(dim=1))
print(t.mean(dim=-1))

#덧셈
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)
print(t.sum()) # 단순히 원소 전체의 덧셈을 수행
print(t.sum(dim=0)) # 행을 제거
print(t.sum(dim=1)) # 열을 제거
print(t.sum(dim=-1)) # 열을 제거

#최대와 아그맥스(ArgMax)
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)

print(t.max())
print(t.max(dim=0))
print('Max: ', t.max(dim=0)[0])
print('Argmax: ', t.max(dim=0)[1])
print(t.max(dim=1))
print(t.max(dim=-1))