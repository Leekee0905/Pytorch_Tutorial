import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x): #시그모이드 함수 정의
    return 1/(1+np.exp(-x))

x = np.arange(-5.0,5.0,0.1)
y1 = sigmoid(0.5*x)
y2 = sigmoid(x)
y3 = sigmoid(2*x)

#가중치에 따른 경사도 변화
plt.plot(x,y1,'r', linestyle = '--')
plt.plot(x,y2,'g')
plt.plot(x,y3,'b',linestyle = '--')
plt.plot([0,0], [1.0, 0.0], ':')
plt.title('Sigmoid Function')
plt.show()