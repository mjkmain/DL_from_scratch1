이전 포스팅에서 **손실함수**에 대한 내용을 다루었습니다.
#### 그럼 과연 손실함수를 왜 설정해야 할까요?
#### 손실함수의 값이 낮아지도록 학습하는 것은 무엇을 위한 것일까요?

>손실함수를 사용하는 궁극적인 목표는 **높은 정확도**를 위함입니다.

<br><br>
#### 그렇다면 "정확도"라는 지표를 두고 "손실함수"를 사용하는 이유가 무엇일까요?

> 신경망의 학습에는 **미분**이 필요합니다. 하지만 정확도는 이산적인 값으로, 미분이 불가능합니다.

10개의 데이터에 대한 정확도(Accuracy)를 생각해봅시다.
0%, 10%, 20% $\cdots$, 90%, 100%로 나누어 질 것입니다.
이를 함수로 나타내면 10칸짜리 계단처럼 생긴 계단함수일 것이고, 미분이 불가능한 부분도 있을 뿐더러 대부분의 위치에서 미분계수가 0입니다. 따라서 **"정확도"**자체를 지표로 사용하는 대신, 오차를 최소화 하는 **"손실함수"**를 지표로 사용합니다.

신경망은 **오차역전파**를 통해 학습합니다. 자세한 내용은 다음 포스팅에 다루도록 하겠습니다.

> PyTorch와 같은 딥러닝 프레임워크를 사용하면 ```loss.backward()```라는 간단한 메서드를 통해 오차역전파가 진행이 되지만, python으로 직접 오차역전파를 구현하게되면 오차역전파는 구현이 어렵기 때문에, 비교적 구현이 쉬운 수치미분을 통하여 오차역전파가 잘 구현되었는지 확인합니다.


이번 글에서는 **오차역전파**의 구현이 잘 되었는지 확인하기 위한 방법인 **수치 미분**에 대해 알아보겠습니다.

---
## 1. 미분

수치미분을 하기에 앞서, 미분의 개념에 대해 먼저 보도록 하겠습니다.
>미분 = 순간변화율 = 평균변화율의 극한

미분은 한 순간의 변화량을 의미합니다.
$$\frac{d}{dx}f(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

위의 식은 미분의 정의입니다. $h$는 "작은 변화"를 의미하는데, $h$라는 "작은 변화"가 $f(x)$를 얼마나 변화시키는지를 의미합니다.

위의 정의를 그대로 딥러닝에서 사용할 수 있으면 좋겠지만, 아쉽게도 코드에서 사용하려면 **수치 미분**으로 미분을 대체해야 합니다.

---
## 2. 수치미분

>수치미분에는 전방차분, 후방차분, 중앙차분의 3가지 방법이 있습니다.
다음의 그림을 참고하여 세 가지 수치 미분 방법에 대해 설명하겠습니다.

<p align="center"><img src="https://velog.velcdn.com/images/for_acl/post/1b3ebc8b-5221-40cf-ab83-04d869b4c7e1/image.png" height="150px" width="400px">

수치미분은 미분의 정의에 따라, 충분히 작은(Sufficiently small) $h > 0$에 대하여, $x=a$라고 하면
  
  $$f'(a)\approx \frac{f(a+h)-f(a)}{h}\eqqcolon D_{+}f(a)
  $$
  으로 근사하는 것이 일반적입니다. 위의 방법을 forward scheme이라고 하고, 문제에 따라 backward scheme, central scheme을 사용합니다.
  수치 미분을 진행할 때, 문제를 간단히 하기 위하여 $f$가 $a$ 근방에서 2번 미분가능하고 이계도함수가 연속이라고 가정합니다. 
  
### 1) 전방차분(Forward Scheme)
$$f'(a) \approx \frac{f(a+h)-f(a)}{h}\eqqcolon D_{+}f(a)
$$

### 2) 후방차분(Backward Scheme)
$$f'(a) \approx \frac{f(a)-f(a-h)}{h}\eqqcolon D_{-}f(a)
$$

### 3) 중앙차분(Central Scheme)
$$f'(a) \approx \frac{f(a+h)-f(a-h)}{2h}\eqqcolon Df(a)
$$

<br><br>
각각의 식은 위와 같습니다.
  
세 가지 방법은 문제에 따라 사용된다고 언급하였습니다. 저희는 forward, backward, central 중 어느 것을 사용해야할까요?

이에 대한 답변을 하기 위해, 우리가 **왜** 수치미분을 사용하는지 생각해보아야 합니다.
  저희는 단순히, 직접적으로 **$f'(a)$를 구하지 못하기 때문에** 수치미분을 사용하려고 하는 것입니다.
  따라서, $f'(a)$와의 오차가 가장 적은 Scheme을 선택해야 합니다. **(정답 : 중앙차분(Central scheme))**
  
---
## 3. 수치미분 - 중앙차분
  
  중앙차분을 python으로 구현하면 다음과 같습니다.
  ```python
  def numerical_diff(f, x):
  	h = 1e-4
  	return (f(x+h) - f(x-h))/(2*h)
  ```
  ---
 수치미분 예시

$$f(x) = 0.01x^2 + 0.1x$$

 위의 함수를 직접 미분해보면 
  $$f'(x) = 0.02x + 0.1$$ 입니다.
  
  $$f'(5) = 0.2, f'(10) = 0.3$$ 입니다.
  
  
  위의 함수를 파이썬으로 구현하면 다음과 같습니다.
  ```python
  def function(x):
  	return 0.01*x**2 + 0.1*x
  ```
  ---

  <p align="center"><img src="https://velog.velcdn.com/images/for_acl/post/c6738075-d021-4b84-95ef-bb2b5c461c9c/image.png" height="150px" width="400px">
    

    
```python
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0.0, 20.0, 0.1)
y = function(x)
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x, y)
plt.show()
```

    
   ---
  수치미분 함수 ```numerical_diff()```를 통해 미분값을 구해보면 다음의 결과를 얻습니다.
  
  ```python
  numerical_diff(function, 5)
  >> 0.1999999999990898
  
  numerical_diff(function, 10)
  >> 0.2999999999986347
  ```
 컴퓨터의 계산 오차와 수치미분의 오차가 있음에도, 실제 미분값과 굉장히 유사한 값을 얻을 수 있습니다. 
 
 수치미분을 통해 구한 ```x=10```에서 미분계수를 통해 접선과 함께 ```function(x)```의 그래프를 그려보면 다음과 같습니다.


<p align="center"><img src="https://velog.velcdn.com/images/for_acl/post/b0de2354-96c5-43a8-989d-c4097b56f4fe/image.png" height="150px" width="400px">
  
  

  
```python
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0.0, 20.0, 0.1)
y = function(x)
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x, y)

tangent_line_y = numerical_diff(function, 10)*(x-10)+function(10)
plt.plot(x, tangent_line_y) 
plt.show()  
  ```
접선이 잘 그려지는 것을 보니 ```numerical_diff()``` 함수가 잘 작동한다는 것을 알 수 있습니다. 
  (물론 실제 미분계수와 유사한 값을 구하기도 했습니다.)

---
  ## 4. 편미분
  
  편미분이란 다변수함수에서 한 개의 변수에 대해서만 미분하는 미분법입니다.
  미분할 변수를 제외하고, 나머지 변수를 **상수로 취급** 합니다.
  
  예를 들어, 
  $$f(x, y) = 5x^2 + 2y^3$$이라고 하면
  
  $x$에 관한 $f$의 편미분 $\frac{\partial}{\partial x}f(x, y) =10x$,
  $y$에 관한 $f$의 편미분 $\frac{\partial}{\partial y}f(x, y) =6y$ 입니다.
  
  코드로 구현할때도 나머지 변수를 **상수로 취급**한다는 개념을 이용합니다.
  
  
  
  
  
  ---
  ## Appendix. 
   수치미분 오차 증명에 관심 있으신 분들만 보시면 됩니다.
> forward scheme에 대한 오차 : $|D_+f(a)-f'(a)|$
  
- For sufficiently small $h > 0$,
  $$f'(a)\approx \frac{f(a+h)-f(a)}{h}\eqqcolon D_{+}f(a) $$

- By Taylor Series,
  $$f(x) = f(a) + f'(a)(x-a) + \frac{f''(c)}{2!}(x-a)^2 $$
  $$c\in[a, a+h]$$
- $x\leftarrow a+h$
  $$f(a+h) = f(a) + f'(a)h+\frac{f''(c)}{2!}h^2$$
  
- Divide by h on both sides
  $$\frac{f(a+h)-f(a)}{h} - f'(a) = \frac{f''(c)}{2!}h$$
  
  $$ D_+f(a)- f'(a) = \frac{f''(c)}{2!}h$$
  
  $$ \therefore Error = |\frac{f''(c)}{2!}h| $$
  
  > backward scheme에 대한 오차 : $|D_-f(a)-f'(a)|$ forward와 동일한 방식으로 진행
  
  $$ \therefore Error = |\frac{f''(c)}{2!}h|$$
  
  > central scheme에 대한 오차 : $|Df(a)-f'(a)|$
  
- For sufficiently small $h > 0$,
  $$f'(x) \approx \frac{f(a+h)-f(a-h)}{2h}\eqqcolon Df(a)$$
- By Taylor Series,
  $$f(x) = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \frac{f^{(3)}(c)}{3!}(x-a)^3$$
  
- $x\leftarrow a+h$
  $$f(x+h) = f(a)+f'(a)h + \frac{f''(a)}{2!}(x-a)^2h^2+\frac{f^{(3)}(c)}{3!}h^3 \quad\cdots (1)$$
  
- $x\leftarrow a-h$
  $$f(x-h) = f(a)-f'(a)h + \frac{f''(a)}{2!}(x-a)^2h^2-\frac{f^{(3)}(c)}{3!}h^3 \quad\cdots (2)$$
  
- (1) - (2)
  
  $$f(x+h)-f(x-h) = 2f'(a)h +2\frac{f^{(3)}(c)}{3!}h^3$$
- Divide by 2h on both sides
  $$\frac{f(x+h)-f(x-h)}{2h} - f'(a) = \frac{f^{(3)}(c)}{3!}h^3|$$
  
  $$\therefore Error = |\frac{f^{(3)}(c)}{3!}h^3|$$
  
  
  ---
-  Error of forward scheme : $|\frac{f''(c)}{2!}h|$
   
- Error of backward scheme : $|\frac{f''(c)}{2!}h|$
- Error of central scheme : $|\frac{f''(c)}{3!}h^3|$
   
  위와 같은 오차를 얻습니다. 처음에 정의에서 For sufficiently small $h > 0$ 라고 $h$를 정의했기 때문에, 오차가 가장 적은 수치미분 방법은 엄청나게 작은 수를 세제곱한 **중앙차분**입니다. 따라서 앞으로의 코드구현에서 중앙차분을 이용한 미분을 진행합니다.
  
  
