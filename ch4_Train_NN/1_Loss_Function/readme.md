## 신경망에서 학습
학습이란 훈련 데이터로부터 가중치 매개변수의 최적값을 자동으로 얻는 것을 의미합니다.
최적값을 얻기 위하여 **손실함수**를 지표로 사용합니다.
>**손실함수(Loss function)**는 실제값(Target)과 모델의 예측값(Predict)의 차이를 수치화해주는 함수로 **"오차"**를 나타냅니다.
딥러닝의 특징은 이 **손실함수를 최소화하는 방향**으로 즉, **오차를 최소화 하는 방향으로** 학습하여 데이터를 통해 매개변수를 **자동으로 최적화** 한다는 것입니다.

손실함수 중 대표적인 것을 두 개만 꼽자면, 평균 제곱 오차(Mean Squared Error, MSE)와 교차 엔트로피 오차(Cross Entropy Error, CEE)가 있습니다.
<br><br>

---
>현재 데이터는 1개라고 생각하하고 Loss function을 알아보도록 하겠습니다.


### 1. Mean Squared Error : MSE
평균 제곱 오차의 수식은 다음과 같습니다.

$$E = \frac{1}{2}\sum_{k=1}(y_k-t_k)^2
$$



수식에서 $k$는 데이터의 차원을 나타냅니다. $y_k$는 데이터의 $k$번째 차원에 대한 **모델의 예측값(Predict)**을 나타내고, 
$t_k$는 데이터의 $k$번째 차원에 대한 **실제값(target)**을 나타냅니다.


즉, MSE는 모든 데이터의 차원에 대하여 오차 $(y_k-t_k)$의 제곱값인 $(y_k-t_k)^2$을 더합니다.
$2$으로 나누어 주는 것은, 단순히 MSE를 미분했을 때 분수를 없애고 싶어서 그런 것으로 보입니다.
<br><br>

$k$ 변수가 헷갈리실 수 있을 것 같은데, 다음과 같이 $y, t$가 주어졌을 때 $k$를 기준으로 표로 정리한 것입니다.
`y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]`
`t = [0,   0,    1,   0,   0,    0,   0,   0,   0,   0]`

|     |$k=1$|$k=2$|$k=3$|$k=4$|$k=5$|$\cdots$|$k=10$|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|$y_k$|0.1  |0.5  | 0.6 | 0.0 | 0.05| $\cdots$|0.0|
|$t_k$|0    |0    | 1 | 0|0|$\cdots$|0

<br><br>

```python
import numpy as np

def mean_squared_error(y, t):
    return np.sum((y-t)**2)/len(y)

y_correct = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
t_correct = [0,   0,    1,   0,   0,    0,   0,   0,   0,   0]
y_wrong   = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
t_wrong   = [0,   0,    1,   0,   0,    0,   0,   0,   0,   0]

print(f"MSE Correct : {mean_squared_error(np.array(y_correct), np.array(t_correct)):.5f}")
print(f"MSE Worng   : {mean_squared_error(np.array(y_wrong), np.array(t_wrong)):.5f}")
```
`>> MSE Correct : 0.01950`
`>> MSE Worng   : 0.11950`

위의 코드에서 target은 2번  index입니다. 

y_correct는 2번 index의 값이 0.6으로 가장 크고, y_wrong은 7번 index값이 0.6으로 가장 큰 값을 갖습니다.
즉, y_correct는 정답을 맞추었고, y_wrong은 맞추지 못하였습니다.

여기에서 두 데이터에 대한 타겟과의 MSE를 비교해보면 MSE Correct는 0.01950, MSE Wrong은 0.11950으로 약 6배 정도의 차이가 나는 것을 확인할 수 있습니다.

신경망 학습을 통하여 이러한 오차를 줄이고, 정답을 더욱 잘 맞추는 모델을 만드는 것이 딥러닝의 목적입니다.

---
### 2. Cross Entropy Error : CEE
교차 엔트로피 에러의 수식은 다음과 같습니다.

$$E = -\sum_{k}t_k\ln{y_k}
$$

수식을 살펴보면, $t_k = 0$일 때는 모든 항이 0이 됩니다. 즉, 정답에 해당하는 예측값에만 집중하여 오차를 계산합니다. 
수식에 $-$(minus)가 붙은 이유는 $log$함수의 그래프와, 출력 값의 범위를 생각해보면 이유를 알 수 있습니다.
![](https://velog.velcdn.com/images/for_acl/post/9131d20d-31a8-4fad-9aa7-d7e2065e057f/image.png)

$log$함수의 그래프를 보면, $(0, 1)$에서는 음수입니다. 
우선, $-$(minus)가 없다고 생각하고 target에 대한 예측 값이 0.6인 경우와, 0.9인 경우를 수식을 통해 비교해보면 다음과 같습니다.
$$E_{0.6} = 1\times\ln(0.6) \approx -0.5108
$$
$$E_{0.9} = 1\times\ln(0.9) \approx -0.1054
$$
타겟은 1이기 때문에, 분명 에러는 $E_{0.6}$에서 더 큰 값을 가져야 하는데, $E_{0.9}$값이 더 큰 것을 확인할 수 있습니다.
딥러닝에서는 에러를(Loss function을) 최소화 하는 것을 목표로 하기 때문에 Cross Entropy 수식에 $-$를 붙여서 사용합니다.

```python
import numpy as np

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t*np.log(y + delta))
    
y_correct = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
t_correct = [0,   0,    1,   0,   0,    0,   0,   0,   0,   0]
y_wrong   = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
t_wrong   = [0,   0,    1,   0,   0,    0,   0,   0,   0,   0]

print(f"CEE Correct : {cross_entropy_error(np.array(y_correct), np.array(t_correct)):.5f}")
print(f"CEE Worng   : {cross_entropy_error(np.array(y_wrong), np.array(t_wrong)):.5f}")
```
`>> CEE Correct : 0.51083`
`>> CEE Worng   : 2.30258`

위의 MSE예제에서와 동일한 코드로, loss function 부분만 Cross Entropy로 수정하였습니다.
이 또한 마찬가지로, 정답을 맞추지 못하는 CEE Wrong의 오차가 더 큰 것을 확인할 수 있습니다.

---
### 3. 미니배치 학습
> 지금까지 MSE와 CEE에서는 한 개의 데이터에 대한 손실 함수만 생각했습니다. 하지만 모델 학습을 한 개의 데이터만으로는 할 수 없습니다. Cross Entropy Error를 예시로 **$N$개의 데이터에 대한 손실 함수**를 생각해보도록 하겠습니다.

$$E = -\frac{1}{N}\sum_{n}\sum_{k}t_{nk}\ln{y_{nk}}
$$

이때 데이터가 $N$개라면 $t_{nk}$는 $n$번째 데이터의  $k$번째 값을 의미합니다. ($y_{nk}$는 신경망의 출력, $t_{nk}$는 target입니다.) 첨자가 많아져서 복잡해 보이지만, 단순히 위에서 보았던 CEE 수식을 데이터 $N$개에 대하여 확장한 것입니다.
$N$으로 나눠주는 이유는 데이터의 수에 따라 값이 변하지 않도록 평균을 구하여 정규화하는 것입니다.

훈련 데이터 중 일부만 뽑아서 학습하는 것을 **미니배치 학습**이라고 합니다. 가령 60,000장의 훈련 데이터가 있다면 이중 100장을 무작위로 Sampling하여 학습을 진행하는 것입니다.

위의 Cross Entropy Error을 미니배치 학습에 맞춰 코드를 작성하면 다음과 같습니다.

```python
def cross_entropy_error(y, t):
    delta = 1e-7

    if y.dim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]
    return -np.sum(t*np.log(y + delta))/batch_size
```


---
Error, Loss, Cost 모두 동일한 맥락으로 사용되는 용어라고 알아두시면 좋을 것 같습니다.

MSE나 CEE 이외에도 Task에 적합한 Loss function이 굉장히 많습니다. 이번 포스팅에서는 대표적인 두 가지 Loss function에 대하여 신경망의 학습 관점에서 살펴보았습니다.

이번 포스팅을 정리해보면
>신경망의 학습 지표는 Loss function이고, 신경망은 이 Loss function을 최소화하는 방향으로 파라미터를 학습한다.
 Loss function에는 대표적으로 MSE, CEE가 있다.

MSE와 CCE의 수식 정도는 기억해두시면 좋을 것 같습니다.
신경망은 Loss function을 지표로 최소화하는 방향으로 학습하는데, 이 과정에서 **미분**이 사용됩니다. 정확히는 **그래디언트(Gradient)**를 계산하며 Loss function을 최소화 합니다.
