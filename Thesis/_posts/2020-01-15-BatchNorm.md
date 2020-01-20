---
layout: post
title: Batch Normalization - Accelerating Deep Network Training by Reducing Internal Covariate Shift
author: Yeonjee Jung
tags : Optimization
use_math : true
---

2015년

---

## [Abstract]

딥러닝 네트워크는 각 layer의 input의 분포가 학습 도중 계속 바뀌기 때문에 어려워진다. 이를 internal covariate shift라고 하는데, 이 논문에서는 이 문제를 layer input을 정규화하므로써 해결했다. 이렇게 하면 더 큰 lr을 사용할 수 있고, 초기화에 덜 민감해진다. 또한 regularizer의 역할을 해서 Dropout등이 필요없게 된다.

## [1] Introduction

SGD는 input의 작은 변화가 모델 파라미터 전체에 끼치는 영향이 확대되기 때문에 lr같은 하이퍼파라미터 튜닝에 굉장히 민감하다. 모델의 input 분포가 학습 도중 계속 바뀌는 현상을 covariate shift라고 하는데 전형적으로 domain adaptation에 의해 해결되었다. 그런데 네트워크는 각 layer를 하나의 네트워크라고 볼 수 있기 때문에 네트워크 전체의 학습을 잘 시킬 수 있는 방법(input의 분포를 고정하는 방법)을 각 layer마다 적용할 수도 있다.

만약 네트워크에서 sigmoid함수를 활성함수로 사용한다면, input의 절댓값이 커지면 기울기는 0으로 갈 것이고, 이 말은 작은 몇몇 input값에서 말고는 업데이트가 느려진다. 이런 현상은 네트워크 전체에서 확대된다. 사실 이 문제는 ReLU를 이용하여 많이 없어지기는 했지만 여전히 존재하며, input의 분산이 학습 중에 계속해서 고정된다면 학습이 더 가속될 것이다.

각 layer의 input 분포가 바뀌는 것을 internal covariate shift라고 하는데, 이를 없애면 확실히 학습을 빨리 할 수 있다. 이 논문에서는 batch normalization이라는 방법을 제안하는데, 이 방법은 layer input의 평균과 분산을 고정시켜주는 방법이다. 이 방법을 사용하면 gradient의 파라미터 스케일이나 초기값에 대한 의존을 줄여주고, 따라서 더 큰 lr을 사용할 수 있게 해준다. 따라서 학습을 더 빠르게 만들어준다.

## [2] Towards Reducing Internal Covariate Shift

이전 연구에서 network의 input이 whitening되면 수렴이 빨라진다는 연구가 있었기 때문에 internal covariate shift를 없앰으로써 수렴속도가 빨라지는 것을 기대했다. 하지만 매 input마다  normalization만 하면 gradient descent가 진행될 때 network의 파라미터가 무시되는 현상이 나타났다. 따라서 dataset 전체의 평균과 분산을 가지고 각 input들을 normalization한다.

## [3] Normalization via Mini-Batch Statistics

normalization은 전체 layer의 표현력을 바꿀 수 있기 때문에, normalization후에 학습가능한 파라미터($\gamma, \beta$)로 다시 선형 변환을 해주는 부분을 넣는다. 또한 SGD에서 사용하기 위해서 dataset 전체의 평균과 분산이 아닌 각 minibatch의 평균과 분산을 이용해 normalization한다.

### [3.1] Training and Inference with Batch-Normalized Networks

training에서는 위와 같은 방법으로 파라미터를 훈련시키고, testing에서는 normalization과 $\gamma, \beta$를 한 과정으로 압축한다.

### [3.2] Batch-Normalized Convolutional Networks

CNN에서는 BN을 적용하는 방식이 조금 달라지는데, feature map별로 서로 다른 파라미터를 적용한다. 또한 minibatch에서 같은 feature map에 있는 input을 묶어서 normalization한다.

### [3.3] Batch Normalization enables higher learning rates

BN을 사용하면 backpropagation이 파라미터의 scale에 영향을 받지 않는다. 그리고 이 논문에서는 BN이 layer의 Jacobian의 eigenvalue들이 1에 가깝게 되도록 한다고 추측한다. 이렇게 되면 학습에 더 도움이 되지만 진짜 저렇게 되는지는 확인된 바가 없다.

### [3.4] Batch Normalization regularizes the model

BN을 사용하면 비슷한 대상에 대해 비슷한 input 분포가 나타나기 때문에 해당 input에 대해 새롭게 파라미터를 변경할 필요가 없다. 따라서 BN은 일반화에도 도움을 준다.

## [4] Experiments

### [4.1] Activations over time

이 실험에서는 MNIST를 썼는데, sota 결과를 달성하는 것이 아닌 baseline과의 비교에 중점을 두었다. BN을 쓴 모델이 test 정확도가 더 높았고, 초반부터 높은 정확도를 보여준다. 또한 각 sigmoid로 들어가는 input의 분포가 BN을 사용하기 전에는 변동이 컸으나 BN을 사용한 것은 변동이 적었다.

### [4.2] ImageNet classification

이 실험을 위해서는 변형된 Inception network를 사용하였다. 또한, BN을 그대로 적용하기보다는 성능 개선을 위해 여러 요소를 변경하였다.

1. lr을 높인다 - BN을 사용하면 더 큰 lr을 사용할 수 있으므로  
2. dropout을 없앤다 - BN을 사용하면 일반화가 더 잘되기 때문에 굳이 필요없으므로  
3. $L_2$ 정규화 비중을 줄인다 - 실험적으로 이렇게 하면 더 좋은 결과를 얻을 수 있으므로  
4. lr decay를 빠르게 한다 - 학습이 빨라지기 때문에 lr도 더 빠르게 줄어야 하므로  
5. Local Response Normalization을 없앤다 - BN이 더 좋은 normalization을 해주므로  
6. input을 더 철저히 섞는다 - 더 랜덤한 batch들을 사용해야 일반화가 더 잘되므로  
7. photometric distortion을 없앤다 - 학습이 더 빠르게 되기 때문에 데이터를 더 적게 보게 되므로  

single-Network 분류를 사용해본 결과, 같은 정확도에 이르기까지가 BN을 사용한 방법이 훨씬 적게 걸림을 알 수 있었다. 또한 lr을 5배로 높이면 이 속도는 더 빨라지는데, 30배로 높이면 좀 더 느려지는 대신 더 큰 최종 test 정확도를 얻을 수 있다. 또한 BN이 없으면 ReLU 대신 sigmoid를 사용했을 때 학습이 불가능하지만, BN을 사용하면 sigmoid로도 왠만한 정확도를 낼 수 있다. (이전 다른 모델들보다는 낮은 정확도지만)

ImageNet 경연에서 좋은 결과를 얻은 모델들은 거의다 ensemble을 사용했기 때문에 BN을 이용해서도 ensemble을 사용해봤는데, sota 결과를 갱신했다.
