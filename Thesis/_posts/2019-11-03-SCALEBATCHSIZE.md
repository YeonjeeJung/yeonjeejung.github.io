---
layout: post
title: Scaling SGD Batch Size to 32K for ImageNet Training
author: Yeonjee Jung
tags : Optimization
use_math : true
---

## [Abastract]

큰 네트워크의 학습 속도를 높이는 자연스러운 방법은 여러개의 GPU를 사용하는 것이다. 확률 기반 방법을 더 많은 프로세서로 확장하려면 각 GPU의 컴퓨팅 파워를 최대로 사용하기 위해 batch size를 늘려야 한다. 그러나 batch size를 계속해서 올리면서 네트워크의 정확도를 유지하는 것은 불가능하다. 현재 sota방법은 batch size에 반비례하게 LR을 늘리고, 초기 최적화의 어려움을 극복하기 위해 LR에 'warm-up'이라는 방법을 사용한다.

LR을 학습 중에 조절하므로써 ImagNet 학습에 큰 batch size를 효과적으로 사용할 수 있다. 그러나 ImageNet-1k 학습을 위해서는 현재 AlexNet이나 ResNet은 batch size를 크게 늘릴 수 없다. 이유는 큰 LR을 사용할 수 없기 때문이다. 큰 batch size를 일반적인 네트워크나 데이터에 대해 사용 가능하게 하기 위해, 이 논문에서는 Layer-wise Adaptive Rate Scaling (LARS)를 제안한다. LARS에서는 weight의 norm과 gradient의 norm을 기반으로 층마다 다른 LR을 사용한다. LARS를 사용하면 ResNet50과 AlexNet에 대해서 큰 batch size를 사용할 수 있다. 큰 batch size는 시스템의 컴퓨팅 파워를 최대로 사용할 수 있다. 이것은 속도의 향상으로 이어진다.

## [1] Introduction

깊은 네트워크의 확장과 속도 향상은 딥러닝의 응용에서 매우 중요하다. 속도 향상을 위해서는 더 많은 프로세서로 확장시켜야 한다. 그러기 위해서는 batch size를 더 크게 해야 한다.  그러나 이전 연구에 따르면 batch size를 늘이는 것은 test accuracy에서 좋지 않은 결과가 나온다. 학습 중에 LR을 조절하면 큰 batch size에서도 좋은 결과를 유지할 수 있다. 하지만 현재까지의 연구들은 1024 이상의 크기의 batch size에서는 사용할 수 없다. 이 논문의 저자들은 batch normallization을 사용해 큰 batch size에서도 좋은 성능을 얻었지만 여전히 정확도를 잃었다. _어디에서 잃었다는건지 모르겠음... 이전에 작은 batch size로 했을 때랑 결과가 똑같은데..?_ 이를 더 개선하기 위해 LARS를 제안하는데, weight의 norm $$\|w\|$$과 gradient의 norm $$\triangledown w\|$$에 따라 층마다 다른 LR을 사용한다. 만약 같은 LR을 쓴다면 $$\frac{\|w\|}{\|\triangledown w\|}$$이 큰 층은 수렴해도 작은 층은 발산할 수도 있다. LARS를 사용하면 더 큰 batch size를 사용해도 같은 정확도를 얻을 수 있다. 또한, GPU의 컴퓨팅 파워를 최대로 사용할 수 있기 때문에 속도 향상 또한 가능하다.

## [2] Background and Related Work

### [2.1] Data-Parallelism Mini-Batch SGD

보통의 mini-batch SGD는 update 식을

$$w_{t+1}=w_t-\frac{\eta}{b}\sum_{b\in B_t}\triangledown l(x, y, w)$$

라고 쓰는데, 복잡하므로 이 논문에서는

$$w_{t+1}=w_t-\eta\triangledown w_t$$

라고 쓴다.

### [2.2] Large-Batch Training Difficulty

GPU를 사용하면 여러 개의 프로세서를 동시에 돌릴 수 있지만, 이를 최대로 사용하기 위해서는 batch size가 커야 하는데, 특정 크기보다 크게 되면 테스트 정확도가 학습 정확도보다 현저하게 낮다. 이전 연구에서는 큰 batch size에 대해 학습 loss가 작아도 test loss는 그것보다 훨씬 크고, 반면 작은 batch size에 대해서는 train loss와 test loss가 비슷하다고 결론지었다. 다른 연구에서는 더 오래 학습하는 것이 일반화에 더 도움이 된다고 했는데, 반면 또다른 연구에서는 LR을 잘 조절하는 것이 정확도를 유지하는데 더 도움이 된다고 했다.

### [2.3] Learning Rate (LR)

batch size를 크게 하면 기본 LR도 그에 따라 올려야 하는데, 그러면서 정확도는 내려가지 않아야 한다. 기존 연구에서는 sqrt scaling rule과 linear scaling rule이 있었다. 또한 warmup scheme도 많이 사용되고 있는데, 처음에 작은 LR을 설정해 놓고 몇 epoch가 지나면 점점 크게 하는 방법이다. 이것이 gradual warmup scheme이고, constant warmup scheme은 초기 몇 번의 epoch에서 constant한 LR을 사용하는 방법이다. _그뒤에 어떻게 한다는 건지 안나와있네...왜 이게 warmup이지_ constant warmup은 object detection이나 segmentation에 효과적이고, gradual warmup은 ResNet-50을 학습시키는 데에 좋다. batch size에 관계없이 LR에 상한이 있다는 또다른 연구도 있는데, 이 논문의 실험 결과는 위 논문의 결과와 같다.

## [3] ImageNet-1k Training

### [3.1] Reproduce and Extend Facebook's result

이 논문에서의 첫 단계는 Facebook에서 한 연구의 결과를 따라하는 것이었는데, 그 연구에서는 multistep LR과 warmup, linear scaling LR을 사용했다. 이 연구에서도 그 연구와 비슷하게 warmup과 linear scaling을 사용했는데, Facebook의 연구와 다른 점은

1. LR을 더 높였다.  
2. multistep rule 대신 poly rule _설명이 없네_ 을 사용하였다.

### [3.2] Train ImageNet by AlexNet

#### [3.2.1] Linear Scaling and Warmup schemes for LR

이 논문에서의 baseline은 Batch-512 AlexNet이고, 100 epoch동안 0.58의 정확도를 얻는 것이다. poly rule을 사용했고, base LR은 0.01이며 poly power는 2이다. 목표는 Batch-4096, Batch-8192이고 정확도 0.58을 100 epoch 안에 얻는 것이다.

첫번째로 Batch-4096 AlexNet을 사용했고 linear scaling을 사용했으며 baseLR을 0.08로 사용했으나, Batch-4096은 0.01의 LR에도 수렴하지 않았다. 이후에는 같은 모델에 linear scaling과 warmup을 사용하였다. 하이퍼파라미터 튜닝을 거쳐 나온 가장 좋은 결과는 0.531에 그쳤다. 또한 Batch-8192의 정확도는 그것보다도 낮았다. 따라서 linear scaling과 warmup은 큰 batch size의 AlexNet을 학습시키는데 충분하지 않다는 것을 알았다.

#### [3.2.2] Batch Normalization for Large-Batch Training

여러 방법을 사용해본 결과, Batch Normalization만 성능을 개선시킨다는 것을 관찰했다. BN을 적용하고 0.01부터 결과가 발산할 때까지 LR을 늘렸는데, Batch-4096은 0.3에서 멈추었고, Batch-8192는 0.41에서 멈추었다. 그러나 이렇게 해도 Batch-512의 정확도에 도달하기 위해서는 정확도가 부족했다. momentum과 weight decay 파라미터를 조정해 보았으나 개선은 없었다.

이후에는 성능이 개선되지 않는 문제가 일반화 문제 때문이 아님을 관찰하였다. 이유는 최적화 어려움 때문이었다.

#### [3.2.3] Layer-wise Adaptive Rate Scaling (LARS) for Large-Batch Training

따라서 새로운 LR updating rule을 디자인했다. 실험 결과에서 층마다 다른 LR이 필요함을 느꼈는데, 그 이유는 $$\|w\|_ 2 $$와 $$\|\triangledown w\|_ 2$$의 비율이 층마다 매우 다르기 때문이었다. 이전에 이런 문제를 해결하기 위한 ResNet50을 위한 해결법 연구가 있었는데, AlexNet에는 효과가 없었다.

이 논문에서 제안한 LARS의 업데이트 룰은 다음과 같다.

$$\eta=l\times\gamma\times\frac{\|w\|_ 2}{\|\triangledown w\|_ 2}$$

여기서 $l$은 scaling factor ($0.001$)이고, $\gamma$는 input LR ($1$~$50$)이다. $\mu$를 momentum이라고 하고 $\beta$를 weight decay 라고 하고 LARS 알고리즘을 살펴보면

1. 각 층마다 local LR을 구한다. $$\alpha = l\times\frac{\|w\|_ 2}{\|\triangledown w\|_ 2+\beta\|\triangledown w\|_ 2}$$  
2. 각 층마다 true LR을 구한다. $$\eta = \gamma\times\alpha$$  
3. gradient를 업데이트 한다. $$\triangledown w=\triangledown w+\beta w$$  
4. 가속 항 $a$를 $$a=\mu a+\eta\triangledown w$$ 를 이용해 업데이트 한다.  
5. weight를 업데이트 한다. $w=w-a$

이다. LARS를 적용해서 baseline과 같은 결과를 얻을 수 있었다. 그 뒤에는 BN을 빼고 기본 AlexNet 모델을 사용했는데, Batch-4096에는 13 epoch, Batch-8192에는 8 epoch의 warmup range를 사용하였다. 큰 batch size를 사용해서 작은 batch size에서의 결과와 비슷한 결과를 얻으려면 BN만으로는 부족하고 LARS까지 사용해야 한다는 것을 알았다. 결과에서 보면 LARS만 사용한 결과도 둘 다를 사용한 결과보다 조금 더 낮다.

## [4] Experimental Results

### [4.2] Implementation Details

8192정도의 Batch size를 한번에 저장할 수 있는 GPU가 없었기 때문에 하나의 Batch를 32개로 나누어 순차적으로 gradient를 계산했다. GPU의 메모리가 Batch 여러 개를 넣기에 충분한 경우에는 multi solver를 넣어 속도를 향상시켰다.

### [4.3] State-of-the-art Results for ResNet50 training

이 논문의 결과는 data augmentation을 하지 않았기 때문에 sota 결과보다는 정확도가 낮다. data augmentation을 추가하면 sota 결과가 나온다. 이 논문에서는 32768의 batch size를 사용했다.

### [4.4] Benefits of Using Large Batch

큰 batch size를 사용할 수 있게 되고 baseline과 같은 결과를 얻게 되자, 이 논문에서는 속도 비교에 중점을 두었다. AlexNet-BN의 batch size가 512에서 4096으로 커지자, 4개의 GPU에서는 속도가 비슷했지만, 8개의 GPU에서는 속도가 3배 빨라졌다.

## [5] Conclusion

최적화 어려움이 큰 batch size에 대해 학습을 어렵게 만들었다. linear scaling이나 warmup같은 방법들은 AlexNet등의 복잡한 모델에는 충분하지 않다. BN같은 모델 구조를 변형하는 방법들도 사용할 수 있겠지만 충분하지 않으므로, 이 논문에서 제안하는 LARS를 사용하면 충분히 baseline 정확도를 얻을 수 있다. LARS는 층마다 weight와 weight의 gradient의 norm에 따라 다른 LR을 사용하는데, 매우 높은 효율성을 보여준다. ImageNet을 학습시키는 AlexNet에서 Batch size를 128에서 8192로 키워도 정확도 손실이 없다. 또한 ResNet50에서는 batch size를 32768까지 키울 수 있었다.

---

수학적인 증명은 별로 없는 논문이었다. 실험적인 결과로만 가설을 세우고 쓴 논문..

설명이 더 필요한 개념 : LR scheduler에서의 poly rule, constant warmup
