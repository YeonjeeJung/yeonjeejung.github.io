---
layout: post
title: On the Variance of the Adaptive Learning Rate and Beyond
author: Yeonjee Jung
tags : Optimization
use_math : true
---

## [Abstract]

learning rate warmup은 학습을 안정화시키고, 수렴을 가속화하고 adaptive SGD의 일반화를 개선하는 데에 좋다. 이 논문에서는 그 매커니즘을 자세히 알아본다. adaptive lr은 초기 단계에 분산이 큰데, warmup이 분산 감소에 효과적이라는 이론을 제안하고 이를 검증한다. 또한 이 논문에서는 RAdam을 제시한다.

## [1] Introduction

최근 adaptive lr이 빠른 수렴도때문에 많이 쓰이는데, 나쁜 local optima에 빠지지 않게 하고 학습을 안정화시키기 위해 warmup을 사용한다. 하지만 이 warmup에 대한 이론적인 토대는 충분하지 않기 때문에, 이를 스케줄링하는데 많은 노력이 들어간다.

이 논문에서는 warmup을 이론적, 실험적으로 해석한다. 정확하게는, adaptive learning이 제한된 training sample로 학습되기 때문에 초기에 큰 분산을 가지게 된다는 것을 보여주려 한다. 따라서 초기에 작은 lr을 이용해 warmup을 하면 더 작은 분산을 갖게되며 학습이 잘 된다는 것을 보여준다. 더해서, Adam의 변형인 RAdam을 제안하는데, 이는 Adam에서 adaptive lr의 분산 항을 수정한 것이다.

## [2] Preliminaries and Motivations

### Generic adaptive methods

모든 알고리즘은 $\phi$와 $\psi$의 선택에 따라 달라진다.

### Learning rate warmup

warmup이 왜 잘되는지 알기 위해 gradient의 절댓값을 log scale로 그려보았는데, warmup이 없을 때에는 큰 gradient가 많았지만 있는 경우에는 작은 gradient가 많았다. 이 말은 처음 몇 단계에서는 나쁜 local optima에 빠진다는 것을 의미한다.

## [3] Variance of Adaptive Rate

"학습 초기에는 샘플이 별로 없어서 adaptive learning rate의 분산이 커져서 나쁜 local optima로 향한다" 가 이논문이 주장하는 가설이다.

### [3.1] Warmup as Variance Reduction

Adam-2k와 Adam-eps라는 Adam의 변형들이 실험에 사용되었는데, 이 논문에서는 실험을 위해 IWSLT'14 German to English 데이터셋을 사용했다. Adam-2k에서는 초기 2k 반복동안에 adaptive learning rate $\psi$만 업데이트되고 momentum $\phi$와 파라미터 $\theta$는 고정시켰는데, 이 방법은 vanilla Adam의 수렴 문제를 해결했고 초기 단계에서의 샘플의 부족이 gradient의 분포를 왜곡시킨다는 점을 알 수 있다. *이게 왜 샘플의 개수 때문이지..?*

또한, 이 논문에서는 adaptive learning rate의 분산을 작게 만듦으로써 이 수렴 문제를 해결할 수 있다는 점도 증명했다. Adam-eps에서는 adaptive lr의 분산을 작게 하기 위해 $\hat{\psi}(.)$에서의 $\epsilon$을 크게 설정했다. 이것은 또한 vanilla Adam의 수렴 문제를 해결했지만, 결과가 좀 나빴다. 이 논문에서는 $\epsilon$을 크게 설정하는 것이 bias를 야기해서 최적화 과정을 느리게 만들기 때문이라고 추측한다.

### [3.2] Analysis of Adaptive Learning Rate Variance

[Thm1] 만약 $\psi^2(.)$이 $\text{Scaled-inv-}\chi^2(\rho, \frac{1}{\sigma^2})$을 따른다면, $\rho$가 증가하면 $\text{Var}(\psi(.))$은 감소한다. 이 이론은 초기 단계의 샘플 부족으로 안해 $\text{Var}(\psi(.))$가 커진다는 것을 보여준다. *이것도..왜 샘플의 개수 때문이지?*

## [4] Rectified Adaptive Learning Rate

### [4.1] Estimate of $\rho$

$$p(\frac{(1-\beta_2)\sum_{i=1}^t\beta_2^{t-i}g_i^2}{1-\beta_2^t})\approx p(\frac{\sum_{i=1}^{f(t, \beta_2)}g_{t+1-i}}{f(t,\beta_2)})$$

로 주로 근사된다. $f(t, \beta_2)$는 SMA의 length인데, 이는 SMA와 EMA가 같은 무게중심을 갖게 해야 한다. 따라서 $f(t,\beta_2)=\frac{2}{1-\beta_2}-1-\frac{2t\beta_2^t}{1-\beta_2^t}$가 되어야 한다.

또한 우리는 $f(t,\beta_2)$로 $\rho$를 근사할 수 있으므로 $f(t,\beta_2)=\rho_t$라고 쓸 것이고, $\frac{2}{1-\beta_2}-1=\rho_{\infty}$라고 쓸 것이다.

### [4.2] Variance Estimation and Rectification

adaptive learning rate $\psi(.)$가 일관되는 분산을 가지게 하기 위해 이 논문에서는 rectification을 이용했다.

$$\text{Var}[r_t\psi(g_1, \cdots, g_t)]=C_{\text{Var}}$$

이때 $r_t=\sqrt{\frac{C_{\text{Var}}}{\text{Var}[\psi(g_1, \cdots, g_t)]}}$이고 $C_{\text{Var}}=\text{Var}[\psi(.)]\|_  {\rho_t=\rho_{\infty}}$이다.

$\text{Var}[\psi(.)]$는 수치적으로 stable하지 않으므로 rectified term을 1차 근사로 계산한다. 결론적으로 구한 $r_t$는

$$r_t=\sqrt{\frac{(\rho_t-4)(\rho_t-2)}{(\rho_\infty-4)(\rho_\infty-2)}}$$

이다. 이것을 적용한 변형된 Adam을 RAdam이라고 한다.

### [4.3] In Comparison with Warmup

$r_t$는 warmup과 비슷한 효과를 내며, warmup이 분산을 감소시키는 방식으로 작동된다는 것을 알았다. RAdam은 분산이 커질 때 adaptive lr을 비활성화시켜서 초기 불안정성을 막고, 하이퍼파라미터도 필요하지 않다.

## [5] Experiments

### [5.1] Comparing to Vanilla Adam

adaptive learning rate은 초기에 큰 분산을 가지고 있다. 이를 보완하기 위해 이 논문에서는 RAdam을 만들었는데, 이는 Adam보다 결과가 좋을 뿐만 아니라 learning rate에도 민감하지 않다. (라는데 learning rate에 민감하지 않은지는 그래프를 봐도 뚜렷하게 나타나지는 않는다.)

이 논문에서는 One billion word dataset (Language Modeling), CIFAR10, ImageNet (Image Classification)에 실험을 했는데, 이때의 결과로는 초기 분산 감소가 빠르고 정확도도 더 높은 결과를 가져오는데 좋은 역할을 했음을 알 수 있다. SGD와도 비교했는데, test accuracy는 SGD보다 좋지 않지만 train accuracy는 SGD보다 더 좋았다. (그런데 이말은 곧 overfitting이 더 된다는 말인 것 같다.)

또, RAdam, vanilla Adam, Adam with warmup을 여러 범위의 lr로 비교한 결과, RAdam이 가장 민감하지 않았다고 하는데, 이 부분은 그래프에서 그리 뚜렷한 차이가 보이지는 않는다.

### [5.2] Comparing to Heuristic Warmup

IWSLT 데이터셋에 실험해본 결과, RAdam은 Adam에 warmup을 사용한 것과 비슷한 양상을 보인다. 또한 RAdam은 더 적은 하이퍼파라미터 튜닝을 요한다. 이 결과 또한 RAdam이 분산이 큰 상황을 없애기 때문이라는 주장에 보탬이 된다.

### [5.3] Simulated Verification

실험으로 1차 근사한 rectification term도 잘 근사되었다는 것을 확인했고, 그 결과 rectification 된 $\psi(.)$의 분산이 일정하다는 점도 확인했다.

---

이 논문에서는 optimizer로 Adam을 사용한다.
