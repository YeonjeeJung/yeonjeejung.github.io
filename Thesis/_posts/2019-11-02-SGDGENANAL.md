---
layout: post
title: SGD - General Analysis and Improved Rates
author: Yeonjee Jung
tags : Optimization
use_math : true
---

## [Abstract]

이 논문에서는 임의 샘플링에 대한 SGD의 수렴을 설명하는 이론을 제안한다. 이 이론은 SGD의 여러 종류의 수렴을 설명하는데, 각각은 mini-batch를 형성하는 데에 특정한 확률 분포에 연관되어 있다. 이런 분석을 한 것은 처음이며, 분석에 사용된 SGD는 대부분 이전에 논문에서는 명시적으로 고려되지 않았던 것들이다. 이 분석은 최근에 소개된 expcted smoothness에 의존하며, stochastic gradient들의 bound에는 의존하지 않는다. replacement sampling과 independent sampling 등과 같은 여러 다른 minibatch 전략에 대해 연구하므로써, 우리는 stepsize를 mini-batch의 크기에 대한 하나의 함수로써 표현해냈다. 이렇게 하면 전체 복잡도를 최소화하는 mini-batch의 크기를 정할 수 있으며, 최소로 evaluate된 stochastic gradient의 분산이 커지면 최적의 mini-batch 크기도 증가한다는 것을 명시적으로 보여주었다. 분산이 0이면 최적의 mini-batch 크기는 $1$이다. 더해서, 상수였던 stepsize를 언제 감소시켜야 하는지를 결정하는 stepsize-switching 규칙을 증명한다.

## [1] Introduction

우선 이런 최적화 문제를 풀려고 한다. (이것은 감독학습 문제의 일반적인 형태이다)

$$x^* = \arg\min_{x\in \mathbb{R}^d}\left[f(x) =\frac{1}{n}\sum_{i=1}^n f_i(x)\right]$$

모든 $f_i : \mathbb{R}^d\rightarrow\mathbb{R}$은 smooth이다. 그리고 $f$는 global minimizer $x^* $를 가지며, $\mu$-strongly quasi-convex이다. 따라서

$$f(x^* )\ge f(x) +\left<\triangledown f(x),x^* −x\right>+\frac{\mu}{2}\|x^* −x\|^2$$

를 만족한다.

### [1.1] Background and contributions

#### Linear convergence of SGD

2011년에는 strongly convex한 $f$를 특정 noise level까지 수렴하게 하는 비점근 분석의 연구가 있었다. 2016년에는 condition number에 대한 2차 의존성을 없애서 위 결과를 개선하고, importance sampling을 고려했다. 위 연구의 결과는 이후에 mini-batch 다양성에 대한 연구로 확장되었다. 이 논문에서는 위 결과를 대부분의 샘플링에 대해 적용할 수 있도록 했다. 분산 감소 방법의 관점에서 expected smoothness assumption을 도입했는데, 이 assumption은 SGD에서 사용하는 $f$와 $\mathcal{D}$의 연관 특징으로, 이 논문에서 일반적인 복잡도 결과(Thm 3.1)를 증명할 수 있게 해주었다. strong convexity 없이(사실은 quasi-convexity를 가정하였다) 선형 수렴도를 얻을 수 있었다. 또한, 함수 $f_i$들은 convex가 아니어도 된다.

#### Gradient noise assumptions

이전의 많은 SGD의 선형 수렴에 관한 연구들은 확률론적 gradient의 분산이 bound됨을 가정하고 증명했다. 이후에는 점점 가정을 약하게 하는 방향으로 가고 있다. 이 논문에서는 직접적으로 gradient의 bound는 가정하지 않고, 더 약한 가정인 expected smoothness assumption을 사용한다.

#### Optimal mini-batch size

이전의 연구 결과로 큰 mini-batch 크기를 사용하는 것이 넓은 범위의 non-convex문제를 푸는 것에 효과적이라는 것이 밝혀졌는데, 저자는 최적의 stepsize가 mini-batch 크기에 따라 선형적으로 늘어난다고 추측했다. 이 논문에서는 이 추측이 최적의 mini-batch 크기까지 커질 때에만 적용되며, mini-batch 크기에 따른 최적의 stepsize의 정확한 공식이 있음을 증명하고 그 공식을 제공한다.

#### Learning schedules

이전 연구에서는 해 주위에서 SGD의 수렴을 감지하는 방법이 제안되었다. 이 논문에서는 stepsize를 언제 줄여야 하는지 알려주는 식을 제공한다. 또한, 샘플링 방법에 따라, 그리고 mini-batch 크기에 따라 stepsize와 반복 복잡도가 어떻게 증가하고 감소하는지 보여준다.

#### Over-parameterized models

이전 연구에서 데이터보다 파라미터 수가 많을 때의 SGD의 수렴 속도를 분석한 연구가 있었다. 이 논문에서는 over-parametrized model의 경우 최적의 mini-batch 크기가 $1$임을 보여준다.

### [1.2] Stochastic reformulation

sampling vector를 이용해서 각 함수와 gradient의 weight를 다르게 만든다. 평균을 취하면 결국 원래의 함수 또는 gradient와 같게 된다. 이 방법은 이전에도 분산을 줄이는 목적으로 사용되었다.

$$x^{k+1}=x^k-\gamma^k\triangledown f_{c^k}(x^k)$$

를 이용해서 gradient descent를 할 수 있다. 이 논문에서는 여러 다른 분포 $\mathcal{D}$에 대한 위 gradient descent의 수렴을 분석할 것이다. 또한 수렴을 얻기 위해 SGD를 변형한다.

## [2] Expected Smoothness and Gradient Noise

### [2.1] Expected smoothness

expected smoothness는 $\mathcal{D}$ 분포의 특성과 $f$의 smoothness 특성을 결합한 것이다.

#### Assumption 2.1 (Expected Smoothness)

$$\mathbb{E}_\mathcal{D}[\|\triangledown f_v(x)-\triangledown f_v(x^* )\|^2]\le 2\mathcal{L}(f(x)-f(x^* )), \forall x\in\mathbb{R}^d$$

인 $\mathcal{L}=\mathcal{L}(f, \mathcal{D})\gt0$이 존재할 때, $f$가 분포 $\mathcal{D}$에 대해 $\mathcal{L}$-smooth in expectation이라고 한다. 간단히, $(f, \mathcal{D})\sim ES(\mathcal{L})$이라고 표현한다. 이것이 expected smoothness의 정의이다. 이 expected smoothness는 $f_i$나 $f$가 convex가 아니어도 성립할 수 있다.

### [2.2] Gradient noise

이 논문에서 쓰이는 중요 가정 중 두번째는 gradient noise의 유한성이다. 이 가정은 매우 약한 가정이고, $f$보다는 $\mathcal{D}$에 관한 가정이다.

### [2.3] Key lemma and connection to the weak growth condition

보통 SGD의 수렴을 증명할 때는

$$\mathbb{E}\|\triangledown f_v(x)\|^2\le c$$

라는 가정이 들어가는데, $f$가 strongly convex일 때는 보통 성립하지 않는다. 이 논문에서는 이 가정 대신에 확률론적 gradient의 기댓값을 bound하기 위해 expected smoothness를 사용한다.

#### [Lem 2.4]

만약 $(f, \mathcal{D})\sim ES(\mathcal{L})$이면,

$$\mathbb{E}_\mathcal{D}[\|\triangledown f_v(x)\|^2]\le 4\mathcal{L}(f(x)-f(x^* ))+2\sigma^2$$

이다.

$\sigma=0$인 zero-noise setting에서는 이 식이 weak growth condition이라고 알려져 있다. 이전에도 비슷한 가정이 있었는데, 이 논문에서 제시된 bound가 더 강력하다.

## [3] Convergence Analysis

### [3.1] Main rules

#### [Thm 3.1]

$f$가 $\mu$-quasi-strongly convex이고 $(f, \mathcal{D})\sim ES(\mathcal{L})$이라고 하자. 모든 $k$에 대하여 $\gamma^k=\gamma\in(0, \frac{1}{2\mathcal{L}}]$을 고른다. 그러면 SGD는 다음을 만족한다.

$$\mathbb{E}\|x^k-x^* \|^2\le(1-\gamma\mu)^k\|x^0-x^* \|^2+\frac{2\gamma\sigma^2}{\mu}$$

따라서 모든 $\epsilon \gt 0$에 대해 $\gamma=\min\{\frac{1}{2\mathcal{L}}, \frac{\epsilon\mu}{4\sigma^2}\}$와 $$k\ge\max\{\frac{2\mathcal{L}}{\mu}, \frac{4\sigma^2}{\epsilon\mu^2}\}\log\left(\frac{2\|x^0-x^* \|^2}{\epsilon}\right)$$을 고르면,

$$\mathbb{E}\|x^k-x^* \|^2\le\epsilon$$

을 만족한다.

이 Thm이 말하는 바는 SGD가 gradient noise $\sigma^2$과 stepsize $\gamma$에 의존하는 $\frac{2\gamma\sigma^2}{\mu}$까지 선형적으로 수렴한다는 것이다. 더 작은 stepsize를 사용하면 더 수렴할 수 있지만, 수렴도가 떨어진다. $\mathcal{D}$를 조절할 수 있으므로, $\sigma^2$와 $\mathcal{L}$도 조절이 가능하다. stepsize 를 계속 감소시키면 위 항을 더 작게 만들 수도 있다.

#### [Thm 3.2]

$f$가 $\mu$-quasi-strongly convex이고 $(f, \mathcal{D})\sim ES(\mathcal{L})$이라고 하자. $\mathcal{K}=\frac{\mathcal{L}}{\mu}$라고 하고,

$$\gamma^k=\begin{cases}\frac{1}{2\mathcal{L}} & \text{for }k\le 4\lceil\mathcal{K}\rceil\\ \frac{2k+1}{(k+1)^2\mu} & \text{for }k\gt 4\lceil\mathcal{\mathcal{K}}\rceil\end{cases}$$

라고 하자. 만약 $k\ge 4\lceil\mathcal{K}\rceil$이면 다음을 만족하며 수렴한다.

$$\mathbb{E}\|x^k-x^* \|^2\le\frac{\sigma^2}{\mu^2}\frac{8}{k}+\frac{16\lceil{\mathcal{K}\rceil^2}}{e^2k^2}\|x^0-x^* \|^2$$

### [3.2] Choosing $\mathcal{D}$

위 gradient descent가 효과적이려면, sampling vector $v$가 sparse 해야 한다._왜?_ 이 논문에서는 proper sampling 만을 고려한다. sampling $S$가 proper 하다는 말은 모든 $i$에 대해, $i$가 $S$안에 존재할 확률 $p_i$, 즉 $i \in C$인 모든 $C$의 확률을 다 더한 것이 양수라는 말이다. ($\mathbb{P}[i\in S] = \sum_{C:i\in C}p_C \ge 0, \forall i$) 이 proper sampling은 sampling $S$에 의존하는 sampling vector를 특수화 한, 이 논문에서는 특수한 케이스이다. proper sampling의 정의를 기반으로 Independent sampling, Partition sampling, Single element sampling, $\tau$-nice sampling을 정의할 수 있다.

### [3.3] Bounding $\mathcal{L}$ and $\sigma^2$

$f$가 convex이고 smooth하다고 가정하면 expected smoothness $\mathcal{L}$과 $\sigma^2$에 대한 closed form expression을 계산할 수 있다. 그리고 sampling을 이용해 $\mathcal{L}$과 $\sigma^2$에 대한 bound를 할 수 있는데, sampling의 종류에 따라 이 bound가 달라진다.

## [4] Optimal Mini-Batch Size
위에서 나온 결과에 $\|S\|=n$인 경우를 대입해 보면 우리가 원래 알고 있던 full-batch gradient descent결과와 같다. 따라서 full-batch gradient descent는 SGD의 특별한 케이스라고 볼 수 있다.

### [Nontation]

$$L_i=\lambda_{\max}(M_i)$$  
$$L_{\max}=\max_{i\in[n]}L_i$$  
$$\bar{h} = \frac{1}{n}\sum_{i\in[n]}\|h_i\|^2$$

### [4.1] Nonzero gradient noise
mini-batch 크기에 따라 복잡도가 어떻게 변하는지 알기 위해 $\|S\|=\tau$인 independent sampling과 $\tau$-nice sampling의 경우를 생각해보자.

Independent sampling의 경우에는 bound된 결과를 Thm 3.1에 넣으면 최소 몇 번의 반복을 해야 하는지 알 수 있는데, 이것은 현재 나온 것 중 최적이다.

$$k\ge\frac{2}{\mu}\max\{L+\max_{i\in[n]}\frac{1-p_i}{np_i}L_i, \frac{2}{\mu\epsilon}\frac{1-p_i}{np_i}\bar{h}\}$$

또한 최적의 step size $\gamma$도 알 수 있게 되고, $\tau$가 커질수록 stepsize는 줄어든다는 것을 알 수 있다. 따라서 전체 반복 수를 최소화하는 mini-batch 크기($\tau$)를 선택할 수 있고, 계산으로 이를 구할 수 있다. $\tau$-nice sampling에서도 independent sampling과 같은 방법으로 최적의 mini-batch 크기를 구할 수 있다.

### [4.2] Zero gradient noise

$\sigma=0$인 경우를 생각해 보자. Thm 3.1에 따르면 SGD의 반복 복잡도는 매우 간단해진다.

$$k\ge\frac{2L}{\mu}$$

이 경우에 $f$는 weak growth condition을 만족하게 되고, 이전의 연구 결과와 직접적으로 비교할 수 있게 된다. $\tau$가 $n$까지 커질수록 복잡도가 개선되지만, 전체 복잡도는 $\tau$가 곱해지기 때문에 이것으로는 충분하지 않아서 $\tau=1$이 최적의 mini-batch 크기가 된다._이부분이 잘 이해가 안됨..저걸로는 충분하지 않아서 batch-size를 1로 한다고..?_

## [5] Importance Sampling

이 논문에서는 single element sampling과 independent sampling에 대한 importance sampling을 각각 제안한다.

### [5.1] Single element sampling

$\mathcal{L}$에 대한 bound와 $\sigma^2$을 Thm 3.1에 대입하면 최소 반복 수를 얻을 수 있는데, $p_i$에 대해 이 최소 반복 수를 최소화하는 문제는 매우 어렵기 때문에 이 논문에서는 $$\mathcal{L}_{\max}$$ 와 $\sigma^2$ 을 줄이는 것에 집중한다. single element sampling에 대해서는 이전 연구에 있었던 partially biased sampling을 떠올릴 수 있다. $$\mathcal{L}_{\max}$$를 최소화시키는 $p_i$는 $p_i^\mathcal{L}=\frac{L_i}{\sum_{j\in [n]} L_j}$인데, 이 확률을 적용하면 이전 연구에 있었던 partially biased sampling과 똑같아진다. uniform sampling과 비교해보면 $L_{\max}=n\bar{L}$인 극한의 상황에서는 partially biased sampling이 절반의 복잡도를 가질 수 있다. _왜 갑자기 uniform sampling이랑 비교하지?_

### [5.2] Minibatches
이 논문에서는 mini-batch SGD를 위한 importance sampling을 최초로 제안한다. 이에 대한 결과는

$$k\ge\max\{(1-\frac{2}{\tau})\frac{2\bar{L}}{\alpha\mu}, (\frac{2}{\tau}-\frac{1}{n})\frac{8\bar{h}}{\epsilon\mu^2}\}$$

인데, $L_{\max}$에 대한 의존성을 없앴을 뿐 아니라, $\tau$가 커지면 복잡도가 더 줄어든다.

## [6] Experiments

### [6.1] Constant vs. decreasing step size
이 실험에서는 ridge regression과 logistic regression 문제에 집중했는데, 여기서는 목적 함수가 strongly convex이다. Thm 3.2에서 예상했듯이, 특정 시점에 step size를 줄이면 성능이 더 좋다는 것을 보여준다.

### [6.2] Minibatches
이 실험에서는 [3.2]에 소개된 여러 다른 분포의 $\mathcal{D}$(single element sampling, $\tau$ independent sampling, $\tau$-nice sampling)를 선택하여 SGD의 수렴도를 비교하였는데, $\tau^* $를 이용한 방법이 가장 좋은 결과를 보였다. _여기 비교했다는건 6갠데 그래프에는 왜 5개만 있음?_

### [6.3] Sum-of-non-convex functions
이 실험에서는 PCA 문제에 집중했는데, 여기서는 $f$는 strongly convex지만 $f_i$는 그렇지 않다. non-convex한 $f_i$에서도 위와 마찬가지로, stepsize를 줄였을 때 더 좋은 성능을 보였고 $\tau^* $가 가장 좋은 성능을 보였다.

---

아직 잘 모르겠는 개념 : $\mu$-strongly quasi-convx, condition number, uniform sampling
