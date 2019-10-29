---
layout: post
title: Optimization Lecture 7
author: Yeonjee Jung
use_math: true
---

# Barrier Method

### Inequality Constrained Problems

최적화해야 하는 함수 $f(x)$가 제한된 집합 $X$에서 정의될 때 projected gradient descent 외의 다른 방법을 소개한다. 이런 constrained optimization 문제를 다르게 말하면, '어떤 함수 $h, g$에 대하여 모든 $1\le i \le m$인 $i$들에 대해 $h_i(x)=0$을 만족하고, 모든 $1\le j \le r$인 $j$들에 대해 $g_j(x)\le 0$을 만족하면서 $f(x)$를 최소화시켜라'라고 표현할 수 있다. 이것을 Inequality Constrained Problem이라고 한다.

### Barrier Method

Barrier Method란, constrained 문제를 unconstrained 문제로 바꾸는 방법을 말한다. 이 때 원래의 최적화해야 하는 함수 뒤에 penalty function을 더하게 된다. 그런데 여기서, 과연 이렇게 찾은 해가 우리가 실제로 원하던 해일까? 라는 의문을 갖게 된다.

### Lagrange Multipliers

라그랑지 계수로 이를 해결할 수 있는데, 위에서 소개한 Inequality Constrained Problem을 이용하자. 먼저 Lagrange function을 다음과 같이 정의할 수 있다.

$$\Lambda(x, \mu, \lambda):=f(x) + \sum_{i=1}^m{\mu_ih_i(x)} + \sum_{j=1}^r{\lambda_jg_j(x)}$$

여기서 $\mu_1, \cdots, \mu_m, \lambda_1, \cdots, \lambda_r$을 Lagrange multiplier라고 한다.

### Lagrange Dual

다음은 Lagrange function을 최적화하는 방법이다. 먼저 Lagrange dual function과 Lagrange dual problem을 정의한다.

Lagrange dual function :

$$L(\mu, \lambda) := \min_x\Lambda(x, \mu, \lambda)$$

Lagrange dual problem :

$$\max_{\mu, \lambda}L(\mu, \lambda) \text{ s.t. }\lambda_j\ge, \forall 1 \le j\le r$$

먼저 Lagrange dual function 값을 구한 뒤, Lagrange dual problem을 푼다. 그 후 $L(\mu, \lambda)$를 최대화시키는 $\lambda$와 $\mu$를 찾아서 $\Lambda(x, \mu, \lambda)$에 넣고 다시 Lagrange dual function을 풀고 $\cdots$를 반복한다. 최종 나오는 $x$값이 우리가 원하던 $x^* $값이다.

Lagrange dual problem을 푸는 과정에서는, $L(\mu, \lambda)$가 concave function이라는 점을 활용하면 gradient ascent를 이용해서 maximization을 할 수 있다.

### (Proof) $L(\mu, \lambda)$ is convex

$$\begin{align}&L(\alpha \mu^{(1)}+(1-\alpha)\mu^{(2)}, \alpha\lambda^{(1)}+(1-\alpha)\lambda^{(2)})\\
&=\min_x(f(x)+\sum(\alpha\mu^{(1)}+(1-\alpha)\mu^{(2)})h_i(x) + \sum(\alpha\lambda^{(1)}+(1-\alpha)\lambda^{(2)})g_j(x))\\
&\ge \alpha(\min(f(x)+\sum\mu^{(1)}h_i(x))+\sum\lambda^{(1)}g_j(x))+(1-\alpha)(\min(f(x)+\sum\mu^{(2)}h_i(x))+\sum\lambda^{(2)}g_j(x))\\
&=\alpha L(\mu^{(1)}, \lambda^{(1)})+(1-\alpha)L(\mu^{(2)}, \lambda^{(2)})\end{align}$$

따라서 $L(\mu, \lambda)$는 concave 함수이다.

### Lagrange Dual and Barrier

다시 Barrier로 돌아오면, Lagrange dual에서 penalty function은 $\sum\mu_ih_i(x)+\sum\lambda_jg_j(x)$이다. Lagrange dual problem에서 우리는 $L(\mu, \lambda)$를 최대화시켜야 하므로,

1. $g_j(x)\lt 0$이면 $\lambda_j^* =0$이어야 하고,  
2. $g_j(x) \gt 0$이면 $\lambda_j^* = \infty$,  
3. $g_j(x)=0$이면 $\lambda_j^* $는 아무 숫자나 되어도 상관없으므로 $\lambda_j^* \gt 0$이게 된다.

barrier의 관점에서 봐도 말이 된다. barrier를 만족하지 못하면 penalty function이 무한대로 가게 되므로, 해가 constrained set 안으로 무조건 들어가도록 하게 된다. 따라서, Lagrange dual로 구한 $x^* $는 우리가 원하던 최적 해가 맞다.

이렇게 구한 해인 $x^* (\mu^* , \lambda^* )$에는 몇 가지의 특성이 있다.

1. $\triangledown f(x^* (\mu^* , \lambda^* ))+\sum\mu_i^* \triangledown h_i(x^* (\mu^* , \lambda^* ))+\sum\lambda_j^* \triangledown g_j(x^* (\mu^* , \lambda^* ))=0$  
2. $h_i(x^* (\mu^* , \lambda^* ))=0, \forall i$  
3. $\lambda_j^* g_j(x^* (\mu^* , \lambda^* ))=0 \text{ when }\lambda_j^* =0, g_j(x^* (\mu^* , \lambda^* ))\lt 0$

$f(x)$나 constrained set이 convex가 아니면 진짜 해와 우리가 구한 해가 차이가 나게 되는데, 이것을 Lagrange Gap이라고 한다.
