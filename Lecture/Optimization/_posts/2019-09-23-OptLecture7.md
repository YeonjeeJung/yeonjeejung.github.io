---
layout: post
title: Optimization Lecture 7
author: Yeonjee Jung
use_math: true
---

# Barrier Method & Proximal Gradient

## Barrier Method

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

## Proximal Gradient

### Proximal Gradient Descent

일반적으로 우리는 $f(x)$가 미분가능한 함수라고 생각하고 문제를 풀었지만, 사실 그렇지 않은 경우가 더 많다. 이런 경우를 잘 해결하기 위해 $f(x)$를 $g(x)$와 $h(x)$로 쪼갤 수 있다.

$$f(x) = g(x) + h(x)$$

여기서 $g(x)$는 미분가능한 nice function이고, $h(x)$는 미분불가능할 수도 있지만 해석하기 쉬운 additional function 이다. $g(x)$와 $h(x)$는 둘다 convex이다.

사실 gradient descent 방법의 함수 다음 $x$를 구하는 방법의 식은 다음과 같다.

$$\begin{align}x_{t+1} &= x_t - \gamma\triangledown f(x)\\
&= \arg\min_y\{f(x_t)+\triangledown f(x_t)^T(y-x_t)+\frac{1}{2\gamma}\|y-x_t\|^2\}\end{align}$$

테일러 전개를 이용한 것인데, 마지막 항은 $\triangledown^2f(x)=\frac{1}{\gamma}I$로 대체한 것이다. _왜 이렇게 대체를 할 수 있는지는 모르겠다_ 다시 써보면,

$$\begin{align}x_{t+1}&=\arg\min_y\{g(y)+h(y)\}\\
&=\arg\min_y\{g(x_t) + \triangledown g(x_t)^T(y-x_t)+\frac{1}{2\gamma}\|y-x_t\|^2 + h(y)\}\\
&=\arg\min_y\{\triangledown g(x_t)^T(y-x_t)+\frac{1}{2\gamma}\|y-x_t\|^2 \frac{\gamma}{2}\|\triangledown g(x_t)\|^2+ h(y)\}\\
&=\arg\min_y\{\frac{1}{2\gamma}(\|y-x_t\|^2 + 2\gamma\triangledown g(x_t)^T(y-x_t)+\gamma^2\|\triangledown g(x_t)\|^2)+h(y)\}\\
&=\arg\min_y\{\frac{1}{2\gamma}\|y-(x_t-\gamma\triangledown g(x_t))\|^2+h(y)\}
\end{align}$$

이다. 중간에 뜬금없이 추가되거나 삭제된 항은 $y$와 관계없는 항이기 때문에 추가가 가능하다. 여기서 $$\text{prox}_{h, \gamma}(z) = \arg\min_y\{\frac{1}{2\gamma}\|y-z\|^2+h(y)\}$$라고 정의하면,

$$x_{t+1} = \text{prox}_{h, \gamma}(x_t-\gamma\triangledown g(x_t))$$

라고 쓸 수 있다. 항상 이 방법이 좋은 것은 아니며, $f(x)$를 어떤 $g(x)$와 $h(x)$로 나누는지에 따라 효과가 달라진다.

### Generalized Gradient

$$G_{h, \gamma}(x) = \frac{1}{\gamma}(x-\text{prox}_{h, \gamma}(x-\gamma\triangledown g(x)))$$

라고 하면, $x_{t+1}= x_t-\gamma G_{h, \gamma}(x)$라고 할 수 있다. 이 식으로 그냥 gradient descent와 projected gradient descent도 포함시킬 수 있는데, 만약 $h(x)=0$이면 그냥 gradient descent이고, $$h(x)=\begin{cases}0 & \text{if }x\in X \\ \infty & \text{otherwise}\end{cases}$$ 라고 정의한다면 projected gradient descent이다.

### Convergence Analysis ($\beta$-smooth)

$\beta$-smooth function이므로 $\gamma=\frac{1}{\beta}$ 라고 하자. $G(x)=G_{h, \gamma}(x)$라고 하자.

$$\begin{align}g(x-\gamma G(x))&\le g(x) - \gamma\triangledown g(x)^TG(x)+\frac{\gamma^2\beta}{2}\|G(x)\|^2 \text{ (by }\beta\text{-smooth)}\\
&\le g(x)-\frac{1}{\beta}\triangledown g(x)^TG(x)+\frac{1}{2\beta}\|G(x)\|^2 \text{ (by }\gamma=\frac{1}{\beta}\text{)}\\
f(x-\frac{1}{\beta}G(x))&\le g(x)-\frac{1}{\beta}\triangledown g(x)^TG(x)+\frac{1}{2\beta}\|G(x)\|^2 + h(x-\frac{1}{\beta}G(x))\\
&\le g(z)+\triangledown g(x)^T(x-z)-\frac{1}{\beta}\triangledown g(x)^TG(x)+\frac{1}{2\beta}\|G(x)\|^2 + h(z)+(G(x)-\triangledown g(x))^T(x-z-\frac{1}{\beta}G(x))\text{ (by Convexity)}\\
& \le g(z)+h(z)+G(x)^T(x-z)-\frac{1}{2\beta}\|G(x)\|^2 \text{ (by }G(x)-\triangledown g(x)=\triangledown h(x-\frac{1}{\beta}G(x)\text{)})\end{align}$$

$x=x_t, z=x^* $를 대입하면,

$$f(x_t-\frac{1}{\beta}G(x_t))\le g(x^* )+h(x^* )+G(x_t)^T(x_t-x^* )-\frac{1}{2\beta}\|G(x_t)\|^2$$

이고, $f(x_t-\frac{1}{\beta}G(x))=f(x_{t+1})$, $g(x^* )+h(x^* )=f(x^* )$이므로

$$f(x_{t+1})-f(x^* )\le G(x_t)^T(x_t-x^* )-\frac{1}{2\beta}\|G(x_t)\|^2$$

이 된다.

### (Proof) $$G(x)-\triangledown g(x)=\triangledown h(x-\gamma G(x))$$

위에서 그냥 넘어간 이 명제를 증명해보자.

$$\begin{align}G(x) &= \frac{1}{\gamma}(x-\text{prox}_{h, \gamma}(x-\gamma\triangledown g(x)))\\
&=\frac{1}{\gamma}(x-\arg\min_y\{\frac{1}{2\gamma}\|y-(x-\gamma\triangledown g(x))\|^2 + h(y)\})\\
\Rightarrow x-\gamma G(x)&=\arg\min_y\{\frac{1}{2\gamma}\|y-(x-\gamma\triangledown g(x))\|^2 + h(y)\}\end{align}$$

$\arg\min$이므로 마지막 식의 우항을 미분해서 좌항을 넣으면 $0$이 되어야 한다.

$$\begin{align}&\Rightarrow \frac{1}{\gamma}(x-\gamma G(x)-x+\gamma \triangledown g(x))+\triangledown h(x-\gamma G(x))=0\\
&\Rightarrow G(x)-\triangledown g(x) = \triangledown h(x-\gamma G(x))\end{align}$$
