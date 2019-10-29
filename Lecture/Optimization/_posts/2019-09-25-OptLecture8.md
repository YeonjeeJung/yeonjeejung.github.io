---
layout: post
title: Optimization Lecture 8
author: Yeonjee Jung
use_math: true
---

# Subgradient & Mirror Descent

## Subgradient

만약 함수 $f(x)$가 미분불가능하다면, 우리는 임의의 gradient를 정해야 한다. subgradient를 정의할 수 있는데,

$$f(y) \ge f(x) + g^T(y-x), \forall y\in X$$

인 모든 $g$를 subgradient라고 한다. $\partial f(x)$가 subgradient의 집합을 의미한다. 원래함수 $f(x)$가 convex라면 subgradient에서 gradient descent를 써서 같은 결과를 낼 수 있다. subgradient는 저 조건만 충족하면 되기 때문에 한 점에서 여러 개의 subgradient가 발생할 수 있다.

### Subgradient Descent : L-Lipschitz continuous

여태까지 우리가 $\gamma = \frac{1}{\beta}$를 쓸 수 있었던 것은 $\beta$-smooth를 가정했기 때문인데, 실제 미분값 대신 subgradient를 사용해야 하는 함수라면 smooth한 함수가 아닐 가능성이 크다. 따라서 모든 subgradient에서는 Lipschitz continuous를 가정한다.

$$\begin{align}f(x_t) -f(x^* ) &\le g_t^T(x_t-x^* )\\
&=\frac{1}{\gamma}(x_t-x_{t+1})^T(x_t-x^* )\\
&= \frac{1}{2\gamma}(\|x_t-x_{t+1}\|^2+\|x_t-x^* \|^2-\|x_{t+1}-x^* \|^2)\\
&\le \|g_t\|^2+\frac{1}{2\gamma}(\|x_t-x^* \|^2-\|x_{t+1}-x^* \|^2)\end{align}$$

라고 할 수 있는데, 모든 $t$에 대하여 다 더하면

$$\sum_{t=0}^{T-1}(f(x_t)-f(x^* ))\le\frac{1}{2\gamma}(\|x_0-x^* \|^2-\|x_T-x^* \|^2)+\sum_{t=0}^{T-1}\frac{\gamma}{2}\|g_t\|^2$$

이라고 할 수 있다. 다시 말하면,

$$\begin{align}f(\bar{x})-f(x^* ) &\le \frac{1}{T}\sum_{t=0}^{T-1}(f(x_t)-f(x^* ))\\
&\le \frac{1}{T}\{\frac{1}{2\gamma}(\|x_0-x^* \|^2-\|x_T-x^* \|^2)+\sum_{t=0}^{T-1}\frac{\gamma}{2}\|g_t\|^2\}\\
&\le \frac{1}{T}(\frac{1}{2\gamma}\|x_0-x^* \|^2+\sum_{t=0}^{T-1}\frac{\gamma}{2}\|g_t\|^2) \\
&\le(\frac{1}{2\gamma}R^2+\frac{\gamma}{2}B^2T)\\
&= \frac{1}{T}(\frac{B\sqrt{T}}{2R}R^2+\frac{R}{2R\sqrt{T}}B^2T)\\
&\le \frac{BR}{\sqrt{T}}\end{align}$$

라고 할 수 있다.

## Mirror Descent

지금까지의 모든 결과들(특히 Lipschitz에 관해서)은 유클리드 공간에서 정의되었다. 그런데 Lipschitz는 norm에 따라서 크기가 달라지는데, 다른 norm에 관해서는 어떤 convergence speed를 가지게 될까 하는 궁금증이 생기게 된다.

### Dual Space

이 궁금증을 해결하기 위해, 먼저 Dual space를 정의한다. 모든 벡터공간 $V$는 $V$에서 정의된 모든 선형 함수에 대해서 항상 dual space $V^* $를 갖는다. 모든 Tangent Line ($y=f(x^* )+f'(x^* )(x-x^* )$)는 항상 선형이기 때문에, 모든 gradient에 대해서는 항상 dual space를 갖는다.

### Dual Norm

$\mathbb{R}^n$에서 정의된 모든 norm $$\|\cdot \|$$에 대해 dual space에서의 norm 또한 항상 존재하는데, dual norm $$\|\cdot\|_ * $$은 다음과 같이 정의한다.

$$\|g\|_ * =\sup_{x\in\mathbb{R}^n:\|x\|\le 1}g^T x$$

말이 어려운데, $p$-norm에 대해 생각해 보면 다음과 같은 관계가 있다고 한다.

$$\frac{1}{p}+\frac{1}{q} = 1$$

즉, 원래 공간에서 $p$-norm을 사용하였다면 dual space에서는 $q$-norm을 사용하면 된다. 이러한 새로운 dual의 정의에서, Lipschitz는 다음과 같이 쓸 수 있다.

$$\|g\|_ * \le L, \forall x \in X, \forall g \in \partial f(x)$$

### Bregman Divergence

Dual space에서의 convergence를 해석하기 위해 Bregman divergence라는 것을 정의한다.

$$D_f(x,y)=f(x)-f(y)-\triangledown f(y)^T(x-y)$$

사실 첫 항을 제외한 항은 Tangent Line을 의미하는 식이다. 결국 Bregman divergence는 한 점에서의 접선에 대해 같은 $y$값에서 원래 함수와 Tangent Line의 차이를 의미한다. Bregman divergence에 관한 특성으로는 다음과 같은 것이 있다.

1. $(\triangledown f(x)-\triangledown f(y)^T(x-z)=D_f(x,y)+D_f(z, x)-D_f(z, y)$  
2. $\lambda$-strongly convex인 함수 $h$에 대하여 $$D_f(x, y)\ge \frac{\lambda}{2}\|x-y\|^2\ge 0$$

### Mirror Map

우선 $D\in\mathbb{R}^n$은 $X\subset\bar{D}$인 open set이라고 하자. Mirror Map $\Phi$는 $D$에서 $\mathbb{R}^n$으로의 mapping function인데, 다음과 같은 조건이 있다.

1. $\Phi$는 convex하고 미분가능한 함수이다.  
2. $\Phi$의 gradient는 어떤 숫자든 가능하다.  
3. $\Phi$의 gradient는 $D$의 가장자리에서 발산한다.

이렇게 놓고 보면, 이전에 우리가 썼던 gradient descent 식 $x_{t+1}=x_t-\gamma\triangledown f(x_t)$가 좀 이상해 보이기 시작한다. $x_t$는 원래 공간인데, $\triangledown f(x_t)$는 dual space에서 정의되는 것이기 때문이다. 사실 이전에는 유클리드 norm을 기준으로 진행했기 때문에 dual space의 norm도 유클리드 norm이 되어서 상관이 없었다. 그렇지만 이제는 다르므로 gradient descent를 새롭게 정의할 필요가 있다.

### Mirror Descent

다음은 Dual space 공간에서의 gradient descent 알고리즘이다. Mirror Descent라고도 한다.

1. $x_t$를 mirror map $\triangledown \Phi (x_t)$에 매핑시킨다. 이후는 모두 dual space이다.  
2. $\triangledown \Phi (x_t)-\gamma \triangledown f(x_t)$  
3. $\triangledown\Phi(y_{t+1})=\triangledown\Phi(x_t)-\gamma\triangledown f(x_t)$를 만족하는 $y_{t+1}$를 찾는다.  
4. 다시 원래 공간으로 가져오는데, constrained set $X$ 안에 $x_{t+1}$이 있어야 하기 때문에 projection을 한다. $$x_{t+1}=\Pi_X^\Phi(y_{t+1})=\arg\min_{x\in X}D_\Phi(x, y_{t+1})$$

Mirror Descent는 proximal gradient와도 연결된다.

$$\begin{align}x_{t+1}&=\arg\min_{x\in X}D_\Phi(x, y_{t+1})\\
&=\arg\min_{x\in X}\{\Phi(x)-\triangledown \Phi(y_{t+1})^Tx-\Phi(y_{t+1})+\triangledown\Phi(y_{t+1})^Ty_{t+1}\}\\
&=\arg\min_{x\in X}\{\Phi(x)-\triangledown\Phi(y_{t+1})^Tx\}\\
&=\arg\min_{x\in X}\{\Phi(x)-(\triangledown\Phi(x_t)-\gamma\triangledown f(x_t))^Tx\}\\
&=\arg\min_{x\in X}\{\gamma\triangledown f(x_t)^Tx+\Phi(x)-\Phi(x_t)-\triangledown\Phi(x_t)^T(x-x_t)\}\\
&=\arg\min_{x\in X}\{\gamma\triangledown f(x_t)^Tx+D_\Phi(x, x_t)\}\end{align}$$

중간에 $y$만에 관한 항들은 $\arg\min$이므로 마음대로 넣어도 상관 없고, 마찬가지로 $x_t$만에 관한 항은 마음대로 넣어도 상관없다.

### Mirror Descent : L-Lipschitz continuous

우선 함수에 대해서, $\Phi$는 $\rho$-strongly convex이고, $f$는 convex이고 L-Lipschitz이다. 그리고 $$R^2=\sup_{x\in X}\{\Phi(x)-\Phi(x_1)\}$$이다.

$$\begin{align}f(x_t)-f(x^* ) & \le g_t^T(x_t-x^* )\\
&= \frac{1}{\gamma}(\triangledown\Phi(x_t)-\triangledown\Phi(y_{t+1}))^T(x_t-x^* )\\
&=\frac{1}{\gamma}(D_\Phi(x^* , x_t)+D_\Phi(x_t, y_{t+1})-D_\Phi(x^* , y_{t+1}))\\
&\le\frac{1}{\gamma}(D_\Phi(x^* , x_t)+D_\Phi(x_t, y_{t+1})-D_\Phi(x^* , x_{t+1})-D_\Phi(x_{t+1}, y_{t+1}))\text{ (by } \triangledown\Phi(x^* , y_{t+1})\ge D_\Phi(x^* , x_{t+1})+D_\Phi(x_{t+1}, y_{t+1})\text{)}\\
&= \frac{1}{\gamma}(D_\Phi(x^* , x_t)-D_\Phi(x^* , x_{t+1}))+\frac{1}{\gamma}(D_\Phi(x_t, y_{t+1})-D_\Phi(x_{t+1}, y_{t+1}))\end{align}$$

모든 $T$에 대해서 다 더하면

$$\sum_{t=1}^T(f(x_t)-f(x^* ))=\frac{1}{\gamma}(D_\Phi(x^* , x_1)-D_\Phi(x^* , x_{T+1}))+\frac{1}{\gamma}\sum_{t=1}^T(D_\Phi(x_t, y_{t+1})-D_\Phi(x_{t+1}, y_{t+1}))$$

인데, 마지막 $\sum$항만 bound할 수 있다.

$$\begin{align}D_\Phi(x_t, y_{t+1})-D_\Phi(x_{t+1}, y_{t+1})&=\Phi(x_t)-\Phi(x_{t+1})-\triangledown\Phi(y_{t+1})^T(x_t-x_{t+1})\\
&\le(\triangledown\Phi(x_t)-\triangledown\Phi(y_{t+1}))^T(x_t-x_{t+1})-\frac{\rho}{2}\|x_t-x_{t+1}\|^2\text{ (by }\rho\text{-strongly convex)}\\
&=\gamma g_t^T(x_t-x_{t+1})-\frac{\rho}{2}\|x_t-x_{t+1}\|^2\\
&\le \gamma L\|x_t-x_{t+1}\|-\frac{\rho}{2}\|x_t-x_{t+1}\|^2\text{ (by L-Lipschitz)}\\
&\le\frac{\gamma^2L^2}{2\rho}\end{align}$$

마지막 항은, 그 전 식이 $$\|x_t-x_{t+1}\|^2$$에 관한 이차식이고, 위로 볼록한 함수이기 때문에 미분해서 $0$이 되는 점이 최대점이라는 점을 이용했다. 다시 $\sum$으로 돌아가면,

$$\begin{align}\sum_{t=1}^T(f(x_t)-f(x^* ))&\le\frac{1}{\gamma}(D_\Phi(x^* , x_1)-D_\Phi(x^* , x_{T+1}))+\frac{1}{\gamma}\cdot T\cdot\frac{\gamma^2L^2}{2\rho}\\
&=\frac{1}{\gamma}(D_\Phi(x^* , x_1)-D_\Phi(x^* , x_{T+1}))+\frac{\gamma TL^2}{2\rho}\\
&\le\frac{1}{\gamma}D_\Phi(x^* , x_1)+\frac{\gamma TL^2}{2\rho}\text{ (by Bregman Divergence property, }D_\Phi(x^* , x_{T+1})\ge 0\text{)}\\
&\le \frac{R^2}{\gamma}+\frac{\gamma TL^2}{2\rho}\text{ (아직도 왜이런지 모르겠음)}\\
&\le RL\sqrt{\frac{2T}{\rho}}\end{align}$$

따라서, 다음과 같은 결론이 나온다.

$$f(\bar{x})-f(x^* )\le RL\sqrt{\frac{2}{\rho T}}$$

### (Proof) $$D_\Phi(x, y_{t+1})\ge D_\Phi(x, x_{t+1})+D_\Phi(x_{t+1}, y_{t+1})$$

위 증명에서 그냥 넘어갔던 위 명제를 증명해보자.

$$\begin{align}D_\Phi(x, x_{t+1})&=D_\Phi(x)-\Phi(x_{t+1})-\triangledown\Phi(x_{t+1})^T(x-x_{t+1})\\
D_\Phi(x_{t+1}, y_{t+1})&=\Phi(x_{t+1})-\Phi(y_{t+1})-\triangledown\Phi(x_{t+1})^T(x_{t+1}-y_{t+1})\end{align}$$

이다. 두 식을 더하면

$$D_\Phi(x, x_{t+1})+D_\Phi(x_{t+1}, y_{t+1})=\Phi(x)-\Phi(y_{t+1})-\triangledown\Phi(y_{t+1})^T(x-y_{t+1})-(\triangledown\Phi(x_{t+1})-\triangledown\Phi(y_{t+1}))^T(x-x_{t+1})$$

을 얻게 되는데, 마지막 항을 제외한 식은 $D_\Phi(x, y_{t+1})$이다. 마지막 항은 $-\triangledown_xD_\Phi(x_{t+1}, y_{t+1})^T(x-x_{t+1})$ 라고도 쓸 수 있는데, $x_{t+1}$은 정의에 따라 Bregman Divergence에서의 최적값이므로, 이 항은 $0$보다 무조건 작다. 따라서

$$D_\Phi(x, y_{t+1})\ge D_\Phi(x, x_{t+1})+D_\Phi(x_{t+1}, y_{t+1})$$

가 성립함을 알 수 있다.
