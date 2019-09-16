---
layout: post
title: Optimization Lecture 3
author: Yeonjee Jung
use_math: true
---

# Convex Optimization
## Gradient Descent
### Gradient Descent Algorithm

$$x_t \leftarrow x_t - \gamma\triangledown f(x_t)$$

이거나, $\gamma$를 $t$의 함수로 표현하여

$$x_t \leftarrow x_t - \gamma_t\triangledown f(x_t)$$

로 업데이트 하여 최적점을 찾아가는 방법을 gradient descent 알고리즘이라고 한다. 이 때 $\gamma_t$는 $\frac{1}{t}$, $\frac{1}{\sqrt{t}}$, $\frac{1}{\log{t}}$, $\cos{\theta t}$가 주로 쓰인다.

### Convergence Rate

Convergence rate은

$$f(x_t)-f(x^* )\le\epsilon$$

을 만족하는 $t$를 찾기까지의 시행 수를 Big O 표기법으로 나타내는 것이다.

## Strong Convex
### $\alpha$-Strong Convex

$$f(x)-f(y)\le\triangledown f(x)^T(x-y)-\frac{\alpha}{2}\|x-y\|^2, \forall x,y$$

를 만족할 때의 $\alpha$를 앞에 붙여 $\alpha$-strong convex라고 한다. 만약 $f$가 미분불가능하면 subgradient로 $\alpha$-strongly convexity를 정할 수 있다.

다르게는
$$f(y)\ge f(x)+\triangledown f(x)^T(y-x)+\frac{\alpha}{2}\|x-y\|^2$$

로 쓸 수 있다.

## Smooth
### $\beta$-Smooth
$\triangledown f$가 $\beta$-Lipschitz이고

$$\|\triangledown f(x)-\triangledown f(y)\|\le\beta\|x-y\|$$

를 만족할 때의 $\beta$를 앞에 붙여 $\beta$-smooth라고 한다.

$$f(x)-f(y) \ge \triangledown f(x)^T(x-y)-\frac{\beta}{2}\|x-y\|^2, \forall x,y$$

라고도 쓸 수 있고, 또는

$$f(y) \le f(x)+\triangledown f(x)^T(y-x)+\frac{\beta}{2}\|x-y\|^2$$

라고도 쓸 수 있다.

$\beta$-smooth 함수에서 gradient descent 알고리즘을 사용해 최적값을 찾을 때에는 $$\|\triangledown f(x)-\triangledown f(x^* )\|\le\beta \|x-x^* \|$$이고, $\triangledown f(x^* )=0$이기 때문에 learning rate가 $\gamma = \frac{1}{\beta}$인 것이 좋다. 하지만 보통의 convex optimization에서는 $\beta$를 모르기 때문에 learning rate를 정하는 것이 어렵다.

어떤 함수가 $\alpha$-strong이고 $\beta$-smooth이면 항상 $\alpha \le \beta$가 성립한다. 또한, convex optimization에서 $\alpha = \beta$일 때 가장 쉽게 해를 찾을 수 있다. 항상 $0 \le \frac{\alpha}{\beta} \le 1$ 이므로, $\frac{\alpha}{\beta}$가 1에 가까울수록 해를 찾기가 쉽다.

## Second-order Characterization of convexity

$f$가 두 번 미분가능하면 $\forall x\in\mathbf{dom}{(f)}$에서 항상 $\triangledown^2f(x)$가 존재한다. 그리고

$$f \text{ is convex }\Leftrightarrow \triangledown^2 f(x) \text{ is positive semidefinite }\forall x\in\mathbf{dom}{(f)}$$

가 항상 성립한다.

### (Def) Positive Definite

$A\in\mathbb{R}^{d\times d}$는

$$x^TAx \gt 0, \forall x\in \mathbb{R}^d$$

를 만족할 때 positive definite라고 한다.

$$x^TAx\ge 0 , \forall x\in \mathbb{R}^d$$

를 만족하면 positive semidefinite이라고 한다.

## Hessian, Smooth, Strong convexity

$$f \text{ 가 }\alpha\text{-strongly convex 이고 } \beta\text{-smooth 이다} \Leftrightarrow \alpha I \le \triangledown^2f(x)\le\beta I$$

를 항상 만족하는데, 행렬 $A, B$에 대하여 $A\le B$는 $B-A$가 positive semidefinite임을 뜻한다. 또한, 이는 $\triangledown^2 f(x)$의 eigenvalue들이 $\alpha$보다 크고 $\beta$보다 작음을 뜻한다.

hessian은 이러한 성질 때문에 step size를 정하기에 매우 중요한데, 이를 second-order method라고 한다. 하지만 계산량이 많고 저장할 숫자도 많으므로 딥러닝에서는 first-order method만 사용한다.

### (Proof) $\Rightarrow$

우선 $\alpha$-strongly convex이기 때문에

$$f(y)-f(x)\le\triangledown f(y)^T(y-x)-\frac{\alpha}{2}\|x-y\|^2$$, $$f(x)-f(y)\le\triangledown f(x)^T(x-y)-\frac{\alpha}{2}\|x-y\|^2$$

가 항상 성립한다. 마찬가지로 $\beta$-smooth이기 때문에

$$f(y)-f(x)\ge\triangledown f(y)^T(y-x)-\frac{\beta}{2}\|x-y\|^2$$, $$f(x)-f(y)\ge\triangledown f(x)^T(x-y)-\frac{\beta}{2}\|x-y\|^2$$

가 항상 성립한다.

$\alpha$에 대한 두 식을 합치면

$$0\le(\triangledown f(x)-\triangledown f(y))^T(x-y)-\alpha\|x-y\|^2\cdots(1)$$

를 얻게 되고, 마찬가지로 $\beta$에 대한 식을 합치면

$$0\ge(\triangledown f(x)-\triangledown f(y))^T(x-y)-\beta\|x-y\|^2\cdots(2)$$

를 얻게 된다.

$x=y+ht$룰 대입하면 ($h$는 벡터, $t$는 스칼라)

$$(\triangledown f(x)-\triangledown f(y))^T(x-y) = \frac{\triangledown f(y+ht)-\triangledown f(y)}{t}ht^2$$

를 얻을 수 있는데, 여기에 극한을 취하면

$$\lim_{t\rightarrow 0}{\frac{\triangledown f(y+ht)-\triangledown f(y)}{t}}ht^2=h^T\triangledown^2f(y)ht^2$$

을 얻을 수 있다.

이를 $(1)$과 $(2)$에 대입하면

$$\alpha t^2\|h\|^2\le h^T\triangledown^2f(y)t^2h\le \beta t^2\|h\|^2$$

를 얻게 되고,

$$h^T\alpha h\le h^T\triangledown^2 f(y)h\le h^T\beta h, \forall h$$

를 얻을 수 있다.

$$0\le h^T(\triangledown^2 f(y)-\alpha)h, 0\le h^T(\beta-\triangledown^2 f(y))h , \forall h$$

이므로,

$$\alpha I \le \triangledown^2 f(y)\le \beta I$$

와 동치이다.

### (Proof) $\Leftarrow$

Taylor Thm을 이용하면

$$f(y) = f(x) + \triangledown f(x)^T(y-x) + \frac{1}{2}(y-x)^T\triangledown^2f(c)(y-x), c=x+(y-x)t, \forall t\in[0, 1]$$

라고 할 수 있는데, 이 때

$$\frac{1}{2}(y-x)T\triangledown f(c)^T(y-x)\ge\frac{1}{2}k\|y-x\|^2$$

를 만족하는 $k$를 $\alpha$라고 하고,

$$\frac{1}{2}(y-x)T\triangledown f(c)^T(y-x)\le\frac{1}{2}k\|y-x\|^2$$

를 만족하는 $k$를 $\beta$라고 하면 $f$는 $\alpha$-strongly convex이고 $\beta$-smooth이다.

## Vanila Analysis
만약 $f$가 strongly convex도 아니고, smooth하지도 않지만 $L$-Lipschitz continuous하다고 했을 때, $f(x)-f(x^* )$를 gradient descent를 이용해 어떻게 bound하는지에 대해 알아보자. 여기서는 $$2a^Tb = \|a\|^2+\|b\|^2-\|a-b\|^2 \cdots(* )$$임을 이용할 것이다.

$f(x_t)-f(x^* )$

$\le\triangledown f(x_t)^T(x_t-x^* )$

$=\frac{1}{\gamma}(x_t-x_{t+1})^T(x_t-x^* )$ ($x_{t+1} = x_t-\gamma\triangledown f(x_t)$, gradient descent)

$$=\frac{1}{2\gamma}(\|x_t-x_{t+1}\|^2+\|x_t-x^* \|^2-\|x_{t+1}-x^* \|^2)$$ (위 $(* )$에 의해)

$$=\frac{\gamma}{2}\|\triangledown f(x_t)\|^2+\frac{1}{2\gamma}(\|x_t-x^* \| -\|x_{t+1}-x^* \|^2)$$ 이다.

이들을 모든 $t$에 대해 다 더하면,

$$\sum_{t=0}^{T-1}\left(f(x_t)-f(x^* )\right)\le\frac{1}{2\gamma}(\|x_0-x^* \|^2-\|x_T-x^* \|^2) + \sum_{t=0}^{T-1}\frac{\gamma}{2}\|\triangledown f(x_t)\|^2$$

가 된다. $f$가 Lipschitz continuous라고 했으므로 항상 $$\|\triangledown f(x_t)\|^2 \le L^2$$이다. $$\bar{x} = \frac{x_0+\cdots+x_{T-1}}{T}$$라고 하면, $$f(\bar{x})\le\frac{1}{T}\sum_{t=0}^{T-1}f(x_t)$$ 임이 성립하고,


$f(\bar{x})-f(x^* )$

$\le\frac{1}{T}\sum_{t=0}^{T-1}\left(f(x_t)-f(x^* )\right)$

$$\le \frac{1}{2\gamma T}\|x_0-x^* \|^2 + \frac{\gamma}{2}L^2$$ (Lipschitz continuous에 의해)

$=\frac{1}{2\gamma T}R^2 + \frac{\gamma}{2}L^2$ ($R = x_0-x^* $라고 하자)

가 된다. 따라서 여기에 적합한 learning rate 를 rough하게 계산해보면 $\gamma = \frac{R}{L\sqrt{T}}$가 적합하다. (마지막 식을 미분해서 0 되는 $\gamma$ 찾음)

이를 대입해 보면

$$f(\bar{x})-f(x^* )\le\frac{RL}{\sqrt{T}}$$

가 성립한다.

### $L$-Lipschitz Continuous

$$\|f(x)-f(y)\| \le L\|x-y\|, \forall x,y$$

를 만족할 때 $L$-Lipschitz continuous라고 하고, $f$가 미분가능하면 $$\|\triangledown f(x)\|\le L$$을 만족할 때 Lipschitz continuous라고 한다.

이 때 $\gamma = \frac{1}{L}$로 잡으면 diverge는 방지할 수 있지만 fast converge는 보장하지 못한다.
