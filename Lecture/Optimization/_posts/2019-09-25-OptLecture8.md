---
layout: post
title: Optimization Lecture 8
author: Yeonjee Jung
use_math: true
---

# Proximal Gradient

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
