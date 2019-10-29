---
layout: post
title: Optimization Lecture 9
author: Yeonjee Jung
use_math: true
---

# Subgradient

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
