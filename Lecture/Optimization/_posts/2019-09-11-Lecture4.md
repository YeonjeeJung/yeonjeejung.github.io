---
layout: post
title: Optimization Lecture 4
author: Yeonjee Jung
use_math: true
---

# Convex Optimization

## $\beta$-smooth functions : $\frac{1}{t}$ learning Rate

$f$가 $\beta$-smooth하다고 하면

$$\begin{align}f(x_t)-f(x_{t+1}) & \ge \triangledown f(x_t)^T(x_t-x_{t+1})-\frac{\beta}{2}\|x_t-x_{t+1}\|^2\\
 & =\gamma\|\triangledown f(x_t)\|^2-\frac{\gamma^2\beta}{2}\|\triangledown f(x_t)\|^2 (x_{t+1}-x_t=-\gamma\triangledown f(x_t), \text{gradient descent})\\
 & =(\gamma-\frac{\gamma^2\beta}{2})\|\triangledown f(x_t)\|^2\end{align}
$$

이다. 이 식에서의 극점은 $\gamma = \frac{1}{\beta}$일 때이므로 이를 대입하면

$$\begin{align}f(x_t)-f(x_{t+1})\ge \frac{1}{2\beta}\|\triangledown f(x_t)\|^2\end{align}$$

를 얻게 된다. 이 때 $t=0$부터 $t=T-1$까지 다 대입한 후 다 더하면

$$\begin{align}f(x_0)-f(x_T) & \ge \sum_{t=0}^{T-1}\frac{1}{2\beta}\|\triangledown f(x_t)\|^2\end{align}$$

을 얻게 된다. 이전 단원의 Vanilla Analysis 중간에서 얻은 식에 $\gamma=\frac{1}{\beta}$를 대입하면

$$\sum_{t=0}^{T-1}{(f(x_t)-f(x^* ))}\le\frac{\beta}{2}\|x_0-x^* \|^2+\sum_{t=0}^{T-1}\frac{1}{2\beta}\|\triangledown f(x_t)\|^2$$

을 얻는데, $(4)$을 이용하면

$$\sum_{t=0}^{T-1}{(f(x_t)-f(x^* ))}\le\frac{\beta}{2}\|x_0-x^* \|^2+\sum_{t=0}^{T-1}\frac{1}{2\beta}\|\triangledown f(x_t)\|^2 \le \frac{\beta}{2}\|x_0-x^* \|^2+ f(x_0)-f(x_T)$$

이라고 할 수 있고, 맨 우측 두 항을 좌변으로 넘기면

$$\sum_{t=1}^T(f(x_t)-f(x^* ))\le\frac{\beta}{2}\|x_0-x^* \|^2$$

를 얻는다. $f(x_T)\le f(x_t), 0\le \forall t\le T$이므로, 최종적으로

$$f(x_T)-f(x^* )\le\frac{1}{T}\sum_{t=1}^T(f(x_t)-f(x^* ))\le\frac{\beta}{2T}\|x_0-x^* \|^2$$

를 얻는다.

*upper bound와 convergence speed에 관한 설명 추가*

### (Def) Linear Convergence

만약

$$\lim_{t\rightarrow\infty}{\frac{f(x_{t+1})-f(x^* )}{f(x_t)-f(x^* )}}=c$$

를 만족하는 $c\in(0, 1)$가 존재한다면 $f$는 선형적으로 수렴한다고 말할 수 있다. 만약 $\frac{f(x_{t+1})-f(x^* )}{f(x_t)-f(x^* )}\le c, \forall t, \forall c\in (0, 1)$라면

$$\begin{align}f(x_t)-f(x^* )\le c^t(f(x_0)-f(x^* ))\end{align}$$

가 항상 성립한다.

## $\alpha$-strongly convex and $\beta$-smooth function

Vanilla Analysis의 중간 과정에서

$$\triangledown f(x_t)^T(x_t-x^* )=\frac{\gamma}{2}\|\triangledown f(x_t)\|^2+\frac{1}{\gamma}(\|x_t-x^* \|^2-\|x_{t+1}-x^* \|^2)$$

의 식을 얻을 수 있었다. 그리고 $f$가 $\alpha$-strongly convex라고 하면

$$\triangledown f(x_t)^T(x_t-x^* )\ge f(x_t)-f(x^* )+\frac{\alpha}{2}\|x_t-x^* \|^2$$

를 만족한다. 둘을 합치고 변형하면,

$$\begin{align} f(x_t)-f(x^* )+\frac{\alpha}{2}\|x_t-x^* \|^2&\le\frac{\gamma}{2}\|\triangledown f(x_t)\|^2+\frac{1}{2\gamma}(\|x_t-x^* \|^2-\|x_{t+1}-x^* \|^2)\\
 -\frac{1}{2\gamma}(\|x_t-x^* \|^2-\|x_{t+1}-x^* \|^2)+\frac{\alpha}{2}\|x_t-x^* \|^2 &\le f(x^* )-f(x_t )+ \frac{\gamma}{2}\|\triangledown f(x_t)\|^2\\
 -\|x_t-x^* \|^2+\|x_{t+1}-x^* \|^2+\alpha\gamma\|x_t-x^* \|^2 &\le 2\gamma(f(x^* )-f(x_t ))+ \gamma^2\|\triangledown f(x_t)\|^2\\
\end{align}$$

을 얻을 수 있다. 이 때 $\gamma = \frac{1}{\beta}$를 대입하면 우변은

$$\begin{align}2\gamma(f(x^* )-f(x_t ))+ \gamma^2\|\triangledown f(x_t)\|^2&=\frac{2}{\beta}(f(x^* )-f(x_t))+\frac{1}{\beta^2}\|f(x_t)\|^2\\
&\le\frac{2}{\beta}(f(x_{t+1})-f(x_t))+\frac{1}{\beta^2}\|\triangledown f(x_t)\|^2\\
&\le-\frac{1}{\beta^2}\|\triangledown f(x_t)\|^2+\frac{1}{\beta^2}\|\triangledown f(x_t)\|^2\\
&=0\end{align}$$

($(11)\rightarrow(12)$은 $(4)$ 때문이다.) $(8)$을 정리하면

$$\|x_{t+1}-x^* \|^2\le \left(1-\frac{\alpha}{\beta}\right)\|x_t-x^* \|^2$$

이라고 할 수 있고, $(6)$에 의해

$$\|x_T-x^* \|^2\le\left(1-\frac{\alpha}{\beta}\right)^T\|x_0-x^* \|^2$$

가 성립한다.

## Convergence

$f$가 $\beta$-smooth라면

$$\begin{align}f(x_T)-f(x^* )&\le\triangledown f(x^* )^T(x_T-x^* )+\frac{\beta}{2}\|x_T-x^* \|^2\\
&=\frac{\beta}{2}\|x_T-x^* \|^2\end{align}$$

이므로,

$$f(x_T)-f(x^* )\le\frac{\beta}{2}\|x_T-x^* \|^2\le\frac{\beta}{2}\left(1-\frac{\alpha}{\beta}\right)^T\|x_0-x^* \|^2$$

가 성립한다. 이 때 $\frac{\alpha}{\beta}$를 condition number라고 하는데, 이 값이 $1$이면, 즉 $\alpha = \beta$이면 한 번의 이터레이션으로 최적값을 찾을 수 있다.
