---
layout: post
title: Optimization Lecture 6
author: Yeonjee Jung
use_math: true
---

# Projected Gradient Descent

## Projected Gradient : $\alpha$-strongly convex & $\beta$-smooth

### (Recall) Unconstrained Vanilla Analysis

vanilla analysis에서는

$$\begin{align}\|x_{t+1}-x^* \|^2&\le\frac{2}{\beta}(f(x^* )-f(x_t))+\frac{1}{\beta}\|\triangledown f(x_t)\|^2+(1-\frac{\alpha}{beta})\|x_t-x^* \|^2\\
&\le(1-\frac{\alpha}{\beta})\|x_t-x^* \|^2\end{align}$$

을 얻을 수 있었다. 이 때 마지막 부등식은

$$f(x^* )-f(x_t)\le f(x_{t+1})-f(x_t)\le-\frac{1}{2\beta}\|\triangledown f(x_t)\|^2\cdots(* 1)$$

라는 성질에 의해

$$\frac{2}{\beta}(f(x^* )-f(x_t))\le -\frac{1}{\beta^2}\|\triangledown f(x_t)\|^2$$

를 얻어서 성립한 것이다.

### Constrained Vanilla Analyasis

constrained에서는 $(* 1)$가 아닌

$$f(x^* )-f(x_t)\le f(x_{t+1})- f(x_t)\le-\frac{1}{2\beta}\|\triangledown f(x_t)\|^2+\frac{\beta}{2}\|y_{t+1}-x_{t+1}\|^2\cdots(* 2)$$

가 성립한다. 따라서,

$$\begin{align}\|x_{t+1}-x^* \|^2&\le\frac{2}{\beta}(f(x^* )-f(x_t))+\frac{1}{\beta^2}\|\triangledown f(x_t)\|^2-\|y_{t+1}-x_{t+1}\|^2+(1-\frac{\alpha}{\beta})\|x_t-x^* \|^2\\
&\le(1-\frac{\alpha}{\beta})\|x_t-x^* \|^2(\text{by }(* 2))\end{align}$$

가 성립하므로,

$$\begin{align}\|x_T-x^* \|^2&\le(1-\frac{\alpha}{\beta})^T\|x_0-x^* \|^2\\
\|x_T-x^* \|&\le(1-\frac{\alpha}{\beta})^{\frac{T}{2}}\|x_0-x^* \|\end{align}$$

이다. 따라서

$$\begin{align}f(x_T)-f(x^* )&\le\triangledown f(x^* )^T(x_T-x^* )+\frac{\beta}{2}\|x_T-x^* \|^2\\
&\le\|\triangledown f(x^* )\|\|x_T-x^* \|+\frac{\beta}{2}\|x_T-x^* \|^2\\
&\le\|\triangledown f(x^* )\|(1-\frac{\alpha}{\beta})^{\frac{T}{2}}\|x_0-x^* \|+\frac{\beta}{2}(1-\frac{\alpha}{\beta})^T\|x_0-x^* \|^2\end{align}$$

결론적으로, Projected Gradient Descent를 사용하면 Unconstrained에서와 거의 비슷하게 수렴하고 bound하게 된다. 하지만 Projection에 많은 계산량이 사용되기 때문에 이 방법은 거의 쓰지 않는다.
