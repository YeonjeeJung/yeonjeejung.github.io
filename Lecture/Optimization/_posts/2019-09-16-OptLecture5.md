---
layout: post
title: Optimization Lecture 5
author: Yeonjee Jung
use_math: true
---

# Projected Gradient Descent

## Constrained Optimization

Constrained Problem은 $f(x)$를 최소화하는 $x$를 찾는 문제인데, $x$의 범위 $X$가 주어져 있다는 점에서 이전과 다르다. 이 문제를 해결하는 방법에는 두 가지가 있는데,

1. Projected Gradient Descent를 이용하는 방법과
2. unconstrained problem으로 바꿔서 해결하는 방법이 있다.

이번 단원은 첫번째 방법에 대한 내용이다.

## Projected Gradient Descent

Project 는 $$\Pi_X(y):=\arg\min_{x\in X}\|x-y\|$$, 즉 $X$바깥에 있는 $y$와 가장 가까운 $x\in X$에 매칭시켜주는 것이다.

Projected gradient를 업데이트 하는 방법은 $$x_{t+1}=\Pi_X(x_t-\gamma\triangledown f(x_t))$$이다.

## (Prop)Convex Constrained Problem

$f$가 convex function이고 $X$가 convex set이고, $x^* $가 $X$범위에 대해 $f$의 minimizer라면,

$$\begin{align}f(x^* )\le f(y), \forall y\in X \Leftrightarrow \triangledown f(x^* )^T(x-x^* )\ge 0\cdots(* 1)\end{align}$$

이 성립한다.

### (Proof) $\Leftarrow$

$f$가 convex function이라고 했으므로,

$$f(x)\ge f(x^* ) + \triangledown f(x^* )^T(x-x^* )$$

를 항상 만족한다. 따라서 $\triangledown f(x^* )^T(x-x^* )\ge 0$이면 $f(x)\ge f(x^* )$이므로 증명은 끝난다.

### (Proof) $\Rightarrow$

대우명제를 이용한다. $\triangledown f(x^* )^T(x-x^* )\lt 0$이라고 하자. 그러면

$$\triangledown f(x^* )^T(x-x^* )=\lim_{t\rightarrow 0+}\frac{f(x^* +t(x-x^* ))-f(x^* )}{t(x-x^* )}(x-x^* )\lt 0$$

이 성립하므로,

$$f(x^* +t(x-x^* ))-f(x^* )\lt 0$$

이 성립하게 된다. 이는 $f(x^* )\le f(y), \forall y\in X$에 부합하지 않으므로, 증명은 끝난다.

## (Prop) Properties of Projection

$X$가 closed 이고 convex라고 하고, $x\in X, y\in \mathbb{R}^d$라고 하자. 그러면

$$\begin{align}&(x-\Pi_X(y))^T(y-\Pi_X(y))\le 0\cdots(* 2)\\
&\|x-\Pi_X(y)\|^2+\|y-\Pi_X(y)\|^2\le\|x-y\|^2\cdots(* 3)\end{align}$$

가 항상 성립한다.

### (Proof) $(* 2)$

$\Pi_X(y)$는 $$d(x)=\|x-y\|^2$$의 minimizer이므로, $(* 1)$의 $\Rightarrow$를 이용하면

$$\triangledown d(\Pi_X(y))^T(x-\Pi_X(y))\ge 0$$

이 항상 성립한다. $$d(x) = \|x-y\|^2=(x-y)^T(x-y)$$이므로, $$\triangledown d(x) = 2(x-y)$$이다. 따라서

$$(\Pi_X(y)-y)^T(x-\Pi_X(y))\ge 0$$

이고, 바꿔 말하면

$$(x-\Pi_X(y))^T(y-\Pi_X(y))\le 0$$

이다.


### (Proof) $(* 3)$

$v:=\Pi_X(y), w:=y-\Pi_X(y)$라고 하자. $(* 2)$에 의해

$$0\le 2v^Tw = \|v\|^2+\|w\|^2-\|v-w\|^2$$

이므로,

$$\|x-\Pi_X(y)\|^2+\|y-\Pi_X(y)\|^2\le\|x-y\|^2$$

가 성립한다.

## Projected Gradient : Lipschitz Convex

$f:\mathbb{R}^d\rightarrow\mathbb{R}$가 convex function이고 미분가능하다고 하자. $X\subset\mathbb{R}^d$는 closed 이고 convex라고 하고, $x^* \in X$는 $f$에 대한 minimizer라고 하자. $$\|x_0-x^* \|\le R, x_0\in X$$이고, $$\|\triangledown f(x)\|\le B, \forall x\in X$$라고 하자 (B-Lipschitz continuous). step size $\gamma = \frac{R}{B\sqrt{T}}$라고 지정하면,

$$f(\bar{x})-f(x^* )\le\frac{1}{T}\sum_{t=0}^{T-1}f(x_t)-f(x^* )\le\frac{RB}{\sqrt{T}}$$

의 boundary를 갖는다.

*Lipschitz Continuous에서는 $f(x_{t+1})\le f(x_t)$를 보장하지 못하기 때문에, $f(x_T)\le f(x_t)$라고 장담할 수가 없다. 따라서 $f(\bar{x})-f(x^ *)$를 bound시키는 것이다.*

### Vanilla Analysis

$$y_{t+1}=x_t-\gamma\triangledown f(x_t), x_{t+1}=\Pi_X(y_{t+1})$$이라고 하자. vanilla anaylsis를 이용하면,

$$\begin{align}f(x_t)-f(x^* ) & \le \triangledown f(x_t)^T(x_t-x^* )\\
&=\frac{1}{\gamma}(x_t-y_{t+1})^T(x_t-x^* )\\
&=\frac{1}{2\gamma}(\|x_t-y_{t+1}\|^2+\|x_t-x^* \|^2-\|y_{t+1}-x^* \|^2)\\
&\le\frac{\gamma}{2}\|\triangledown f(x_t)\|^2+\frac{1}{2\gamma}(\|x_t-x^* \|-\|x_{t+1}-x^* \|^2)-\frac{1}{2\gamma}\|y_{t+1}-x_{t+1}\|^2
\end{align}$$

(마지막 줄은 $$(y_{t+1}=x_t-\gamma\triangledown f(x_t))$$와 $(3)$을 사용했다). 이 식은 원래의 vanilla anaylsis에서 $-\frac{1}{2\gamma}\|y_{t+1}-x_{t+1}\|^2$만 추가된 것이다. 모든 $t$에 대해서 다 더하면,

$$\begin{align}\sum_{t=0}^{T-1}(f(x_t)-f(x^* ))&\le\frac{1}{2\gamma}(\|x_0-x^* \|^2-\|x_T-x^* \|^2)+\sum_{t=0}^{T-1}\frac{\gamma}{2}\|\triangledown f(x_t)\|^2-\sum_{t=0}^{T-1}\frac{1}{2\gamma}\|y_{t+1}-x_{t+1}\|^2 \\
&\le \frac{1}{2\gamma}(\|x_0-x^* \|^2-\|x_T-x^* \|^2)+\sum_{t=0}^{T-1}\frac{\gamma}{2}\|\triangledown f(x_t)\|^2\end{align}$$

를 얻게 되는데, 이는 결국 vanilla anaylsis와 같은 결론이다.

$$\bar{x}=\frac{x_0+\cdots+x_{T-1}}{T}$$ 라고 하면,

$$\begin{align}f(\bar{x})-f(x^* )&\le\frac{1}{T}\sum_{t=0}^{T-1}(f(x_t)-f(x^* ))\text{(Jensen's inequality)}\\
&\le\frac{1}{2\gamma T}R^2+\frac{1}{T}\sum_{t=0}^{T-1}\frac{\gamma}{2}\|\triangledown f(x_t)\|^2\\
&\le\frac{RB}{2\sqrt{T}}+\frac{RB}{2\sqrt{T}}(\gamma = \frac{R}{B\sqrt{T}})\\
&\le\frac{RB}{\sqrt{T}}\end{align}$$

## Projected Gradient : $\beta$-smooth functions

### (Recall) $\beta$-smooth

$$\begin{align}f(x_t)-f(x_{t+1})&\ge\triangledown f(x_t-x_{t+1})-\frac{\beta}{2}\|x_t-x_{t+1}\|^2\\
&=\gamma\|\triangledown f(x_t)\|^2-\frac{\gamma^2\beta}{2}\|\triangledown f(x_t)\|^2\end{align}$$

unconstrained 에서와는 다르게, constrained에서는

$$\gamma\triangledown f(x_t)\neq x_t-x_{t+1}$$

이다. 대신 $\gamma = \frac{1}{\beta}$를 대입하면

$$f(x_{t+1})\le f(x_t)-\frac{1}{2\beta}\|\triangledown f(x_t)\|^2 + \frac{\beta}{2}\|y_{t+1}-x_{t+1}\|^2\cdots(* 4)$$

를 얻을 수 있다. 주의할 점은, $$\frac{\beta}{2}\|y_{t+1}-x_{t+1}\|^2$$때문에 monotone decrese ($$f(x_{t+1}\le f(x_t))$$)를 확신할 수 없으므로 이를 먼저 증명해야 한다.

### (Proof) $$f(x_{t+1})\le f(x_t)$$

$$\begin{align}f(x_{t+1})&\le f(x_t)+\triangledown f(x_t)^T(x_{t+1}-x_t)+\frac{\beta}{2}\|x_{t+1}-x_t\|^2 (\beta\text{-smooth})\\
&=f(x_t)-\beta(y_{t+1}-x_t)^T(x_{t+1}-x_t)+\frac{\beta}{2}\|x_{t+1}-x_t\|^2(y_{t+1}-x_t=\frac{1}{\beta}\triangledown f(x_t))\\
&=f(x_t)-\frac{\beta}{2}(\|y_{t+1}-x_t\|^2+\|x_{t+1}-x_t\|^2-\|y_{t+1}-x_{t+1}\|^2)+\frac{\beta}{2}\|x_{t+1}-x_t\|^2\\
&=f(x_t)-\frac{\beta}{2}\|y_{t+1}-x_t\|^2+\frac{\beta}{2}\|y_{t+1}-x_{t+1}\|^2\end{align}$$

$(* 3)$에 의해

$$\|x_t-x_{t+1}\|^2+\|y_{t+1}-x_{t+1}\|^2\le\|x_t-y_{t+1}\|^2$$

이므로

$$-\frac{\beta}{2}\|y_{t+1}-x_t\|^2+\frac{\beta}{2}\|y_{t+1}-x_{t+1}\|^2\le 0$$

이 된다. 따라서

$$\begin{align}f(x_{t+1})-f(x_t)&\le-\frac{\beta}{2}\|y_{t+1}-x_t\|^2+\frac{\beta}{2}\|y_{t+1}-x_{t+1}\|^2\\&\le 0\end{align}$$

이므로, $f(x_{t+1})\le f(x_t)$를 얻을 수 있다.

### Vanilla Analysis

$$\begin{align}\sum_{t=0}^{T-1}(f(x_t)-f(x^* ))&\le\frac{\beta}{2}(\|x_0-x^* \|^2-\|x_T-x^* \|^2)+\sum_{t=0}^{T-1}\left(\frac{1}{2\beta}\|\triangledown f(x_t)\|^2-\frac{\beta}{2}\|y_{t+1}-x_{t+1}\|^2\right)\\
&\le\frac{\beta}{2}\|x_0-x^* \|^2+\sum_{t=0}^{T-1}(f(x_t)-f(x_{t+1}))((* 4)\text{에 의해})\end{align}$$

우변의 마지막 항을 좌변으로 넘기면

$$\sum_{t=1}^T(f(x_t)-f(x^* ))\le\frac{\beta}{2}\|x_0-x^* \|^2$$

를 얻게 되고,

$$\begin{align}f(x_T)-f(x^* )&\le\frac{1}{T}\sum_{t=1}^T(f(x_t)-f(x^* ))(f(x_T)\le f(x_t),\forall t\lt T)\\
&\le\frac{\beta}{2T}\|x_0-x^* \|^2\end{align}$$

로 bound된다.
