---
layout: post
title: Optimization Lecture 2
author: Yeonjee Jung
use_math: true
---

# Convex Optimization

## Convex Optimization

Convex Optimization이란 $x^* = \text{min}_{x \in C} f(x)$를 찾는 문제이다. 이 때 $f$는 convex function이고, $C$는 convex set이다. 수렴도도 중요하지만, 얼마나 빨리 수렴하는지도 수렴 알고리즘의 선택에서 중요한 요소이다.

### Convergence Rate

convergence rate은 $f(x^* )$가 optimal value일 때,

$$\frac{f(x_t)-f(x^* )}{g(t)} \le c$$

에서의 $g(t)$이다. 이 때 $x_t$는 $x$가 $t$번 업데이트 된 값이고, $c$는 상수이다. 주로  
1. $g(t) = \frac{1}{t}$  
2. $g(t) = \frac{1}{\sqrt{t}}$  
3. $g(t) = e^{-t}$

가 쓰이는데, $g(t) = e^{-t}$가 가장 수렴속도가 빠르다.

$f(x_t)-f(x^* ) \le \epsilon$으로 만들 때,  
1. $g(t) = \frac{1}{t}$일 때는 $\frac{1}{\epsilon}$번의 step이 필요하고,  
2. $g(t) = \frac{1}{\sqrt{t}}$일 때는 $\frac{1}{\epsilon^2}$번의 step,  
3. $g(t) = e^{-t}$일 때는 $\ln{\frac{1}{\epsilon}}$번의 step이 필요하며, 가장 빠르다.

## Convex Functions

Convex Function의 예시에는 다음과 같은 함수들이 있다.  
1. Linear Functions : $f(x) = a^Tx$  
2. Affine Functions : $f(x) = a^Tx + b$  
3. Exponential Functions : $f(x) = e^{\alpha x}$  
4. Norms

### Norms

주어진 벡터공간 $V$에서의 norm은 0보다 크거나 같은 scalar function $p$이다.

$$p : V \rightarrow [0, +\infty)$$

norm은 다음과 같은 특징을 가진다.  
1. $p(x) + p(x) \ge p(x+y), \forall x, y\in V$  
2. $p(ax) = \| a\|p(x)$  
3. $p(x) = 0 \Leftrightarrow x=0$

### p-norm

$$\|x\|_ p = \left(\sum_{i=1}^d |x_i|^p\right)^{\frac{1}{p}}$$

주로 쓰이는 $p$값은 $1, 2, \infty$이다.

### Example of Norms
young's inequality : $ab\le\frac{a^p}{p}+\frac{b^q}{q}, \forall p, q \ge 1, \frac{1}{p}+\frac{1}{q}=1$  
Hoelder inequality : $\|\sum_{i=1}^n u_i\cdot v_i\| \le \left(\sum_{i=1}^n\|u_i\|^p\right)^{\frac{1}{p}}\left(\sum_{i=1}^n\|v_i\|^q\right)^{\frac{1}{q}}$  
Minkowski inequality : $\left(\sum_{i=1}^n\|x_i+y_i\|^p\right)^{\frac{1}{p}}\le\left(\sum_{i=1}^n\|x_i\|^p\right)^{\frac{1}{p}}+\left(\sum_{i=1}^n\|y_i\|^p\right)^{\frac{1}{p}}$

Hoelder's inequality를 증명하는 데에 young's inequality를 사용하고, Minkowski inequality를 증명하는 데에 Hoelder's inequality를 사용한다.

### (Proof) Young's inequality

### (Proof) Hoelder's inequality

### (Proof) Minkowski inequality

## (Proof) Every norm is Convex
(Recall) triangle inequality : $p(x) + p(y) \ge p(x+y), \forall x, y \in V$

$\alpha p(x) + (1-\alpha)p(y)$  
$= p(\alpha x) + p((1-\alpha)y)$  
$\ge p(\alpha x + (1-\alpha)y)$

## (Lemma) Jensen's inequality

$f$가 convex이고, $x_1, x_2, \cdots, x_m \in \text{dom}(f)$이고, $$\lambda_1, \lambda_2, \cdots, \lambda_m \in \mathbb{R}_+$$ s.t. $\sum_{i=1}^m \lambda_i = 1$일 때,

$$f\left(\sum_{i=1}^m \lambda_ix_i\right) \le \sum_{i=1}^m \lambda_i f(x_i)$$

$x$가 random variable이라면

$$f(E(x)) \le E[f(x)]$$

이 부등식은 $\lambda_1 + \lambda_2 =1$이고 $\lambda_1, \lambda_2 \ge 0$일 때

$$\lambda_1f(x_1) +\lambda_2f(x_2) \ge f(\lambda_1x_1+\lambda_2x_2)$$

라는 convex의 정의를 이용해, 수학적 귀납법으로 증명할 수 있다.

### (Proof)
m일 때 성립한다고 가정한다. 그러면  
$f(\sum_{i=1}^{m+1}\lambda_ix_i)$  
$=f(\sum_{i=1}^m\lambda_ix_i + \lambda_{m+1}x_{m+1})$  
$=f((1-\lambda_{m+1})\sum_{i=1}^m\frac{\lambda_i}{1-\lambda_{m+1}}x_i+\lambda_{m+1}x_{m+1})$  
$\le(1-\lambda_{m+1})f(\sum_{i=1}^m\frac{\lambda_i}{1-\lambda_{m+1}}x_i)+\lambda_{m+1}f(x_{m+1})$  
$\le(1-\lambda_{m+1})\sum_{i=1}^m\frac{\lambda_i}{1-\lambda_{m+1}}f(x_i)+\lambda_{m+1}f(x_{m+1})$  
$=\sum_{i=1}^{m+1}\lambda_if(x_i)$

따라서,

$$f\left(\sum_{i=1}^{m+1}\lambda_ix_i\right)\le\sum_{i=1}^{m+1}\lambda_if(x_i)$$

가 성립한다.

## Differentiable Functions

### Differentiable
$f$가 continuous한 일차미분식을 갖는다면, $f$는 differentiable하다. 이 특징은 매우 중요한데, 대부분의 최적화 iteration은 gradient값을 사용하기 때문이다.

### Tangent Hyperplane

$$g(x) = f(x_0) + \triangledown f(x_0)^T(x-x_0)$$

이 tangent hyperplane이다. 이 직선은 $(x, f(x)$를 지나고 $\triangledown f(x_0)$의 기울기를 갖는다.

## First-order Characterization of Convexity

$$\triangledown f(x) = \left(\frac{\partial f(x)}{\partial x_1}, \cdots, \frac{\partial f(x)}{\partial x_d}\right)^T$$

가 first-order derivative이다.

$$f : \text{ convex function} \Leftrightarrow f(y) \ge f(x) + \triangledown f(x)^T (y-x), \forall x, y \in \text{dom}(f)$$

### (Proof) $\Leftarrow$

Let $z = \theta x + (1-\theta) y, \theta\in [0, 1]$  
$(f(x) \ge f(z) + \triangledown f(z)^T(x-z)) \times\theta$  
$(f(y) \ge f(z) + \triangledown f(z)^T(y-z))\times(1-\theta)$  
$\Rightarrow \theta f(x) + (1-\theta)f(y)\ge f(z) + \triangledown f(z)^T(\theta x+(1-\theta)y-z)$  
$\therefore\theta f(x)+(1-\theta)f(y)\ge f(\theta x+(1-\theta)y)$

### (Proof) $\Rightarrow$

$(1-t)f(x) + tf(y)\ge f(x+(y-x)t)$  
$\Rightarrow f(y)\ge f(x) + \frac{f(x+(y-x)t)-f(x)}{t}$  
as $t\rightarrow 0, f(y)\ge f(x)+\triangledown f(x)^T(y-x)$

## Nondifferentiable Functions
$f(x) = \|x\|$같은 함수의 경우, $x=0$부근에서 많은 접점이 있지만 정확한 극한값은 없다. 이 때 그냥 하나를 마음대로 정할 수 있는데, 이것이 subgradient이다.

$$f(y)\ge f(x) + g_x^T(y-x)$$

에서 $g_x$가 subgradient. 위에서 예를 든 $f(x) = \|x\|$의 경우, $g\in[-1,1]$은 모두 subgradient가 될 수 있다.

## Second-order Characterization of Convexity

일변수함수에 대해서는 이차미분식이 그냥 하나의 식이지만, 다변수함수에서는 이차미분식이 Hessian이다. Hessian은

$$\triangledown^2f(x) = \left(\begin{matrix}
  \frac{\partial^2f}{\partial x_1 \partial x_1} & \frac{\partial^2f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2f}{\partial x_1 \partial x_d} \\
  \frac{\partial^2f}{\partial x_2 \partial x_1} & \frac{\partial^2f}{\partial x_2 \partial x_2} & \cdots & \frac{\partial^2f}{\partial x_2 \partial x_d} \\
  \vdots & \vdots & \cdots & \vdots \\
  \frac{\partial^2f}{\partial x_d \partial x_1} & \frac{\partial^2f}{\partial x_d \partial x_2} & \cdots & \frac{\partial^2f}{\partial x_d \partial x_d}
  \end{matrix}\right)$$

으로 정의된다.

### (Def) Positive Definite
$$A\in \mathbb{R}^{d\times d}$$가 symmetric하고 $x^TAx \gt 0, \forall x\in\mathbb{R}^d$일 때 $f$는 positive definite이다.

### (Proof) $f : $ convex function $\Leftrightarrow \triangledown^2f(x)$ is positive semidefinite
$g(t) = f(x+(y-x)t)$는 $t$에 대한 convex function이다.  
$\Leftrightarrow g'(t) = \triangledown f(x+(y-x)t)(y-x)$  
$$\Leftrightarrow g''(t) = (y-x)^T\triangledown^2f(x+(y-x)t)(y-x)$$  
$g(t)$가 convex function이므로, $$g''(t) \ge 0, \forall t, \forall x, y$$를 만족한다.  
따라서 $t=0$일 때도 만족하는데, 이를 대입해보면 다음과 같다.  
$$g''(t)\mid_ {t=0} = (y-x)^T\triangledown^2f(x)(y-x) \ge 0, \forall x, y$$  
이는 positive semidefinite의 정의이므로, 둘은 동치이다.

## Local Minima, Critical Point
### (Def) Local Minima
$\epsilon \gt0$에 대해, $f(x)\le f(y), \forall y \text{ s.t. }\|y-x\|\le\epsilon$을 만족하면 $x$는 local minima이다.

### (Def) Critical Point
$\triangledown f(x)=0$을 만족하면 $x$는 critical point이다. 만약 $f$가 convex라면, critical point는 global minima이다. critical point는 최적화에서는 bad point인데, tangent line이 기울기가 0이기 때문에 gradient가 존재하지 않아 local minima를 빠져나가기가 어렵다.

## Constrained Minimization
$f$가 convex function이고, $X$가 convex set이면

$$f(x^* )\le f(y), \forall y\in X \Leftrightarrow \triangledown f(x^* )^T(x-x^* ) \ge 0, \forall x\in X$$

### (Proof)$\Rightarrow$
Let $f(x^* )\le f(y), \forall y \in X$  
Since $f$ is convex, $tf(x^* )+(1-t)f(y) \ge f(y+(x^* -y)t)$  
$\Rightarrow f(x^* ) \ge \frac{f(y+(x^* -y)t)-f(y)}{t} + f(y)$  
as $t\rightarrow0$,  
$=(x^* -y)\triangledown f(y)+f(y)$  
$\Rightarrow (y-x^* )\triangledown f(y)\ge f(y)-f(x^* ) \ge 0$  
$\therefore (y-x^* )\triangledown f(y) \ge 0, \forall y\in X$

### (Proof) $\Leftarrow$
Let $\triangledown f(x^* )^T(y-x^* )\ge 0, \forall y\in X$  
$f$가 convex이기 때문에, $f(y)\ge f(x^* )+\triangledown f(x^* )^T(y-x^* )$이다.  
$\Rightarrow f(y)-f(x^* )\ge\triangledown f(x^* )^T(y-x^* )\ge 0$  
$\therefore f(y)\ge f(x^* ), \forall y\in X$

## (Thm) Tayler

$f(x) = f(0) + f'(0)x + \frac{1}{2}f^{\prime\prime}(0)x^2+\cdots$  
$f(b) = f(a) + f'(a)(b-a) + \frac{1}{2}f^{\prime\prime}(c)(b-a)^2, c\in[a, b]$  
$ = f(a) + \triangledown f(a)^T(b-a) + \frac{1}{2}(b-a)^T\triangledown^2f(c)(b-a), c\in[a, a+\theta(b-a)], \theta\in[0, 1]$  
$\therefore f(b) \ge f(a)+\triangledown f(a)^T(b-a)$
