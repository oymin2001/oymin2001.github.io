---
layout: single
title: "추정량의 비교 2"
description: "모수의 함수에 대한 추정량의 비교2"
categories: Statistics
tag: [Sufficiency, Complete, UMVUE, Rao-Blackwell, Lehmann-Scheffé, Cramer-Rao Inequality ]
typora-root-url: ..\
author_profile: false
use_math: true
toc: true
---

이전 포스팅에 이어서 다양한 확률분포에 대한 UMVUE의 유도와 추정량의 비교를 직접 구해볼 예정이다.



# 균등 분포

## 최소값과 최댓값이 대칭인 경우

균등분포 $$U[-\theta, \theta]$$에서의 랜덤표본 $X_1,...,X_n$의 결합확률밀도함수는 다음과 같다.



$$ \begin{align*} f(x;\theta) &= (2\theta)^{-n}I[-\theta \leq x_{(1)}]I[x_{(n)} \leq \theta ] \text{ where } x_{(1)} \leq ... \leq x_{(n)} \\ &= (2\theta)^{-n}I[\max(-x_{(1)},x_{(n)}) \leq \theta] \end{align*} $$



분해 정리에 의해 $(X_{(1)}, X_{(n)}), \max(-X_{(1)}, X_{(n)})$ 모두 $\theta$에 대한 충분통계량이지만, $(X_{(1)}, X_{(n)})$의 경우 $X_{(1)} \overset{d}{=} -X_{(n)}$임으로, 기댓값의 선형성에 의해  $$\mathbb{E}_{\theta}[X_{(1)} + X_{(n)}]= -\mathbb{E}_{\theta}[-X_{(1)}] + \mathbb{E}[X_{(n)}]=0$$이기에 완비통계량은 아니다.

한편  $\max(-X_{(1)}, X_{(n)})$의 경우 $Y=\max_{i}\|X\|_i$로 다시 쓸 수 있고, $Pr(\|X_1\|  \leq x) = (x/\theta)I[0\leq x \leq \theta]$이기에 결합확률밀도함수는 다음과 같다.

$$ f_Y(y) = n\theta^{-n}y^{n-1}I(0\leq y \leq \theta) $$

즉, 고정된 $\theta$에 대해 $$\mathbb{E}_{\theta}[g(Y)] = n\theta_{-n}\int_{0}^{\theta}g(y)y^{n-1}dy=0$$을 만족하기 위해서는 다음을 만족해야 한다.

$$ \int_{0}^{\theta}g(y)y^{n-1}dy = 0, \  \forall \theta>0 $$

이제 피적분함수 $g(y)y^{n-1}$가 연속, 즉 연속함수인 $g$에 대해 미적분학의 기본정리를 사용하여, 다음의 등식을 만족해야함을 알 수 있다.

$$ g(\theta)\theta^{n-1} = 0, \ \forall \theta>0 $$

따라서, $g$는 모수공간에서 모두 $0$을 만족해야 함을 의미함으로 $Pr_{\theta}(g(Y)=0)=1$이 성립한다. 즉,   $\max(-X_{(1)}, X_{(n)})$는 완비충분통계량이다.

__Note__. 충분히 매끄러운 $g$에 대해서만 성립합을 보였지만, 일반적인 완비성을 보이기 위해서는 이보다 약한 가정인 측정가능성, 국소적 적분가능성을 통해 $g=0  \ a.e.$를 보여야 한다.

### 불편 추정량을 통한 유도

이제 관측된 통계량의 범위 $R_n=X_{(n)} - X_{(1)}$에 대한 기댓값은 다음과 같다.

$$ \begin{align*} \mathbb{E}[R_n] &= \int_{-\theta}^{\theta}\int_{y_1}^{\theta}(y_2-y_1) \cdot n(n-1)(2\theta)^{-2n}(y_2-y_1)^{n-2}dy_2dy_1 \\ &= \int_{-\theta}^{\theta}(n-1)(2\theta)^{-2n}(\theta-y_1)^{n}dy_1\\ &=\frac{(n-1)(2\theta)^{n+1}}{(2\theta)^n(n+1)} = \frac{2(n-1)}{n+1} \cdot \theta

\end{align*} $$

즉, $c_n = \frac{n+1}{2(n-1)}$로 정의하여 $\hat{\theta}=c_nR_n$라고 하면, $\hat{\theta}$는 $\theta$의 불편추정량이다. 따라서,   $Y=\max(-X_{(1)}, X_{(n)})$를 조건부로한 $\hat{\theta}$의 기댓값은 레만-셰페 정리에의해 UMVUE이다.

$$ \begin{align*} \mathbb{E}[\hat{\theta} \mid Y=y] &= c_n\mathbb{E}[X_{(n)}-X_{(1)} \mid \max(X_{(n)}, -X_{(1)})=y] \\ &= c_n[\mathbb{E}[X_{(n)} \mid \max(X_{(n)}, -X_{(1)})=y] -  \mathbb{E}[X_{(1)} \mid \max(X_{(n)}, -X_{(1)})=y]] \\ &= 2c_n  \mathbb{E}[X_{(n)} \mid \max(X_{(n)}, -X_{(1)})=y]

\end{align*} $$

또한 조건부 기댓값의 성질에 의해 다음이 성립한다.

$$ \begin{align*} \mathbb{E}[X_{(n)} \mid \max(X_{(n)}, -X_{(1)})=y] &= \mathbb{E}[X_{(n)} \mid X_{(n)}=y,X_{(1)} \geq -y]\cdot Pr[X_{(n)}=y,X_{(1)} \geq -y \mid \max(X_{(n)}, -X_{(1)})=y] + \mathbb{E}[X_{(n)} \mid X_{(1)}=-y,X_{(n)} \leq y]\cdot Pr[X_{(1)}=-y,X_{(n)} \leq y \mid \max(X_{(n)}, - X_{(1)})=y] \\ &= \frac{1}{2}[y+\mathbb{E}[X_{(n)} \mid X_{(1)}=-y, X_{(n)}\leq y]]

\end{align*} $$

한편 $\mathbb{E}[X_{(n)} \mid X_{(1)}=-y, X_{(n)}\leq y]$는 최솟값은 고정되어 있고, 나머지는 $[-y,y]$상에서의 서로 독립이고 최솟값과도 독립인 균등분포를 조건부로 하기에, $U[-y,y]$로부터의 $(n-1)$개의 랜덤샘플의 최댓값에 대한 기댓값임을 알 수 있다.

$$ \mathbb{E}[X_{(n)} \mid X_{(1)}=-y, X_{(n)}\leq y]= \int_{-y}^y x\cdot \frac{n-1}{(2y)^{n-1}}(x+y)^{n-2}dx = \frac{n-2}{n}y $$

그러므로, UMVUE는 다음과 같다.

$$ \begin{align*} \mathbb{E}[\hat{\theta} \mid Y=y] &=  2c_n\cdot \frac{1}{2}(y+\frac{n-2}{n}y) \\ &= \frac{n+1}{2(n-1)}\cdot \frac{2n-2}{n}y \\ &= \frac{n+1}{n}y

\end{align*} $$

### 완비추정량에 대한 함수로부터의 유도

$\mathbb{E}_{\theta}[g(Y)] = \theta$가 되는 $g$를 찾으면 충분하다. 즉, 다음을 만족하는 $g(y)$를 구한다.

$$ \begin{align*} \int_{0}^{\theta}g(y) n\theta^{-n}y^{n-1}dy  &= \theta \\ \int_{0}^{\theta}g(y)y^{n-1}dy = \frac{\theta^{n+1}}{n} \\

\end{align*} $$

이제 양변을 $\theta$에 대해 미분하고, 미적분학의 기본정리를 사용하면 다음의 등식이 성립해야한다.

$$ \begin{align*} g(\theta)\theta^{n-1} &= \frac{n+1}{n}\theta^{n} \\ g(\theta)&= \frac{n+1}{n}\theta \end{align*} $$

따라서 UMVUE는 다음과 같다.

$$ \hat{\theta}^{\text{UMVUE}} = \frac{n+1}{n}Y = \frac{n+1}{n}\max(-X_{(1)},X_{(n)}) $$

## 비대칭인 경우

$$U[-\theta, 2\theta+1]$$로부터의 랜덤샘플 $X_1,...,X_n$을 가정하자. 결합확률밀도함수는 다음과 같다.

$$ f(x;\theta) = (3\theta+1)^{-n}I(-\theta<x_{(1)}, \frac{x_{(n)}-1}{2} \leq \theta) $$

즉, $Y=\max(-X_{(1)}, (X_{(n)}-1)/2)$는 충분통계량이고, $(X_i+\theta)/(3\theta+1) \overset{d}{=}U$ 가 성립함으로 $Y$의 결합확률밀도는 다음과 같이 쓸 수 있다.

$$ \begin{align*} Pr(Y\leq y) &=  Pr(-y\leq X_i \leq 2y+1)^nI(y\leq \theta, -y \leq 2y+1) \\ &= (\frac{3y+1}{3\theta+1})^nI(-\frac{1}{3} \leq y \leq \theta) \\ f_{Y}(y) &= \frac{3n}{(3\theta+1)^n}(3y+1)^{n-1}I(-\frac{1}{3} \leq y \leq \theta) \end{align*} $$

따라서, 대칭인 경우와 비슷하게 완비충분통계량임을 유도할 수 있고, UMVUE는 다음의 관계식을 만족한는 $g(y)$를 구하여 유도할 수 있다.


$$
\begin{align} \int_{-1/3}^{\theta}g(y)\frac{3n}{(3\theta+1)^n}(3y+1)^{n-1}  &= \theta \\ \int_{-1/3}^{\theta}g(y)(3y+1)^{n-1} &=  \frac{\theta(3\theta+1)^n}{3n}

\end{align}
$$




마찬가지로 양변을 $\theta$에 대해 미분하고미적분학의 기본정리를 사용하여 다음과 같이 나타낼 수 있다.


$$
\begin{align}g(\theta)(3\theta+1)^{n-1} &= \frac{(3\theta+1)^n + 3n\theta (3\theta+1)^{n-1}}{3n} \\ g(\theta)&=\frac{(3\theta+1)+ 3n\theta }{3n} = \frac{3(n+1)\theta+1}{3n} = \frac{n+1}{n}\theta + \frac{1}{3n} \end{align} 
$$




따라서 UMVUE는 다음과 같다.

$$ \hat{\theta}^{\text{UMVUE}} = \frac{n+1}{n}Y+ \frac{1}{3n} = \frac{(n+1)}{n}\max(-X_{(1)}, \frac{X_{(n)}-1}{2}) + \frac{1}{3n} $$

## 이산 균등분포인 경우

모수공간은 $\Omega = \mathbb{N}$이고, $n$개의 랜덤표본은 다음의 확률밀도함수를 따른다.

$$ f(x_i;\theta) = \frac{1}{\theta}, x_i \in\{1,...,\theta\}, \ i=1,...,n $$

이러한 확률분포를 이산 균등분포 $U\{1,...,\theta\}$라고 한다.

먼저 $X_1,...,X_n$의 결합확률밀도함수는 다음과 같다.

$$ f(x;\theta) = \prod_{i=1}^nf(x_i;\theta)  = \theta^{-n}I[x_{(n)}\leq \theta] \text{ where } x_{(1)} \leq ... \leq  x_{(n)} $$

따라서, $Y=X_{(n)}$은 $\theta$에 대한 충분통계량이고, 확률밀도함수는 다음과 같다.

$$

\begin{align*} Pr(Y\leq y) &= Pr(X_i\leq y)^n \\ &= \frac{y^n}{\theta^n}I[y\in\{1,...,\theta\}] \\ f_Y(y) &= Pr(Y\leq y) - Pr(Y\leq y-1) \\ &= \frac{y^n-(y-1)^n}{\theta^n}\cdot I[y \in \{1,...\theta\}] \end{align*} $$

즉, 고정된 $\theta$에 대해 $\mathbb{E}_{\theta}[g(Y)]=0$을 만족하는 $g$는 다음을 만족한다.

$$ \begin{align*} \mathbb{E}_{\theta}[g(Y)] &= \sum_{y=1}^{\theta}g(y) \cdot \frac{y^n-(y-1)^n}{\theta^n} \\ &= \theta^{-n}[(g(1)-g(2))1^n + (g(2)-g(3))2^n +... + (g(\theta-1)-g(\theta))\theta^{n-1} + g(\theta)\theta^n] \\ &=0

\end{align*} $$

즉, $0=g(\theta)=g(\theta-1)=...=g(1)$을 만족해야함으로, $Y$는 완비충분통계량이다.

이제 $Y$에 대한 함수로 UMVUE를 유도하면 다음과 같다.


$$
\begin{align} \mathbb{E}_{\theta}[g(Y)] &= \sum_{y=1}^{\theta}g(y) \cdot \frac{y^n-(y-1)^n}{\theta^n}=\theta \end{align} 
$$




한편 $\sum_{y=1}^{\theta}[y^{n+1}-(y-1)^{n+1}] = \theta^{n+1}$이다. 따라서, UMVUE는 다음과 같다.


$$
\hat{\theta}^{\text{UMVUE}} = \frac{Y^{n+1}-(Y-1)^{n+1}}{Y^n-(Y-1)^n} = \frac{X_{(n)}^{n+1}-(X_{(n)}-1)^{n+1}}{X_{(n)}^n-(X_{(n)}-1)^n}
$$




# 정규 분포

$\mathcal{N}(\mu,\sigma^2)$을 따르는 랜덤샘플 $X_1,...,X_n (n\geq 2)$을 가정하자. 이는 지수족이므로 $(\bar{X},S^2)$이 $(\mu,\sigma^2)$에 대한 완비충분통계량이다. 이제 모수에 대한 함수 $\eta=\eta(\mu,\sigma^2)$에 대한 UMVUE를 구해보려한다.

## $\eta = \mu/\sigma$

완비충분통계량에 대한 함수 $\bar{X}/\sqrt{S^2}$에 적절한 상수 $c_n$을 곱하여 충분성을 보여 UMVUE를 유도하려한다.

$$ \mathbb{E}[c_n \frac{\bar{X}}{\sqrt{S^2}}] = c_n\mathbb{E}[\bar{X}]\cdot \mathbb{E}[\frac{1}{\sqrt{S^2}}] = c_n \mu \cdot  \mathbb{E}[\frac{1}{\sqrt{S^2}}]  =  \frac{\mu}{\sigma} $$

한편 $Y=(n-1)S^2/\sigma^2 \sim \chi^2(n-1)$를 이용하여 다음을 보일 수 있다.

$$ \begin{align*} \mathbb{E}[\frac{1}{\sqrt{S^2}}] &= \mathbb{E}[\sqrt{\frac{n-1}{\sigma^2Y} }] \\ &= \frac{\sqrt{n-1}}{\sigma}\int_{0}^{\infty}y^{-1/2}\frac{1}{\Gamma(\frac{n-1}{2})2^{\frac{n-1}{2}}}y^{\frac{n-1}{2}-1}e^{-y/2}dy \\ &= \frac{\sqrt{n-1}}{\sigma} \cdot \frac{\Gamma(\frac{n-2}{2})2^{\frac{n-2}{2}}}{\Gamma(\frac{n-1}{2})2^{\frac{n-1}{2}}}=\frac{1}{\sigma}\sqrt{\frac{n-1}{2}}\frac{\Gamma(\frac{n-2}{2})}{\Gamma(\frac{n-1}{2})} \end{align*} $$

따라서, UMVUE는 다음과 같다.

$$ \hat{\eta}^{\text{UMVUE}} = \sqrt{\frac{2}{n-1}}\frac{\Gamma(\frac{n-1}{2})}{\Gamma(\frac{n-2}{2})} \cdot \frac{\bar{X}}{S} $$

## $\eta = Pr_{\theta}(X_1 \geq r)$

$Pr_{\theta}(X_1 \geq r) = \mathbb{E}[I(X_1 \geq r)]$임으로 $\hat{\eta}^{\text{UB}} = I(X_1 \geq r)$은 불편추정량이다. $(\bar{X},S^2)$에 대한 조건부 기댓값은 다음과 같이 구할 수 있다.


$$
\begin{align*}
\mathbb{E}[ I(X_1>r|\bar{X}=y_1,S^2=y_2) &= Pr[\frac{X_1-\bar{X}}{S} > \frac{r-y_1}{\sqrt{y_2}}|y_1,y_2] \\
&= 1- Pr[\frac{X_1-\bar{X}}{S} \leq \frac{r-y_1}{\sqrt{y_2}}|y_1,y_2] \\


\end{align*}
$$






이제 $$W = (X_1-\bar{X})/S$$를 이용하여 모수 $\theta$에 의존하지 않는 적절한 보조통계량을 찾아야한다. $S$는 $$(X_1-\bar{X},...,X_n-\bar{X})$$에 대한 함수임으로 $$X_1-\bar{X}$$와 독립이 아니기에, t-분포를 사용할 수 없다. 따라서, $W$를 제곱하여 서로 독립인 카이제곱분포 $V_1,V_2$에 대해 $c_n\frac{V_1}{V_1+V_2}$의 베타분포로부터의 유도가 필요하다.

$$ W^2 = (n-1)\frac{(X_1-\bar{X})^2}{(n-1)S^2} $$

$j$를 1-벡터, $e_1$은 첫번째 원소가 1이고 나머지는 0인 벡터라고 하면, 다음이 성립한다.

$$ \begin{align*} X_1 -\bar{X} &= e_1^TX - (j^Tj)^{-1}j^TX = (e_1-(j^Tj)^{-1}j)^TX\\ X - \bar{X}j &= X - j(j^Tj)^{-1}j^TX =(I-j(j^Tj)^{-1}j^T)X \end{align*} $$

이제 $a = e_1 - (j^Tj)^{-1}j, H_1 = j(j^Tj)^{-1}j^T$라고 하자.

$$ \begin{align*} (X_1 -\bar{X})^2 &= X^T(aa^T)X = (a^Ta)X^TH_aX \text{ where } H_a = a(a^Ta)^{-1}a^T \\ (n-1)S^2 &= X^T(I-H_1)X = X^T\{(I-H_1-H_a )+H_a\}X \end{align*} $$

이제 $(I-H_1-H_a)$와 $H_a$가 직교함을 보이면 $(n-1)S^2$을 서로 독립인 카이제곱 분포의 합으로 나타낼 수 있다. 한편, $a^Tj=0$이기에 $span(a) \subset span(j)^{\perp}$, 즉 $(I-H_1)H_a = H_a$임을 알 수 있다. 마찬가지로 다음이 성립한다.

$$ \begin{align*} \frac{(X-\mu j)^TH_a(X -\mu j)}{\sigma^2} &= \frac{X^TH_aX}{\sigma^2} \sim \chi^2(tr(H_a)) \\ \frac{X^T(I-H_1-H_a)X}{\sigma^2} &\sim \chi^2(tr(I-H_1-H_a)) \end{align*} $$

$tr(H_1) = tr((j^Tj)^{-1}j^Tj) = 1, tr(H_a) = tr((a^Ta)^{-1}a^Ta) = 1$임으로 서로 독립인 $V_1 \sim \chi^2(1), V_2 \sim \chi^2(n-2)$를 통해 다음과 같이 $W^2$을 나타낼 수 있다.

$$ W^2 = (n-1)\frac{a^TaV_1}{V_1 + V_2} \overset{d}{=} \frac{(n-1)^2}{n}B \text{ where } B \sim \text{Beta}(\frac{1}{2}, \frac{n-2}{2}) $$

따라서 $B$의 누적밀도함수를 $\nu$라고 나타내면 $W^2$의 누적밀도함수는 다음과 같다.

$$ Pr[W^2 \leq t] = Pr[\frac{(n-1)^2}{n}B \leq t] = \nu(\frac{nt}{(n-1)^2}) $$

이제 $W^2$와 $W$의 누적밀도함수의 관계를 확인하자.

$W \overset{d}{=} -W$이기에 $W^2$과는 다음의 관계가 성립한다.

$$ \begin{align*} Pr[W^2 \leq t] &= Pr[-\sqrt{t} \leq W  \leq \sqrt{t}] \\ &= 2Pr[W \leq \sqrt{t}] - 1 \\ Pr[W \leq w] &= \frac{1+Pr[W^2\leq w^2]}{2} \text { if } w\geq0 \\ Pr[W \leq w] &= 1- Pr[W \leq -w] \\ &= \frac{1-Pr[W^2\leq (-w)^2]}{2} \text { if } w<0

\end{align*} $$

따라서, 다음이 성립한다.

$$ \begin{align*} Pr[W > w] &= 1-\frac{1+sgn(w)Pr[W^2 \leq w^2]}{2} \\ &=\frac{1-sgn(w)\nu(\frac{nw^2}{(n-1)^2})}{2}

\end{align*} $$

이제 $w=(r-y_1)/\sqrt{y_2}$를 대입하여 다음과 같이 UMVUE를 구할 수 있다.

$$ \hat{\eta}^{\text{UMVUE}} = \frac{1}{2}[1-sgn(r-\bar{X})\cdot \nu(\frac{n(r-\bar{X})^2}{(n-1)^2S^2})] $$





## 추정량의 비교

표본표준편차의 불편추정량은 $\hat{\sigma}^{\text{UB}} = S_n$이며, UMVUE는 다음의 관계식을 통해 구할 수 있다.

$$ \begin{align*} \mathbb{E}[S_n|\bar{X},S_n^2] &= \int_{0}^{\infty} x^{1/2}\frac{1}{\Gamma(\frac{n-1}{2})2^{(n-1)/2}}x^{\frac{n-1}{2}-1}e^{-x/2}dx \cdot \frac{\sigma}{\sqrt{n-1}} \ (\because \frac{(n-1)S_n^2}{\sigma^2} \sim \chi^2(n-1) \\ &= \frac{\Gamma(\frac{n}{2})}{\Gamma(\frac{n-1}{2})}\sqrt{\frac{2}{n-1}} \sigma \\

\hat{\sigma}^{\text{UMVUE}} &= \frac{\Gamma(\frac{n-1}{2})}{\Gamma(\frac{n}{2})}\sqrt{\frac{n-1}{2}}\cdot  S_n \end{align*} $$

먼저 불편추정량에 대한 극한분포는 다음을 통해 구할 수 있다.

$$ \begin{align*} \sqrt{n}(S_n^2- \sigma^2) &= \sqrt{n}(c_n\bar{Y}_n - \sigma^2) - \sqrt{n}c_n(\bar{X}-\mu)^2 \text{ where } Y_i  = (X_i-\mu)^2, c_n = \frac{n}{n-1} \\ &=\sqrt{n}c_n(\bar{Y}_n-\sigma^2)+\sqrt{n}(c_n-1)\sigma^2 -\sqrt{n}c_n(\bar{X}-\mu)\cdot (\bar{X}-\mu) \\ &\xrightarrow{d} \mathcal{N}(0, \text{Var}(Y_1)) + 0+0 \text{ where } \text{Var}(Y_1) = \mathbb{E}[(X_1-\mu)^4]-\sigma^4

\end{align*} $$

한편 $(X_1-\mu)^2  \overset{d}{=} \sigma^2W, W \sim \chi^2(1)$이고, $\mathbb{E}[W]=1, \text{Var}[W]=2$임으로, $\mathbb{E}[(X_1-\mu)^4] = \sigma^4\mathbb{E}[W^2] = 3\sigma^4$이다. 즉, $\sqrt{n}(S_n^2 - \sigma^2) \xrightarrow{d} \mathcal{N}(0,2\sigma^4)$이다. 이제, $x>0$에서 일대일 변환인 $g(x)=\sqrt{x}$를 사용하여, $\sqrt{n}(\hat{\sigma}^{\text{UB}}-\sigma) \xrightarrow{d} \mathcal{N}(0,\frac{\sigma^2}{2})$임을 알 수 있다.

이제 $\hat{\sigma}^{\text{UMVUE}}$에 대한 극한분포를 유도해보자.

$$ \begin{align*}

\sqrt{n}(\hat{\sigma}^{\text{UMVUE}}-\sigma) &= \sqrt{n}(c_n S_n  - \sigma) \text{ where } c_n = \frac{\Gamma(\frac{n-1}{2})}{\Gamma(\frac{n}{2})}\sqrt{\frac{n-1}{2}}  \\ &= \sqrt{n}c_n(S_n-\sigma) + \sqrt{n}(c_n-1)\sigma \end{align*} $$

만약 $c_n = 1+O(n^{-p})(p\geq1/2)$꼴이면, 불편추정량과 같이 $\mathcal{N}(0,\frac{\sigma^2}{2})$로 분포수렴할 것이다. 한편 스털링근사와 테일러 전개를 통해 다음이 성립함을 알고 있다.

1. $\Gamma(m+1) = m! = m^{m+1/2}e^{-m}\sqrt{2\pi}(1+ r_m), \ \sqrt{m}r_m \rightarrow 0$
2. $(1+a/n)^n = e^{a}(1+R_n), \ \sqrt{n}R_n \rightarrow 0$

이제 1의 스털링 근사식에 $m+1=n/2$를 통해 $c_n$에 대입하면 다음과 같다.

$$ \begin{align*}

c_n&= \frac{\Gamma(\frac{n-1}{2})}{\Gamma(\frac{n}{2})}\sqrt{\frac{n-1}{2}} = \frac{\Gamma(m-1/2+1)}{\Gamma(m+1)} \sqrt{m+\frac{1}{2}} \\&=\frac{(m-\frac{1}{2})^{m}e^{-m+1/2}\sqrt{2\pi}(1+r_{m-1/2})}{m^{m+1/2}e^{-m}\sqrt{2\pi}(1+r_m)} \cdot \sqrt{\frac{2m+1}{2}} \\ &= (1-\frac{1}{2m})^{m}\sqrt{e \cdot \frac{2m+1}{2m}}\cdot \frac{1+r_{m-1/2}}{1+r_m} \\ &=(1+R_m)\cdot \sqrt{\frac{2m+1}{2m}} \cdot \frac{1+r_{m-1/2}}{1+r_m} \end{align*} $$

세번째 항은 1로 수렴하고 첫번째와 두번째 항을 전개해보자.

$$ \begin{align*}

1+R_n &= e^{1/2}(1-\frac{1}{2m})^m\\ &= e^{1/2}e^{m\log(1-\frac{1}{2m})} \\ &=\exp[\frac{1}{2} + m(-\frac{1}{2m}-\frac{1}{8m^2}-O(\frac{1}{m^3})]\\ &= \exp[-\frac{1}{8m}+O(\frac{1}{m^2})] \\ &= 1-\frac{1}{8m}+O(\frac{1}{m^2})\\ (1+\frac{1}{2m})^{1/2} &= 1 + \frac{1}{2}\cdot\frac{1}{2m} + O(\frac{1}{m^2}) \end{align*} $$

따라서, 다음이 성립한다.

$$ \begin{align*}

c_n &= (1-\frac{1}{8m}+O(\frac{1}{m^2}))(1+\frac{1}{4m}+O(\frac{1}{m^2})) \\ &= 1+\frac{1}{8m}+O(\frac{1}{m^2})\\ &=1+\frac{1}{4n}+O(\frac{1}{n^2})

\end{align*} $$

그러므로, $\text{ARE}(\{\hat{\sigma}\}^{\text{UB}}, \{\hat{\sigma}\}^{\text{UMVUE}}) = 1$이다.





# 지수분포

모수 $(\mu, \sigma) \in \mathbb{R} \times \mathbb{R}^+$에 대하여 $Exp(\mu,\sigma)$인 랜덤표본 $X_1,...,X_n (n\geq 2)$로부터 모수의 함수에 대한 UMVUE를 구해보자.

$$ \begin{align*} f(x;\mu,\sigma) &= \prod_{i=1}^n\frac{1}{\sigma}\exp(-\frac{x_i-\mu}{\sigma})I(x_i \geq \mu) \\ &= \log[-\frac{n\bar{x}-n\mu}{\sigma}+\log I(x_{(1)}\geq \mu) - n\log \sigma] \text{ where } x_{(1)} \leq ... \leq x_{(n)}

\end{align*} $$

즉, 모수공간이 2차원 열린구간 다면체를 포함하는 지수족임으로, $(X_{(1)}, \bar{X})$는 $(\mu,\sigma)$에 대한 완비충분통계량이다.

## 기댓값과 분산

$X_i \overset{d}{=}  \sigma Z_i + \mu, Z_i \sim Exp(1)$이고, $Z_{(1)} \leq ... \leq Z_{(n)}$에 대해

$$ \begin{align*} Z_{(1)}  &= \frac{1}{n}Y_1, Z_{(r)}-Z_{(r-1)}  = \frac{1}{n-r+1}Y_r, r=2,...,n

\end{align*} $$

라고 하면, $Y_1,...,Y_n$은 서로 독립인 $Exp(1)$을 따름을 알고 있다. 따라서, 다음이 성립한다.

$$ \begin{align*}

Z_{(r)} - Z_{(1)}  &= \sum_{k=2}^r\frac{1}{n-k+1}Y_k \\ \sum_{r=1}^n(Z_{(r)}-Z_{(1)}) &= \sum_{r=1}^n\sum_{k=2}^r\frac{1}{n-k+1}Y_k \\ &= Y_2+...+Y_n \sim \text{Gamma}(n-1,1)

\end{align*} $$

따라서, $\sum_{r=1}^n(Z_{(r)}-Z_{(1)})$는 모수에 의존하지 않는 보조 통계량이다. 그런데, $(X_{(1)}, \bar{X})$에서 $(X_{(1)}, \sum_{r=1}^n(X_{(r)}-X_{(1)})=:(S,T)$로의 변환은 일대일 변환임으로 마찬가지로 완비충분통계량이며, 각각은 $Y_1$과 $(Y_2,...,Y_n)$에 대한 함수로 독립이기에 다음을 만족한다.

$$ \begin{align*}

\mathbb{E}[X_{(1)}|S=s,T=t] &= \mathbb{E}[\sigma Z_{(1)}+\mu|s]\\ &= \frac{\sigma}{n}+\mu \\ \mathbb{E}[\sum_{r=1}^n(X_{(r)}-X_{(1)})|S=s,T=t] &= \sigma \mathbb{E}[\sum_{r=1}^n(Z_{(r)}-Z_{(1)})] \\ &=(n-1)\sigma

\end{align*} $$

따라서 UMVUE는 다음과 같다.

$$ \begin{align*}

\hat{\mu}^{\text{UMVUE}}  &= X_{(1)} - \frac{1}{n(n-1)}\sum_{r=1}^n(X_{(r)}-X_{(1)}) \\ \hat{\sigma}^{\text{UMVUE}} &= \frac{1}{n-1}\sum_{r=1}^n(X_{(r)}-X_{(1)})

\end{align*} $$

## $\eta = Pr_{\theta}(X_1 >a)$

위의 경우와 마찬가지로 $(S,T)=(X_{(1)}, \sum_{r=1}^n(X_{(r)}-X_{(1)}))$을 완비충분통계량으로 하여 다음을 구한다.

$$ Pr[\frac{X_1-X_{(1)}}{\sum_{i=1}^nX_i - X_{(1)}}>\frac{a-s}{t}|S=s,T=t] $$

이제 $Q=(X_1-X_{(1)})/(\sum_{i=1}^nX_i - X_{(1)})$의 누적분포함수는 다음과 같은 조건부 혼합모델로 쓸 수 있다.

$$ \begin{align*}

Pr[Q\leq q] &= Pr[X_1=X_{(1)}]Pr[0 \leq q] + Pr[X_1 \neq X_{(1)}]\cdot F(q) \\ &= \frac{1}{n}I[0\leq q] + (1-\frac{1}{n})F(q)

\end{align*} $$

여기서 $F$는 $X_1 > X_{(1)}$일 때, $Q$의 조건부 분포의 누적밀도함수이다. 먼저, $X_1 > X_{(1)}$하에서, 지수분포의 무기억성에 의해 $X_1-X_{(1)}$은 마찬가지로 $Exp(\sigma)$를 따른다. 한편 분모에 대해서는 다음과 같다.

$$ \begin{align*} \sum_{i=1}^nX_i - X_{(1)} &= X_1-X_{(1)} + \sum_{i=2}^nX_i - X_{(1)} \\ &=X_1-X_{(1)} + \sum_{i:X_i \neq X_{(1)}}(X_i-X_{(1)}) + X_{(1)}-X_{(1)} \\ &\overset{d}{=} \ \sum_{i=1}^{n-1} V_i \text{ where } V_i \overset{\text{iid}}{\sim} Exp(\sigma) \end{align*} $$

**Note.** 지수분포의 무기억성은 위치척도를 빼준 $Y_i \overset{d}{=} X_i-\mu, Y_i \sim Exp(\sigma)$에 대해 $Y_i - Y_{(1)}$로부터 유도되기에, 위치척도 $\mu$는 사라진다.

**Note**. $X_i$들은 연속형 확률변수임으로, 일반적으로 $Pr[\{i\neq j : X_i=X_j\}]=0$이다. 다시 말해서 $n$개의 랜덤샘플 중 최솟값을 갖는 랜덤샘플은 오직 하나이다.

따라서, $F$는 $V_1/(V_1+V_2+...V_n) \sim \text{Beta}(1,n-2)(n\geq3)$의 누적밀도함수와 같다.

$$ F(q) = \begin{cases} 0 & \text{if } q <0 \\ 1-(1-q)^{n-2} & \text{if } 0\leq q \leq 1\\ 1 & \text{if } q > 1 \end{cases}

$$

즉, $Q$의 분포는 다음과 같다.

$$ \begin{align*}

Pr[Q> q]

&=\begin{cases} 1 & \text{if } q <0 \\ \frac{n-1}{n}(1 - q)^{n-2} & \text{if } 0\leq q \leq 1\\ 0 & \text{if } q > 1 \end{cases}

\end{align*} $$

즉, 모수에 의존하지 않는 보조통계량임으로 완비충분통계량과 독립이다. 따라서, UMVUE는 $q$에 $(a-S)/T$를 대입한 것과 같다.

$$ \hat{\eta}^{\text{UMVUE}} = \begin{cases} 1 & \text{if } a < S \\ \frac{n-1}{n}\left(1 - \frac{a-S}{T}\right)^{n-2} & \text{if } S \leq a \leq  S+T \\ 0 & \text{if } a > S+T \end{cases} $$

한편 $n=2$인경우, 즉 베타분포로부터의 유도가 불가능한 경우를 고려해보자. 완비충분통계량은 $(S,T) = (X_{(1)}, X_{(2)}-X_{(1)})$이고, UMVUE는 다음과 같다.

$$ \begin{align*} \hat{\eta}^{\text{UMVUE}} &= Pr(X_1>a \mid X_{(1)},X_{(2)}) \\ &=Pr(X_{(1)}=X_1)Pr(X_{(1)}>a\mid X_{(1)},X_{(2)}) + Pr(X_{(2)}=X_1)Pr(X_{(2)}>a\mid X_{(1)},X_{(2)}) \\ &= \frac{1}{2}[I(X_{(1)} > a) + I(X_{(2)} > a)] \\ &= \begin{cases} 1 & \text{if } a < X_{(1)} \ (\text{ or } a <S) \\ \frac{1}{2} & \text{if } X_{(1)}\leq a \leq X_{(2)} \ ( \text{ or } S \leq a \leq S+T)  \\ 0 & \text{if } X_{(2)} < a \ ( \text{ or } a > S+T) \end{cases} \end{align*} $$

즉, $n\geq 3$인 경우와 같다.



