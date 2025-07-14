---
layout: single
title: "Weighting based multiple testing cont"
categories: Statistics
tag: [Multiple testing, Weighthing based]
typora-root-url: ..\
author_profile: false
use_math: true
toc: true
header:
  overlay_image: /images/2025-07-12-multiple_test2/graph_representation.png
  overlay_filter: 0.5
---

저번 포스팅에 이어 이번에는 wBH절차 하에서 가설의 수가 충분히 많은 경우에 대해  데이터에만 의존 FDR의 상한을 유도해보자. 먼저, 확률변수의 수렴에 대해 복습해보자. 

Genovese, C. R., Roeder, K., & Wasserman, L. (2006) False Discovery Control With P-Value Weighting의 내용을 참고하였다.



# Convergence in probability v.s. Convergence almost surely



확률변수 열 $X_1,...,X_n$과 확률변수 $Z$에 대해 다음을 만족하면, 확률변수 열$X_n$은 확률변수$Z$로 확률수렴한다.

$$ \lim_{n \rightarrow \infty}Pr(|X_n-Z| \geq \epsilon) = 0, \ \forall \epsilon >0 $$

$X_n \xrightarrow{p} Z$라고 표기한다.

이번엔 다음과 같이 극한이 확률 함수 내부로 들어가서, 다음을 만족하면 확률변수 열$X_n$은 확률변수$Z$로 거의 확실하게 수렴한다.

$$ Pr(\lim_{n \rightarrow \infty}X_n=Z) = 1 $$

$X_n \xrightarrow{a.s.} Z$라고 표기한다. 확률 함수 내부의 집합을 표본공간 $\Omega$를 이용하여 아래와 같이 쓴다.

$$ C=\{w \in \Omega : \lim_{n \rightarrow \infty} X_n(w) = Z(w)\} $$

확률변수는 결국 표본공간 에서 실수를 매핑해주는 함수이므로, 이는 $X_n$이 $Z$로 수렴하는 모든 결과 $w$들을 모아놓은 집합이고 converges almost surely라는 것은 결국 이러한 집합의 확률이 $1$이라는 것이다. 반대로, 수렴하지 않는 예외적인 결과들의 집합의 확률이 $0$이다.

$C$는 잘 정의된 사건인가? 즉, 집합 $C$가 확률을 매길 수 있는 사건인지 확인해보자.

$\lim_{n \rightarrow \infty} X_n(w) = Z(w)$가 성립한다는 것은 엡실론-델타 논법과, 아르키메데스 성질에 의해 다음을 만족한다는 것과 같다.

$$ \forall k \in \mathbb{N}, \  \exists K \in \mathbb{N} \ s.t. \ \forall n\geq K, \ |X_n(w) - Z(w)| \leq \frac{1}{k} $$

먼저, 가장 내부의 집합을 아래와 같이 $A_{n,k}$로 나타내면 이는 자명하게 사건이다.

$$ A_{n,k} = \{w: |X_n(w)-Z(w)| \leq \frac{1}{k}\} $$

즉, $C$는 사건들의 가산 합집합과 교집합의 형태로 다음과 같이 나타낼 수 있으므로, 잘 정의된 사건이다.

$$ C = \cap_{k=1}^{\infty}(\cup_{n=1}^{\infty}(\cap_{n=K}^{\infty}A_{n,k})) $$



## 거의 확실한 수렴의 성질

### 거의 확실한 수렴이면 확률수렴한다.

거의 확실한 수렴이 더 강한 조건이다. 즉, $X_n \xrightarrow{a.s.} Z$이면, $X_n \xrightarrow{p} Z$이다. 이를 보이기 위해 다음과 같은 사건열 $B_n$을 고려해보자.

$$ B_n = \cup_{m=n}^{\infty}\{|X_m-Z| > \epsilon \} $$

사건열 $B_n$은 포함관계가 작아지는 사건들이기에, 사건의 극한을 정의할 수 있고, $w$가 모든 $B_n$에 속한다는 말은 $m$이 아무리 커도 $X_m(w)$가 $Z(w)$ 근방을 벗어난다는 말이다. 즉 $C$의 여사건임을 알 수 있다.

$$ \lim_{n \rightarrow \infty}B_n = \cap_{n=1}^{\infty}B_n = \{w \in \Omega: \lim_{m \rightarrow \infty}X_m(w)\neq  Z(w)\}=C^c $$

$X_n \xrightarrow{a.s.} Z$임을 가정하면, 이러한 사건의 극한의 확률값은 $0$임을 알 수 있다. 또한 확률측도의 연속성에 의해 다음이 성립한다.

$$ \begin{align*} 0&=Pr(\lim_{n \rightarrow \infty}B_n) = \lim_{n \rightarrow \infty}Pr(B_n) \\&=\lim_{n \rightarrow \infty}Pr( \cup_{m=n}^{\infty}\{|X_m-Z| > \epsilon \}) \geq \lim_{n \rightarrow \infty}Pr( |X_n-Z| > \epsilon) \geq 0

\end{align*} $$

따라서 $X_n \xrightarrow{p} Z$이다.



### 연속성 사상 정리 (The continuous mapping theorem, CMT)

거의 확실한 수렴에 대해서도 연속성 사상정리가 성립한다. 즉, 연속함수 $g$에 대해 $X_n \xrightarrow{a.s.} Z$이면 $g(X_n) \xrightarrow{a.s.} g(Z)$



### 지배 수렴 정리(The Dominated Convergence Theorem, DCT)

1. $X_n \xrightarrow{a.s.} Z$ **(수렴조건)**
2. $Y_n = \|X_n-Z\|$에 대해 항상 크거나 같은 값을 갖는 확률변수 $W$가 존재하고, $W$의 기댓값이 존재한다. **(지배조건)**

위 두 조건을 만족하면, 아래와 같이 적분과 극한의 순서를 바꿀 수 있다.

$$ \lim_{n\rightarrow \infty}\mathbb{E}[|X_n-Z|] = \mathbb{E}[\lim_{n\rightarrow \infty}|X_n-Z|] =0 $$





# FDR의 상한 (가설의 수가 충분히 큰 경우)

이제 wBH 절차의 임계점을 $T_{m}$이라고 표기하자.

$$ T_{m} = \sup\{t:\hat{C}_m(t)\geq \frac{1}{\alpha} \} \text{ where } \hat{C}_m(t) = \frac{\hat{D}_m(t)}{t\bar{W}*m} = \frac{\sum*{i}I(P_i\leq W_it)}{t\sum_iW_i} $$

$\hat{C}_m(t)$는 $Q$의 주변부 cdf $D$와 $W_i$의 기댓값 $\mu$의 추정량을 사용한 함수이므로, 점근분석을 위해 다음을 고려해보자.

$$ t_* = \sup\{t:C(t)\geq \frac{1}{\alpha} \} \text{ where } C(t) = \frac{D(t)}{t\mu} $$

Note. $D(t) = (1-a) \mu_0 t+ a \int F(wt)dQ_1(w)$임을 보였다.

만약 대립가설 하에서의 $P_i$의 조건부 cdf $F$가 support $[0,1]$에서 순오목함수라면, 즉 다음을 만족한다고 가정하자.

$$ F(\lambda x + (1-\lambda) y)>\lambda F(x) + (1-\lambda)F(y), \ \forall \lambda, x,y\in[0,1] $$

이러한 경우 $P_i$의 주변부 cdf $G = (1-a)U + aF$는 마찬가지로 $[0,1]$에서 순오목 함수이며, $Q$의 주변부 cdf $D$ 또한 $Q_1$의 support가 $(0,\infty)$이기에, $[0,1]$에서 적분항 내부에  $F(w(\lambda x + (1-\lambda) y))$가  $\lambda F(wx) + (1-\lambda)F(wy)$보다 항상 크기에 동일한 구간에 대해 적분한 결과도 항상 크다. 따라서 $D$도 $[0,1]$에서 순오목 함수이다.

또한 $D(0)=0$이며, $0<t_0<t_1<1$에 대해 $\lambda = t_0/t_1$을 대입하면 $t_0 = (1-\lambda)+\lambda t_1$이다. 즉, 다음과 같이 $C(t)$는 $(0,1)$에서 단조 감소함수이다.

$$ C(t_1) = \frac{D(t_1)}{t_1} = \frac{(1-\lambda)D(0)+\lambda D(t_1)}{t_0} < \frac{D(t_0)}{t_0} = C(t_0) $$

만약 $F$가 $[0,1]$에서 순오목함수라면, $T_m \xrightarrow{a.s.} t_*$이며, FDR은 $\alpha +o(1)$보다 작거나 같다.



##   1. $T_m$에서 $t_*$로의 거의 확실한 수렴



먼저, $\sum_iW_i=m$인 제약조건이 없는 서로 독립인 경우를 생각해보자. 양수 $b,\epsilon$ 를 고정하여 $T_m$이 $(t_*-b, t_*+b)$에 갇힘을 통해 수렴성을 보이려 한다.

다음은 증명없이 사용한다.

- **(글리벤코-칸텔리 정리)** $Q_i$는 서로 독립임으로, 주변부 cdf $D$와 이에 대한 emprical cdf $\hat{D}_m$에 대하여 충분히 큰 $m$에 대해 다음이 성립한다.

$$ \sup_u |\hat{D}_m(u) - D(u)| < \epsilon $$



- **(대수의 강법칙)** $\bar{W}_m \xrightarrow{a.s.}  \mu$

$C$는 단조 감소임으로, $$C(t_*+b) = 1/\alpha - \delta$$  ( $\delta >0$ )이다. 따라서 $t>t_*+b$인 경우의 위 두 정리를 사용하면 다음의 부등식이 성립한다.


$$
\begin{align*}\hat{C}_m(t) &= \frac{\hat{D}_m(t)}{t\bar{W}_m} \leq \frac{D(t) + \sup_u |\hat{D}_m(u) - D(u)|}{t\mu -t|\bar{W}_m-\mu|} \\ &\leq \frac{D(t) +\epsilon}{t\mu-t\epsilon} = C(t)(\frac{\mu}{\mu-\epsilon}) + \frac{\epsilon}{t(\mu-\epsilon)} \\ &\leq (\frac{1}{\alpha}-\delta)(\frac{\mu}{\mu-\epsilon}) + \frac{\epsilon}{(t_*+b)(\mu-\epsilon)} < \frac{1}{\alpha} \end{align*} 
$$




한편, $\hat{C}(T_m) \geq 1/\alpha$임으로, $T_m \leq t_*+b$이다.

마찬가지로, $$t < t_*-b$$인 경우에 대해서는 $$C(t_*-b) = 1/\alpha +\delta$$  ( $$\delta >0$$ )임으로,  $$\hat{C}_m(t) > 1/\alpha$$이다. 한편, $T_m$은 이러한 $t$들의 상한임으로 $$T_m \geq t_*-b$$여야 한다. 따라서, $T_m \xrightarrow{a.s.} t_*$이다.



## 2. $FDP(T_m)$에서  $FDP(t_*)$로의 거의 확실한 수렴

이제, $FDP(T_m)$이 $FDP(t_*)$로 거의 확실하게 수렴함을 보여보자.




$$
 FDP(T_m) = \frac{\sum_{i}I(Q_i \leq T_{m})(1-H_i)/m}{\sum_iI(Q_i\leq T_{m})/m} = \frac{\hat{V}_m(T_m)}{\hat{D}_m(T_m)} 
$$




충분히 큰 $m$에 대해 $$\hat{V}(T_m), \hat{D}_m(T_m)$$이 각각 $$V(t_*), D(t_*)$$로 확실하게 수렴함을 보이면, $x/y$는 연속함수이기에 $FDP(t_*)$로 거의 확실하게 수렴한다. 두가지 모두 증명의 방식은 동일하기에 분모의 경우만 보여보자.


$$
|\hat{D}_m(T_m) - D(t_*)| \leq \sup_u |\hat{D}_m(u) - D(u)| + |D(T_m) - D(t_*)| \rightarrow 0
$$




오른쪽 항의 첫번째 항은 글리벤코-칸텔리 정리에 의해 $0$으로 수렴하고, $$T_m \xrightarrow{a.s.} t_*$$이고 $D$는 순오목 함수이므로, 연속함수이다. 따라서 $D(T_m) \xrightarrow{a.s.} D(t_*)$이기에, 위는 $0$으로 수렴한다.

또한, $FDP$는 항상 $0,1$사이의 값을 갖으므로, 지배수렴정리에 의해 다음이 성립한다.


$$
\begin{align*} &|\mathbb{E}[FDP(T_m)] - \mathbb{E}[FDP(t_*)]| \le  \mathbb{E}[|FDP(T_m)-FDP(t_*)|] \rightarrow 0 \end{align*} 
$$




즉, FDR은 $\mathbb{E}[FDP(t_*)]$로 수렴한다. 또한 수렴값의 상한은 아래와 같다.


$$
\mathbb{E}[FDP(t_*)] = \frac{\mathbb{E}[(1-H_i)I(Q_i\leq t_*)]}{\mathbb{E}[\hat{D}(t_*)]} = \frac{(1-a)\mu_0t_*}{t_*\mu/\alpha} \leq \alpha 
$$




(위 부등식은 $\mu = (1-a)\mu_0+a\mu_1$이고, $W$의 support는 음이 아니기에 성립한다.)

따라서, $FDR \leq \alpha + o(1)$임을 알 수 있다.

