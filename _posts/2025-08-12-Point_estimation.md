---
layout: single
title: "추정량의 비교 1"
description: "모수의 함수에 대한 추정량의 비교"
categories: Statistics
tag: [Sufficiency, Complete, UMVUE, Rao-Blackwell, Lehmann-Scheffé, Cramer-Rao Inequality ]
typora-root-url: ..\
author_profile: false
use_math: true
toc: true
---



랜덤 표본이 $X_1,...,X_n \sim F_{\theta}, \theta \in \Omega$의 형태로, 즉 모집단의 분포가 모수 $\theta$로 표현될 수 있는 모형에 대해서 모수에 대한 함수로 표현될 수 있는 $\eta = \eta(\theta)$의 추정량에 대한 비교를 다룬다.

일반적으로 추정량의 비교 기준은 모수 $\theta$에 의존하지 않는 기준을 고안하며, 가장 보편적인 비교 기준으로는 최대평균제곱오차와 베이지안 평균제곱오차가 있다.


$$
 \begin{align*} \max_{\theta \in \Omega}MSE(\hat{\eta}, \theta) &= \max_{\theta \in \Omega} \mathbb{E}_{\theta}[(\hat{\eta} - \eta)^2] = \max_{\theta \in \Omega} [Var_{\theta}(\hat{\eta}) + (\mathbb{E}(\hat{\eta}) - \eta)^2] \\ \int_{\Omega} MSE(\hat{\eta},\theta)\pi(\theta)d\theta &=\int_{\Omega} \mathbb{E}_{\theta}[(\hat{\eta}-\eta)^2]\pi(\theta)d\theta =: r(\pi, \hat{\theta})

\end{align*}
$$




최대평균제곱오차를 최소로 하는 추정량 $\hat{\eta}^*$을 최소최대 평균제곱오차 추정량이라고 하며, 사전 밀도 함수 $\pi$에 대해 베이지안 평균제곱오차를 최소로 하는 추정량 $\hat{\eta}^{\pi}$를 $\pi$에 대한 베이지안 평균제곱오차 추정량이라고 한다.

# UMVUE(Uniformly Minimum Variance Unbiased Estimator)

추정량의 후보를 제한하여, 제한한 후보들중에서 가장 좋은 추정량을 찾는 방법을 고려해볼 수도 있다.

추정량의 후보를 불편 추정량으로 제한한다면, 즉 임의의 $\theta \in \Omega$에 대해 $\mathbb{E}_{\theta}[\hat{\eta}] = \eta$를 만족하는 추정량들에 대해서는 평균제곱오차가 추정량의 분산으로 나타남을 알 수 있다.

이러한 후보들 중에서 다시 임의의 $\theta \in \Omega$에 대해서 분산이 가장 작은 추정량이 존재한다면, 이를 전역최소분산불편추정량(UMVUE)라고 한다.

만약 $\eta$의 분산이 실수라면, UMVUE는 유일하게 존재한다. 이는 다음과 같이 보일 수 있다.

만약 $$\hat{\eta}_1, \hat{\eta}_2$$가 UMVUE라면, 임의의 $\theta \in \Omega$에 대해  $$\mathbb{E}_{\theta}[\hat{\eta}_1-\hat{\eta}_2]=0$$이다. 이제 $$\text{Var}_{\theta}[\hat{\eta}_1-\hat{\eta}_2]=0$$ 또한 모든 $\theta$에 대해 성립한다면, 체비셰프 부등식을 사용하여, $$Pr_{\theta}(\hat{\eta}_1=\hat{\eta}_2)=1$$임을 보일 수 있기에 유일성이 보장된다.


$$
 0 \leq \text{Var}_{\theta}[\hat{\eta}_1-\hat{\eta}_2] = 2(\text{Var}_{\theta}[\hat{\eta}_1]-\text{Cov}[\hat{\eta}_1, \hat{\eta}_2]) 
$$




즉,임의의 $\theta$에 대해  $$\text{Cov}_{\theta}(\hat{\eta}_1, \hat{\eta}_2) \geq \text{Var}{\theta}(\hat{\eta}_1)$$임을 보이면 충분하다.

$\hat{\eta}_3=(\hat{\eta}_1+\hat{\eta}_2)/2$라고 정의하면 다음이 성립한다.


$$
 \begin{align*} \text{Var}_{\theta}(\hat{\eta}_3) - \text{Var}_{\theta}(\hat{\eta}_1) &= \frac{1}{2}[Var_{\theta}(\hat{\eta}_1)+\text{Cov}_{\theta}(\hat{\eta}_1, \hat{\eta}_2)] - \text{Var}_{\theta}(\hat{\eta}_1) \\ &= \frac{1}{2}[\text{Cov}_{\theta}(\hat{\eta}_1, \hat{\eta}_2)] - \text{Var}_{\theta}[(\hat{\eta}_1) ]\geq 0, \ \forall \theta \in \Omega

\end{align*} 
$$




즉, UMVUE는 유일하게 존재한다.

UMVUE를 구하는 방법으로는 완비충분통계량을 사용하는 방법이 있다.

### 충분통계량

랜덤샘플 $X=(X_1,...,X_n)$은 $n$차원의 복잡한 데이터이다. 이 모든 값을 개별적으로 다루는 대신, 적절한 통계량을 사용하여 전체 표본공간을 분할하는 방법을 고려해볼 수 있다.

다시말해서, $X$의 support $$\mathcal{X} =\{x=(x_1,...,x_n):f(x)>0\}$$를 함수 $u:\mathcal{X} \rightarrow \mathcal{T}$를 사용하여 $\{A_t\}$로 분할하여, 인덱스$t\in \mathcal{T}$에 대한 각각의 집합 $$A_t =\{x=(x_1,...,x_n):T(x)=t \}$$에 속하는 표본들에 대해서는 동등하게 취급한다는 것이다.

한편, 우리는 모수의 추정량에 대해 다루고 있기에, 이러한 데이터 축약 과정에서 모수 $\theta$의 정보는 소실되지 않아야 할 것이다. 이를 모수 $\theta$에 대한 충분성이라고 한다.

통계량 $Y = u(X_1,...,X_n)$에 대하여 $Y$를 조건부로한 랜덤 표본이 모수 $\theta$에 의존하지 않으면, $Y$를 $\theta \in \Omega$에 관한 충분통계량이라고 한다. 즉, 다음을 만족한다.

$$ Pr_{\theta_1}[(X_1,...,X_n)^T\in A|Y=y] = Pr_{\theta_2}[(X_1,...,X_n)^T\in A|Y=y], \ \forall A,y,\theta_1,\theta_2 $$

즉, 충분통계량은 모수 $\theta$에 대한 모든 정보를 담고 있는 요약하는 통계량이다. 다음은 주어진 확률분포로부터 충분통계량을 계산하는 방법에 대한 정리이다.

**[네이만-피셔 분해정리]** 통계량 $Y = u(X_1,...,X_n)$가 $\theta \in \Omega$에 관한 충분통계량이기 위한 필요충분조건은 다음과 같다.

$$ \forall \theta \in \Omega, \ \exists k_1,k_2 \text { s.t. } f(x;\theta)=\prod_{i=1}^nf(x_i;\theta) = k_1(u(x),\theta)k_2(x), \ \forall x =(x_1,....,x_n) $$

이산형 확률분포에 대해서 성립하는지를 보여보자.

임의의 순서통계량 $Y$와 임의의 $x=(x_1,...,x_n),\theta$에 대해 다음이 성립한다.


$$
\begin{align*}

\prod_{i=1}^n Pr_{\theta}(X_i=x_i) &= Pr_{\theta}(X=x|Y=u(x))Pr_{\theta}(Y=u(x))\\ &= Pr(X=x|Y=u(x))Pr_{\theta}(Y=u(x)) \\ &= k_2(x)k_1(u(x),\theta)

\end{align*}
$$




$Pr(X=x\|\theta)=k_1(u(x),\theta)k_2(x)$반대로, 임의의 $x=(x_1,...,x_n),y,\theta$에 대해서 를 만족하는 $k_1,k_2$가 존재한다고 가정하자. 이제 다음이 성립한다.


$$
\begin{align*}

Pr_{\theta}(X=x|Y=y) &= \frac{Pr_{\theta}(X=x,Y=y=u(x))}{Pr_{\theta}(Y=y=u(x))} \\ &= \frac{Pr_{\theta}(X=x,X\in u^{-1}(y))}{Pr_{\theta}(X \in u^{-1}(y))} \\ &=\frac{k_1(u(x),\theta)k_2(x)I[u(x)=y]}{\sum_{x':u(x')=y}k_1(u(x'),\theta)k_2(x')} \\ &=\frac{k_2(x)I[u(x)=y]}{\sum_{x':u(x')=y}k_2(x')} \end{align*}
$$




즉 $Pr_{\theta}(X=x\|Y=y)$는 $\theta$에 의존하지 않음으로, $Y$는 $\theta$에 대한 충분통계량이다.

(연속형의 경우는 확률밀도함수를 사용하여 합에 대한 부분을 적분으로 바꿔 비슷하게 증명할 수 있다.)

### 충분 순서 통계량

비모수추정과 같이 랜덤샘플 $X_1,...,X_n$이 확률밀도함수 $f$로부터 추출되었을 때에도 다음은 성립한다.

$$ f(x) = \prod_{i=1}^nf(x_i) = \frac{1}{n!}\cdot(n!\prod_{i=1}^nf(x_{(i)}) )\text{ where } x_{(1)} \leq ... \leq x_{(n)} $$

분해정리에 의해 $(X_{(1)},...,X_{(n)})^T$는 확률밀도함수 $f$에 대한 순서통계량이다. 이처럼 $f$가 주어지지 않은 경우에도 랜덤샘플의 순서를 고정하는 방법으로 데이터를 압축할 수 있다. 분포의 형태에 대한 어떠한 가정도 없으므로, 표본의 경험적 분포를 온전히 보존하는 순서통계량보다 더 압축은 불가능하다.

심지어 $f$가 주어진 경우에도, 순서통계량보다 더 데이터를 압축할 수 있는 분포들도 있다. 코쉬분포와 로지스틱 분포가 그러한 경우이다.

$$ \begin{align*}

f(x;\mu,\sigma)&=\frac{1}{\pi\sigma(1+(\frac{x-\mu}{\sigma})^2)} \\

f(x;\mu,\sigma)&=\frac{\exp(-\frac{x-\mu}{\sigma})}{\sigma(1+\exp(-\frac{x-\mu}{\sigma}))^2} \end{align*} $$

한편, 지수족에 속한 확률분포는 다음과 같이 분해된다.

$$ \begin{align*}

f(x;\theta) &= \prod_{i=1}^n\exp[\eta(\theta)^TT(x_i)-A(\theta)+S(x_i)] \\ &= \exp[\eta(\theta)^T(\sum_{i=1}^n(T(x_i)) - nA(\theta) + \sum_{i=1}^nS(x_i)]

\end{align*} $$

즉, 충분통계량은 표본의 수 $n$에 상관없이 $\sum_{i=1}^nT(X_i)$라는 고정된 차원의 벡터로 주어진다.



신뢰성 분석이나 생존 분석에서는 $X_1,...,X_n$의 전체 데이터가 아닌 순서통계량 $X_{(1)} \leq ... \leq X_{(r)} (1\leq r \leq n)$과 같이 중도 절단(censored)된 데이터가 획득된다. 즉, $X_{(r)}$보다 크거나 같은 $(n-r)$개의 데이터에 관한 정보는 주어지지 않는다.

이제 $X_1,...,X_n$의 확률밀도함수를 $f(x;\theta)$, 누적분포함수를 $F_{\theta}(x)$라고 하면 $(X_{(1)},...,X_{(r)})$의 결합확률밀도 함수는 다음과 같이 쓸 수 있다.

$$ \begin{align*} [\frac{n!}{(n-r)!} (1-F_{\theta}(x_{(r)}))^{n-r} ]\cdot [\prod_{i=1}^rf(x_{(i)};\theta) ] \cdot I(x_{(1)}\leq ...\leq x_{(r)})

\end{align*} $$

모수 $(\mu,\sigma) \in \mathbb{R} \times \mathbb{R}^+$를 갖는 지수분포 $Exp(\mu, \theta)$를 예로들면 $Y=(X_{(1)},...,X_{(r)})$의 결합확률밀도함수는 다음과 같이 나타난다.

$$ \begin{align*} f(y;\mu,\sigma)&= \frac{n!}{(n-r)!} \cdot \frac{1}{\sigma^r}\exp[- (n-r)\frac{y_r-\mu}{\sigma} - \sum_{i=1}^r\frac{y_i - \mu}{\sigma}] \cdot I(y_1 \geq \mu) \\ &= \frac{n!}{(n-r)!} \cdot \exp[-\frac{(n-r)y_r+\sum_{i=1}^r y_i - n\mu}{\sigma}]\cdot \frac{I(y_1\geq \mu)}{\sigma^r}

\end{align*} $$

따라서, 분해정리에 의해 $(\mu, \sigma)$에 대한 충분통계량은 $(X_{(1)}, (n-r)X_{(r)}+X_{(1)}+...+X_{(r)})$이다.

추가적으로 분해정리에 의해 $\mu$가 알려진 경우에 $\sigma$에 대한 충분통계량은 $(n-r)X_{(r)}+X_{(1)}+...+X_{(r)}$이고 $\sigma$가 알려진 경우에 $\mu$에 대한 충분통계량은 $X_{(1)}$임을 알 수 있다.

$$ \begin{align*} &\frac{n!}{(n-r)!} \cdot \exp[-\frac{(n-r)y_r+\sum_{i=1}^r y_i - n\mu}{\sigma}]\frac{I(y_1\geq \mu)}{\sigma^r} \\ =&\frac{n!I(y_1\geq \mu)}{(n-r)!}\cdot  \frac{1}{\sigma^r} \exp[-\frac{(n-r)y_r+\sum_{i=1}^r y_i - n\mu}{\sigma}] \\ =& \frac{n!}{(n-r)!\sigma^r} \exp[-\frac{(n-r)y_r+\sum_{i=1}^r y_i}{\sigma}] \cdot \exp(\frac{n\mu}{\sigma})I(y_1 \geq \mu) \end{align*} $$



### 최소 충분 통계량

한편, 충분통계량의 정의를 살펴보면 충분통계량 $Y$의 일대일 변환으로 주어지는 통계량 $W=g(Y)$에 대해서도 마찬가지로 해당 모수에 대한 충분통계량임을 보일 수 있다.

즉, 특정 모수에 대해 충분통계량은 유일하게 존재하지 않는다. 충분통계량은 항상 특정 모수에 대해 정의되며, 데이터로부터 해당 모수의 정보를 모두 포함하고 있는 통계량이다.

따라서, 자연스럽게 데이터로부터 모수에 대한 정보를 가장 효율적으로 압축한, 가장 간단한 형태의 통계량을 떠올릴 수 있다.

다시말해서, 이는 다른 어떤 충분통계량으로도 표현될 수 있는 가장 간단한 형태의 충분통계량이다. 이러한 통계량을 최소충분통계량이라고 한다.

임의의 충분통계량 $S=v(X_1,...,X_n)$에 대해 $Y=g(S)$를 만족하는 함수 $g$가 항상 존재하면, $Y=u(X_1,...,X_n)$는 최소 충분통계량이다.

이를 다시말하면, 임의의 $x=(x_1,...,x_n),y=(y_1,...,y_n)$에 대해 $v(x)=v(y)$를 만족하면, $u(x)=u(y)$이다.

랜덤샘플 $X=(X_1,...,X_n)$의 확률밀도함수를 $f(x;\theta), \theta\in \Omega$라고 할 때, 임의의 $\theta \in \Omega$에 대해 다음을 만족하는 함수 $u(X)$가 존재한다면, $Y=u(X)$는 최소충분통계량이다.

$$ \forall x,y\in\mathcal{X}, \  \frac{f(x;\theta)}{f(y;\theta)}=h(x,y)   \Leftrightarrow u(x)=u(y)

$$

증명과정은 다음과 같다.

먼저 $Y=u(X)$가 충분통계량임을 보이자. $\mathcal{X}$에 대한 $u$의 상을 $\mathcal{T}$라고 하면, 임의의 $x \in \mathcal{X}$에 대해 $x \in A_t (t\in\mathcal{T})$이고, $A_t$에 속하는 다른 표본을 $x_{u(x)}$라고하자. 즉, $u(x) =u(x_{u(x)})$를 만족한다. 따라서, $f(x;\theta)/f(x_{u(x)};\theta)$는 $h(x,x_{u(x)})$이고 $x_{u(x)}$또한 $x$에 대응하는 표본으로 고정하였기에, $x \in \mathcal{X}$에서 $f(x;\theta)/f(x_{u(x)};\theta)$로의 변환은 $\theta$에 의존하지 않는 $x$에 대한 함수이다.  이를 $k_2(x)$라고 정의하자.

이제, $t\in \mathcal{T}, \theta \in \Omega$에 대해 $u(x_t)=t$를 만족하는 $x_t$를 하나 잡아서 $k_1(t,\theta) = f(x_t;\theta)$를 정의하면 다음이 성립한다.

$$ f(x;\theta) = \frac{f(x;\theta)}{f(x_{u(x)};\theta)} \cdot f(x_{u(x)};\theta) = k_1(u(x),\theta)k_2(x) $$

분해정리에 의해 $Y$는 충분통계량이다. 이제, 임의의 충분통계량 $S=v(X)$에 대해 다음을 만족하면 $Y$는 최소충분통계량이다.

$$ \forall x,y\in \mathcal{X}, \ v(x)=v(y) \Rightarrow u(x)=u(y) $$

$S$는 충분통계량임으로 분해정리에 의해 $f(x;\theta) = k_1'(v(x),\theta)k_2'(x)$를 만족하는 $k_1',k_2'$가 존재한다. 따라서 $v(x)=v(y)$인 $(x,y)$에 대해 다음이 성립한다.

$$ \frac{f(x;\theta)}{f(y;\theta)} = \frac{k'_2(x)}{k_2'(y)}=h(x,y) \Rightarrow u(x)=u(y) $$

그러므로 $Y$는 최소충분통계량이다.

e.g. 랜덤샘플이 $X_1,...,X_n \sim \mathcal{N}(\mu,\sigma^2)$를 만족한다고하자. 임의의 $x=(x_1,...,x_n),y=(y_1,...,y_n)$에 대해 다음이 성립한다.

$$ \begin{align*} \frac{f(x;\mu,\sigma^2)}{f(y;\mu,\sigma^2)} &= \exp[-\sum_{i=1}^n\frac{(x_i-\mu)^2 - (y_i-\mu)^2}{2\sigma^2}] \\ &=\exp[\frac{-(n-1)(s_x^2-s_y)^2+n((\bar{x}-\mu)^2-(\bar{y}-\mu))^2}{2\sigma^2}] \text{ where } s_x^2=\frac{1}{n-1}\sum_{i=1}^n(x_i-\bar{x})^2 \\

\end{align*} $$

즉, 위 비가 $(\mu,\sigma^2)$에 의존하지 않을 필요충분조건은 $\bar{x}=\bar{y}$이고 $s_x^2=s_y^2$이다. 따라서, $(\bar{X},S_X^2)$은 최소충분통계량이다.

한편, 최대가능도추정량과  임의의 충분통계량 $S=u(X_1,...,X_n)$에 대한 분해정리로부터 다음이 성립한다.  


$$
\begin{align*} \hat{\theta}^{\text{MLE}} &= \underset{\theta \in \Omega}{\operatorname{argmax}}\prod_{i=1}^nf(x_i;\theta) = \underset{\theta \in \Omega}{\operatorname{argmax}}k_1(u(x),\theta)k_2(x) \\ &= \underset{\theta \in \Omega}{\operatorname{argmax}}k_1(u(x),\theta)

\end{align*}
$$




즉, 최대가능도추정량 $\hat{\theta}^{\text{MLE}}$가 유일하게 존재한다면, 이는 임의의 충분통계량에 관한 함수로 나타낼 수 있음을 의미한다.

또한, $\hat{\theta}^{\text{MLE}}$가 충분통계량이라면, 이는 임의의 충분통계량의 함수로 나타낼 수 있는 충분통계량이다.

그러므로, 유일한 $\hat{\theta}^{\text{MLE}}$가 존재하며 심지어 충분통계량이라면, 최소충분통계량임을 알 수 있다.

e.g. 랜덤샘플이 $X_1,...,X_n \sim \mathcal{N}(\mu,\sigma^2)$이면, $\hat{\theta}^{\text{MLE}} = (\bar{X}, \sum_{i=1}^n(X_i-\bar{X})^2/n)$으로 유일하게 주어짐을 알고 있다. 즉, $(\bar{X}, \sum_{i=1}^n(X_i-\bar{X})^2/n)$는 최소 충분 통계량이다. 위에서 구한 경우에서 $g(x,y) = (x,ny/(n-1))$로의 일대일 변환한 결과와 같다.

Note. 최소충분통계량은 유일하게 존재하지 않는다. 최소충분통계량 $Y$의 일대일 변환 $g(Y)$ 또한 최소충분통계량이다.

### 라오-블랙웰 정리(Rao-Blackwell Theorem)

충분통계량 $Y=u(X_1,...,X_n)$을 사용하여, 임의의 추정량 $\hat{\eta}$의 평균제곱오차를 감소시킬 수 있다.

$$ \hat{\eta}^{\text{RB}}= \hat{\eta}^{\text{RB}}(y) = \mathbb{E}[\hat{\eta}|Y=y] $$

충분통계량을 조건부로 고정하였으므로, $\hat{\eta}^{\text{RB}}$는 모수 $\theta$에 의존하지 않기에 통계량이라 할 수 있고, 따라서 $\eta$의 추정량으로 사용할 수 있다. (기댓값에 $\theta$를 표기하지 않은 이유도, 이미 $\theta$의 모든 정보를 포함한 $Y$가 고정되었기 때문이다.)

또한, 조건부 기댓값의 성질로 인해 다음이 성립한다.


$$
 \begin{align*} \mathbb{E}_{\theta}[\hat{\eta}^{\text{RB}}] &=\mathbb{E}_{\theta}[[ \mathbb{E}[\hat{\eta}|Y=y]] = \mathbb{E}[\hat{\eta}] , \ \forall \theta \in \Omega \\ \text{Var}_{\theta}[\hat{\eta}] &\geq \text{Var}_{\theta}[[ \mathbb{E}[\hat{\eta}|Y=y]] = \text{Var}[\hat{\eta}^{\text{RB}}],  \forall \theta\in \Omega

\end{align*} 
$$




즉, 편향은 변함이 없고 분산이 감소함으로 평균제곱오차는 감소한다.

### 완비 통계량

통계량 $Y = u(X_1,...,X_n)$에 대하여 다음을 만족하면, $\theta \in \Omega$에 관한 완비통계량이라고 한다.


$$
 \forall \theta \in \Omega , \ \mathbb{E}_{\theta}(g(Y)) = 0 \Rightarrow  \forall \theta \in \Omega ,Pr_{\theta}(g(Y)=0)=1
$$




다중모수 지수족의 경우, 모수에 대한 일대일 변환 $g:\Omega \rightarrow N$을 통해 모수에 대한 함수 $\eta = (\eta_1,...,\eta _k)^T= g(\theta)$를 사용하여 확률밀도 함수를 다음과 같이 쓸 수 있다.

$$ f(x;\eta) = \exp[\eta^TT(x) - A(\eta)-S(x)], \ T(x) = (T_1(x),...,T_k(x))^T $$

다음 세가지 조건을 만족하면, $\eta \in N$에 관한 완비충분통계량은 아래와 같다.

$$ \sum_{i=1}^nT(x_i) = (\sum_{i=1}^nT_1(x_i),...,\sum_{i=1}^nT_k(x_i))^T $$

1. support $\mathcal{X}=\{x:f(x;\eta)>0\}$은 모수에 의존하지 않는다.
2. 모수공간 $N$이 $(a_1,b_1)\times...\times(a_k,b_k) \subset \mathbb{R}^k$를 포함한다.
3. 임의의 $c \in \mathbb{R}^k (c \neq 0)$에 대해 $c^TT(X)$는 상수가 아니다.

이는 충분성에 대해서는 이미 분해정리를 통해 위에서 보였고, 완비성에 대해서는 라플라스 변환의 유일성을 이용하여 보일수 있다.

완비성의 조건이 의미하는 바는 다음과 같다.

서로 다른 추정량 $\hat{\eta}_1, \hat{\eta}_2$에 대해서 라오-블랙웰 정리는 충분통계량을 조건부로하여 각각의 추정량보다 분산이 작은 $\hat{\eta}^{RB}_1, \hat{\eta}^{RB}_2$를 만들 수 있음을 보였지만, $\hat{\eta}^{RB}_1$의 분산이 $\hat{\eta}_2$보다 작은지는 확인할 수 없다.(반대의 경우도 마찬가지)

위의 UMVUE의 유일성에 관한 증명과정을 참고하면, 이러한 조건을 이용하여 두개의 라오-블랙웰 추정량 $\hat{\eta}^{RB}_1, \hat{\eta}^{RB}_2$를 이용해 $g(Y) = \hat{\eta}^{RB}_1-\hat{\eta}^{RB}_2$로 정의하여, 충분통계량 $Y$가 완비통계량이기도 하면 완비성의 정의에 의해 $\hat{\eta}^{RB}_1=\hat{\eta}^{RB}_2$를 만족하기에,  $\hat{\eta}^{RB}_1$의 분산이 $\hat{\eta}_2$보다 작음을 보일 수 있기에 UMVUE임을 확인할 수 있다.



### 레만-셰페 정리 (Lehmann-Scheffé Theorem)

완비충분통계량을 사용하여 UMVUE를 구하는 방법이다.

통계량 $Y = u(X_1,...,X_n)$이 $\theta\in \Omega$에 대한 완비충분통계량이라면 다음의 두가지 방법으로 $\eta=\eta(\theta)$의 UMVUE를 구할 수 있다.

1. 임의의 불편추정량 $\hat{\eta}^{UE}$를 구하여, $Y$에 대해 조건부로 하는 라오-블랙웰 추정량을 구하면 UMVUE이다.
2. $Y$에 대한 함수 $\delta(Y)$가 $\eta$의 불편추정량이면, UMVUE이다.

1의 경우 위의 논의에서 충분히 보였고, 2에 대해서는 $g(Y) = \delta(Y)- \hat{\eta}^{\text{RB}}$를 이용하여, 완비성의 성질을 통해 보일 수 있다.

정리하자면, 완비충분통계량을 이용하여 UMVUE를 구하는 두가지 방법으로 첫번째는 불편추정량을 구하여 완비충분통계량이 주어진 조건 하에서 조건부 기댓값을 구하거나, 완비충분통계량에 대한 함수로 불편추정량을 구하는 방법이 있다.

### 바수의 정리 (Basu's Theorem)

한편, 첫번째 방법으로  완비충분통계량이 주어진 조건 하에서 조건부 기댓값을 구하는데에 있어서 완비충분통계량과 서로 독립인 통계량을 사용하면 계산이 더 간편해진다.

통계량 $Z=v(X_1,...,X_n)$가 완비 충분통계량 $Y=u(X_1,...,X_n)$과 독립임은 다음과 동치이다.

$$ Pr(Z\in B|Y) = Pr(Z \in B), \  \forall B $$

$Y$의 완비성을 이용하여 위에 대한 충분조건은 고정된 $B$에 대해 다음을 만족하는 것이다.

$$ \forall \theta \in \Omega , \ \mathbb{E}_{\theta}[Pr(Z\in B|Y)] = \mathbb{E}_{\theta}[Pr(Z \in B)] $$

$Y$는 충분통계량이기도 하기에 좌변은 $\theta$에 의존하지 않으므로 위 조건을 다음과 같이 다시 쓸 수 있다.

$$ \begin{align*} \mathbb{E}_{\theta}[Pr(Z \in B)] &= \mathbb{E}[Pr(Z\in B |Y)] = Pr(Z\in B), \ \forall \theta \in \Omega \\ \int_B f_X(v(x);\theta)dx &= \int_Bf_X(v(x))dx, \ \forall \theta \in \Omega

\end{align*} $$

즉, 통계량이 $\theta$에 의존하지 않으면, 완비충분통계량과 독립이다. 이러한 모수 $\theta \in \Omega$에 의존하지 않는 통계량을 $\theta$에 관한 보조 통계량이라고 한다.

이를 이용하여, 모수 $\theta>0$에 대한 지수분포로부터의 랜덤표본으로부터 $a>0$에서의 신뢰도 $\eta_a$의 UMVUE를 구해보자.

$$ \eta_a = Pr_{\theta}(X_1 > a) = \int_{a}^{\infty}\frac{1}{\theta}\exp(-\frac{x}{\theta})dx = \exp(-\frac{a}{\theta}) $$

즉, $\eta_a$는 모수 $\theta$에 관한 함수이다. 한편, $$\mathbb{E}_{\theta}[I(X_1>a)] = Pr_{\theta}(X_1>a)$$임으로 $$\hat{\eta}_a = I(X_1>a)$$는 $\eta_a$의 불편추정량이다.

이제 다음과 같이 $\theta$에 대한 완비충분통계량을 구할 수 있다.

$$ \begin{align*} f(x_i;\theta) &= \frac{1}{\theta}\exp(-\frac{x_i}{\theta}) \cdot I(x_i>0)\\ &=\exp[ -\frac{x_i}{\theta}+\log[I(x_i>0)] +\log\frac{1}{\theta}] \\ &=\exp[\eta^TT(x) -A(x)+S(\eta) \text { where } \eta=-\frac{1}{\theta}, \ T(x_i)= x_i \end{align*} $$

즉 모수공간은 $(-\infty,0)$임으로 열린구간을 포함하기에, $Y=\sum_{i=1}^nT(X_i)=X_1+...+X_n$은 $\eta$에 대한 완비충분통계량이고, 적절한 일대일 변환을 통해 $\theta$에 대한 완비충분통계량이 된다. 따라서 1의 방법으로 다음은 $\eta_a$의 UMVUE이다.

$$ \begin{align*} \mathbb{E}[I(X_1>a)|Y=y] &= Pr(X_1>a|Y=y)

\end{align*} $$

한편, $$Exp(\theta) \overset{d}{=} Gamma(1,\theta)$$이기에 다음이 성립한다.

$$ \frac{X_1}{X_1+...+X_n} \overset{d}{=} \frac{Gamma(1,\theta)}{Gamma(1,\theta)+Gamma(n-1,\theta)} \overset{d}{=} Beta(1,n-1) $$

또한, 이는 모수 $\theta$에 의존하지 않기에 보조통계량으로 완비충분통계량 $Y$와 독립이다. 따라서 다음이 성립한다.

$$ \begin{align*} \mathbb{E}[I(X_1>a)|Y=y] &= Pr(X_1>a|Y=y) \\ &= Pr(\frac{X_1}{X_1+...+X_n}>\frac{a}{y}|Y=y) \\ &= Pr(\frac{X_1}{X_1+...+X_n}>\frac{a}{y}) \\ &=\int_{a/y}^1 (n-1)(1-z)^{n-2}dz\cdot I(0<\frac{a}{y}<1) \\ &=(1-\frac{a}{y})^{n-1}I(a<y)

\end{align*} $$

그러므로, $\eta_a$의 UMVUE는 다음과 같다.

$$ \hat{\eta}_a^{\text{UMVUE}} = (1-\frac{a}{n\bar{X}})^{n-1}\cdot I(\bar{X} > \frac{a}{n}) $$

Note. $n$이 충분히 크면, $(1-\frac{a/\bar{X}}{n})^{n-1} \approx e^{-a/\bar{X}}$이 성립한다. 즉, $\hat{\eta}^{\text{MLE}}$로 근사될 수 있다.



## 추정량의 비교

$n$개의 랜덤표본으로부터의 두 추정량 $\hat{\theta}^1, \hat{\theta}^2$에 대해 다음이 성립한다고 가정하자.

$$ \sqrt{n}(\hat{\theta}^i-\theta) \xrightarrow{d} \mathcal{N}(0,\sigma_i^2(\theta)), \ i=1,2 $$

$\hat{\theta}^1$의 $\hat{\theta}^2$에 대한 점근상대효율성(asymptotic relative efficiency)는 다음과 같이 정의한다.

$$ \text{ARE}(\{\hat{\theta}^1_n\},\{\hat{\theta}^2_n\}) = \frac{\sigma_2^2(\theta)}{\sigma_1^2(\theta)} $$

즉, 점근상대효율성이 크다는 것은 표본의 수가 많아질수록 상대적으로 추정의 정밀도가 더 좋음을 의미한다. 즉, 더 적은 표본 수로도 요구되는 추정오차한계를 만족할 수 있다.



### 크래머-라오 부등식 (Cramer-Rao Inequality)

랜덤 표본 $X=(X_1,...,X_n) \sim f(x;\theta), \ \theta\in\Omega$인 확률모형에 대해 적절한 조건하에서, 모수에 대한 함수 $\eta = \eta(\theta)$의 추정량 $\hat{\eta}_n$의 분산에 대해 다음의 부등식이 성립한다.


$$
\text{Var}(\hat{\eta}_n) \succeq (\frac{\partial}{\partial \theta}\mathbb{E}_{\theta}(\hat{\eta}_n))^T(nI(\theta))^{-1}(\frac{\partial}{\partial \theta}\mathbb{E}_{\theta}(\hat{\eta}_n)), \ \forall \theta \in \Omega
$$




$\eta \in \mathbb{R}^1$인 경우 다음과 같이 쓸 수 있다.


$$
 \text{Var}(\hat{\eta}_n) \geq (\frac{\partial}{\partial \theta}\mathbb{E}_{\theta}(\hat{\eta}_n))^T(nI(\theta))^{-1}(\frac{\partial}{\partial \theta}\mathbb{E}_{\theta}(\hat{\eta}_n)), \ \forall \theta \in \Omega 
$$






먼저 $\theta\in\mathbb{R}^1, \eta:\mathbb{R}^1 \rightarrow \mathbb{R}^1$인 경우 다음과 같이 보일 수 있다.

적절한 조건 하에서 추정량 $\hat{\eta}_n$과 점수함수 $l'_n(\theta)$의 공분산에 대해 다음이 성립함을 알고 있다.


$$
\begin{align*} \text{Cov}_{\theta}[\hat{\eta}_n, l'_n(\theta)] &= \mathbb{E}_{\theta}[\hat{\eta}_nl'_n(\theta)] \ (\because \mathbb{E}_{\theta}[l'_n(\theta)] = 0) \\ &= \int \hat{\eta}_n \frac{\partial }{\partial \theta}\log f(x;\theta) \cdot  f(x;\theta)dx \ \text{ where } f(x;\theta) = \prod_{i=1}^nf(x_i;\theta)\\ &= \int \hat{\eta}_n \frac{\partial f(x;\theta)/\partial \theta}{f(x;\theta)} f(x;\theta)dx = \frac{\partial}{\partial \theta}\mathbb{E}_{\theta}[\hat{\eta}_n] \end{align*}
$$




한편 상관계수의 정의 또는 코시-슈바르츠 부등식에 의해 다음이 성립한다.


$$
 \begin{align*} \text{Cov}_{\theta}(\hat{\eta}_n, l_n'(\theta))^2 &\leq \text{Var}_{\theta}(\hat{\eta}_n)\text{Var}_{\theta}(l_n'(\theta)) \\ (\frac{\partial }{\partial \theta}\mathbb{E}_{\theta}[\hat{\eta}_n])^2 &\leq  \text{Var}_{\theta}(\hat{\eta}_n)\cdot(nI(\theta)) \\ \text{Var}_{\theta}(\hat{\eta}_n) &\geq (\frac{\partial }{\partial \theta}\mathbb{E}_{\theta}[\hat{\eta}_n])^2(nI(\theta))^{-1}, \ \forall \theta \in \Omega \end{align*}
$$




이제 $\eta \in \mathbb{R}^m, \theta \in \mathbb{R}^p$의 일반적인 경우의 증명은 다음과 같다.

먼저 표현상의 편의를 위해 $\hat{\eta}=\hat{\eta}_n$이라고 하고 점수함수 벡터를 다음과 같이 나타내자.


$$
S(\theta) := (\frac{\partial}{\partial\theta_j}l(\theta))_{j=1,...,p} \in \mathbb{R}^p \text{ and } \mathbb{E}_{\theta}[S(\theta)] = 0 \in \mathbb{R}^p 
$$




이제 일차원에서와 같이 $\hat{\eta}, S(\theta)$간의 공분산은 다음과 같이 계산할 수 있다.


$$
\begin{align*} \text{Cov}_{\theta}[\hat{\eta}, S(\theta)] &= \mathbb{E}_{\theta}[(\hat{\eta}-\mathbb{E}_{\theta}[\hat{\eta}])(S(\theta) - \mathbb{E}_{\theta}[S(\theta)])^T]\\ &= \mathbb{E}[\hat{\eta}S(\theta)^T] \\ &= (\frac{\partial}{\partial \theta} \mathbb{E}[\hat{\eta}])^T \in \mathbb{R}^{m \times p}

\end{align*} 
$$




한편 다음의 결합 확률 벡터 $(\hat{\eta}, S(\theta))\in \mathbb{R}^{m+p}$의 분산행렬은 양의 준정부호이다.


$$
 \begin{pmatrix} \text{Var}_{\theta}(\hat{\eta}) & \text{Cov}_{\theta}(\hat{\eta}, S(\theta)) \\ \text{Cov}_{\theta}(\hat{\eta}, S(\theta))^T & \text{Var}_{\theta}(S(\theta)) \end{pmatrix} =

\begin{pmatrix} \text{Var}_{\theta}(\hat{\eta}) & \frac{\partial}{\partial \theta} \mathbb{E}[\hat{\eta}]^T \\ \frac{\partial}{\partial \theta} \mathbb{E}[\hat{\eta}] & nI(\theta) \end{pmatrix} \succeq 0 
$$




이에 대한 필요충분 조건은 다음과 같다.

 
$$
\text{Var}_{\theta}(\hat{\eta}) - \text{Cov}_{\theta}(\hat{\eta}, S(\theta))[\text{Var}_{\theta}(S(\theta))]^{-1}\text{Cov}_{\theta}(\hat{\eta}, S(\theta))^T \succeq 0
$$




이제 위 부등식에 앞에서 구한 공분산행렬에 대한 관계식을 대입하면 다음이 성립한다.


$$
 \text{Var}_{\theta}(\hat{\eta}) - (\frac{\partial}{\partial \theta} \mathbb{E}[\hat{\eta}])^T[nI(\theta)]^{-1} (\frac{\partial}{\partial \theta} \mathbb{E}[\hat{\eta}]) \succeq 0, \ \forall \theta \in \Omega
$$




Note. 위 정보량 부등식의 하한에서 $(nI(\theta))^{-1}$를 구하기 힘든 경우, $\zeta = g(\theta)$이고 $g:\mathbb{R}^p \rightarrow \mathbb{R}^p$가 전단사이고, 미분가능한 함수이면 다음과 같이 $\zeta$를 이용하여 구할 수 있다.

다차원 모수 $\theta=(\theta_1,...,\theta_p)^T$에 대응하는 $\zeta=(\zeta_1,...,\zeta_p)^T$에 대하여 야코비안 행렬을다음과 같이 표기하자.

$$ \frac{\partial{\zeta}}{\partial{\theta}} := (\frac{\partial{\zeta_i}}{\partial{\theta_j}})_{i,j} \in \mathbb{R}^{p \times p} $$

또한, 연쇄 법칙에 의해 다음이 성립한다.

$$ \frac{\partial f}{\partial \theta_j} = \sum_{i=1}^p\frac{\partial f}{\partial \zeta_i} \cdot \frac{\partial \zeta_i}{\partial \theta_j} = (\frac{\partial{\zeta}}{\partial{\theta}}^T \cdot \frac{\partial f}{\partial \zeta})_j, \ j=1,...,p $$

따라서, 다음의 관계식이 성립한다.


$$
 \begin{align*} \frac{\partial}{\partial \theta}f(x;\theta) &= (\frac{\partial \zeta}{\partial \theta})^T\cdot\frac{\partial }{\partial \zeta}f(x;g^{-1}(\zeta)) \\ \frac{\partial}{\partial \theta} \mathbb{E}_{\theta}[\hat{\eta}] &=(\frac{\partial \zeta}{\partial \theta})^T\cdot  \frac{\partial }{\partial \zeta} \mathbb{E}_{\zeta}[\hat{\eta}] \\ \text{Var}_{\theta}[\frac{\partial}{\partial \theta} \log f(X_1;\theta)] &= (\frac{\partial \zeta}{\partial \theta})^T\text{Var}_{\zeta}[\frac{\partial }{\partial \zeta}\log f(X_1;g^{-1}(\zeta))](\frac{\partial \zeta}{\partial \theta})

\end{align*}
$$




이제 이를 정보량 부등식의 하한에 대입하면 다음과 같다.


$$
\begin{align*} (\frac{\partial}{\partial \theta}\mathbb{E}_{\theta}(\hat{\eta}_n))^T(n \text{Var}_{\theta}[\frac{\partial}{\partial \theta}\log f(X_1;\theta)])^{-1}(\frac{\partial}{\partial \theta}\mathbb{E}_{\theta}(\hat{\eta}_n)) &= (\frac{\partial \zeta}{\partial \theta}^T  \frac{\partial }{\partial \zeta} \mathbb{E}_{\zeta}[\hat{\eta}])^T[n\frac{\partial \zeta}{\partial \theta}^T\text{Var}_{\zeta}[\frac{\partial }{\partial \zeta}\log f(X_1;g^{-1}(\zeta))]\frac{\partial \zeta}{\partial \theta}]^{-1}(\frac{\partial \zeta}{\partial \theta}^T  \frac{\partial }{\partial \zeta} \mathbb{E}_{\zeta}[\hat{\eta}])\\ &= (\frac{\partial }{\partial \zeta} \mathbb{E}_{\zeta}[\hat{\eta}])^T \frac{\partial \zeta}{\partial \theta}(\frac{\partial \zeta}{\partial \theta})^{-1}[n\text{Var}_{\zeta}[\frac{\partial }{\partial \zeta}\log f(X_1;g^{-1}(\zeta))]]^{-1} (\frac{\partial \zeta}{\partial \theta}^T)^{-1}\frac{\partial \zeta}{\partial \theta}^T(\frac{\partial }{\partial \zeta} \mathbb{E}_{\zeta}[\hat{\eta}]) \\ &= (\frac{\partial }{\partial \zeta} \mathbb{E}_{\zeta}[\hat{\eta}])^T [n\text{Var}_{\zeta}[\frac{\partial }{\partial \zeta}\log f(X_1;g^{-1}(\zeta))]]^{-1} (\frac{\partial}{\partial \zeta}\mathbb{E}_{\zeta}[\hat{\eta}]))

\end{align*} 
$$




즉, 하한은 모수의 일대일 변환에 대해 불변이다.

**Note**. $\eta(\theta) = \theta$이고,  $\theta$의 불편 추정량을 $\hat{\theta}^{\text{UE}}$라고 하면, $\hat{\eta} = \eta(\hat{\theta}^{\text{UE}}) = \hat{\theta}^{\text{UE}}$에 대한 정보량 부등식의 하한은 다음과 같이 주어진다.


$$
\begin{align*} (\frac{\partial}{\partial \theta} \mathbb{E}[\hat{\eta}])^T[nI(\theta)]^{-1} (\frac{\partial}{\partial \theta} \mathbb{E}[\hat{\eta}]) &= (\frac{\partial}{\partial \theta}\theta)^T[nI(\theta)]^{-1} (\frac{\partial}{\partial \theta} \theta) \\ &= I_p^T[nI(\theta)]^{-1}I_p = \frac{1}{n}I(\theta)^{-1}

\end{align*}
$$




즉, 다음의 부등식이 성립한다.

$$ \text{Var}_{\theta}[\sqrt{n} (\hat{\theta}^{\text{UE}} - \theta)]=n\text{Var}(\hat{\eta}) \succeq I(\theta)^{-1} $$

따라서 $\sqrt{n}(\hat{\theta} - \theta) \xrightarrow{d} \mathcal{N}(0, \sigma^2(\theta))$와 같은 점근정규성이 성립하는 모든 추정량의 극한분포의 분산 $\sigma^2(\theta)$

에 대해 항상 $\sigma^2(\theta) \succeq I(\theta)^{-1}$이 성립한다. 다시 말해서,

$$ c^T(\sigma^2(\theta) - I(\theta)^{-1})c \geq 0, \ \forall c\in \mathbb{R}^p $$

한편 $\hat{\theta}^{\text{MLE}}$가 일치성을 만족한다면, 잉여항의 적절한 처리를 통해 $\sqrt{n}(\hat{\theta}^{\text{MLE}} - \theta) \xrightarrow{d} \mathcal{N}(0,I(\theta)^{-1})$의 점근정규성이 성립함을 알고 있다. 즉 적절한 조건 하에서 최대 가능도 추정량은 점근 정규성을 갖는 추정량 중에서 가장 작은 분산을 갖는다.
