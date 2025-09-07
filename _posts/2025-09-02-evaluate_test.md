---
layout: single
title: "검정의 비교"
description: "UMP, UMPU, 비모수적 부호 검정, 검정의 점근 효율성"
categories: Statistics
tag: [UMP, UMPU, 부호순위 검정, 검정의 비교 ]
typora-root-url: ..\
author_profile: false
use_math: true
toc: true

---



모집단의 분포가 확률밀도 함수 $$f(x;\theta), \ \theta \in \Omega$$중의 하나인 랜덤표본 $$X_1,...,X_n$$을 활용하여 모수에 대한 가설을 검정해보자. 가장 일반적으로 $$\Omega_0 \cup \Omega_1 = \Omega, \ \Omega_0 \cap \Omega_1 = \emptyset$$이 되도록 귀무모수공간 $$\Omega_0$$와  대립모수공간 $$\Omega_1$$를 분할하여, 귀무가설 $$H_0:\theta \in \Omega_0$$ 와 대립가설 $$H_1: \theta \in \Omega_1$$에 대해 유의수준 유의수준 $$\alpha \in (0,1)$$에서 검정할 때, 대립가설의 각 모수 값에서의 검정력을 크게하는 검정이 좋은 검정이다.

검정 $$\phi$$에 대한 모수 $$\theta$$의 검정력을 다음과 같은 함수로 나타내자.

$$ \gamma_{\phi}(\theta) := \mathbb{E}_{\theta}[\phi(X)] $$

$$\sup_{\theta \in \Theta_0} \gamma_{\phi}(\theta) = \alpha$$를 만족하는 검정 $$\phi$$를 크기 $$\alpha$$ 검정이라고 한다.

$$\sup_{\theta \in \Theta_0} \gamma_{\phi}(\theta) \leq \alpha$$를 만족하는 검정 $$\phi$$를 유의수준 $$\alpha$$ 검정이라고 한다. 즉, 크기가 $$\alpha$$인 검정 $$\phi$$는 유의수준 $$\alpha$$검정이다.

# 전역 최강력 검정

귀무가설 $$H_0:\theta \in \Omega_0$$ 와 대립가설 $$H_1: \theta \in \Omega_1$$을 검정할 때 다음을 만족하는 검정 $$\phi^{\text{UMP}}$$를 유의수준 $$\alpha \in (0,1)$$의 전역최강력 검정이라고 한다.

1. 귀무가설 전역에서 유의수준은 $$\alpha$$이하이다. 다시말해서, $$\sup_{\theta \in \Omega_0} \gamma_{\phi^{\text{UMP}}}(\theta) \leq \alpha$$를 만족한다.
2. 대립가설 전역에서 최대의 검정력을 갖는다. 즉, 임의의 귀무가설 전역에서 유의수준 $$\alpha$$이하인 검정 $$\phi$$와 임의의 $$\theta \in \Omega_1$$에 대해 $$\gamma_{\phi}(\theta) \leq \gamma_{\phi^{\text{UMP}}}(\theta)$$를 만족한다.

예를들어, $$\mathcal{N}(\mu,1)$$에서의 랜덤표본을 이용하여 $$H_0:\mu \leq \mu_0 \text{ v.s. } H_1: \mu \geq \mu_0$$를 검정한다고 할 때, 유의수준 $$\alpha$$의 전역최강력 검정을 구해보자.

먼저 더 유도가 쉬운 귀무,대립 모수공간의 원소가 한개인 경우인 $$H_0:\mu = \mu_0 \text{ v.s. } H_1: \mu = \mu_1 (\mu_1\geq \mu_0)$$에서의 전역최강력 검정 $$\phi^{\text{UMP}}$$을 유도한다. 여기서 $$\phi^{\text{UMP}}$$는 $$\{\phi:\gamma_{\phi}(\mu_0) \leq \alpha\}$$에서 가장 큰 검정력을 갖는다. 자연스럽게, 이에 대한 부분 집합 $$\{\phi:\gamma_{\phi}(\mu)\leq \alpha, \forall\mu\leq \mu_1\}$$의 검정들과 비교해도 더 높은 검정력을 갖는다.

따라서 만약 검정력 함수 $$\gamma_{\phi^{\text{UMP}}}(\mu)$$가 $$\mu \leq \mu_0$$에서 증가함수라면,  $$\max_{\mu \leq \mu_0} \gamma_{\phi^{\text{UMP}}}(\mu) = \gamma_{\phi^{\text{UMP}}}(\mu_0)  \leq \alpha$$를 만족하여 마찬가지로 $$H_0:\mu \leq \mu_0 \text{ v.s. } H_1: \mu \geq \mu_0$$에서도 전역최강력 검정이다.

이제 $$H_0:\mu = \mu_0 \text{ v.s. } H_1: \mu = \mu_1$$에서의 $$\phi^{\text{UMP}}$$를 구해보자. $$X=(X_1,...,X_n)^T$$의 support를 $$\mathcal{X}$$라고 하고, $$\mu$$에 대한 pdf를 $$f(x;\mu)$$라고 하자.

$$ f(x;\mu) = \frac{1}{\sqrt{2\pi}} \exp[-\frac{\sum_{i=1}^n(x_i-\bar{x})^2 - n(\bar{x}-\mu)^2}{2}] $$

$$\phi^{\text{UMP}}$$는 다음을 만족하는 해이다.

$$ \max_{\phi:\mathcal{X} \rightarrow [0,1]} \gamma_{\phi}(\mu_1) \text{ subject to }\gamma_{\phi}(\mu_0) \leq \alpha $$

라그랑주 승수 $$k \geq 0$$를 통한 쌍대문제는 다음과 같다. $$\phi$$는 볼록집합 $$[0,1]$$에서 값을 갖고, 검정력함수 또한 $$\phi$$에 대해 선형임으로, 이는 볼록최적화 문제로 strong duality가 성립한다. 즉, 다음의 해가 UMP이기도하다.

$$ \begin{align*} &\max_{\phi:\mathcal{X} \rightarrow [0,1]} \gamma_{\phi}(\mu_1) - k(\gamma_{\phi}(\mu_0) -\alpha) \\ &= \max_{\phi:\mathcal{X} \rightarrow [0,1]} \gamma_{\phi}(\mu_1) - k\gamma_{\phi}(\mu_0)

\end{align*} $$

최대화해야하는 함수를 pdf를 통해 다시 쓰면 다음과 같다.


$$
\begin{align*} \gamma_{\phi}(\theta_1) - k\gamma_{\phi}(\theta_0) &= \mathbb{E}_{\theta_1}[\phi(X)I(f(x;\mu_0)=0)] + \mathbb{E}_{\theta_0}[\phi(X)(\frac{f(x;\mu_1)}{f(x;\mu_0)}-k)] \end{align*}
$$




$$\phi$$는 0과 1사이의 값을 갖으므로, 이를 최대화하는 $$\phi^{\text{UMP}}$$는 다음과 같다.

$$ \begin{align*} \phi^{\text{UMP}}(x) &= \begin{cases} 1 &\text{ if }  f(x;\mu_1) >kf(x;\mu_0) \\ 0 &\text{ if }  f(x;\mu_1) <kf(x;\mu_0)\end{cases}  \\ &=I[\frac{f(x;\mu_1)}{f(x;\mu_0)}\geq k] \end{align*} $$

Note. 이는 최대가능도비 검정의 기각역의 형태임을 알 수 있다.

이제, 위의 가능도비를 풀어쓰면 다음과 같다.

$$ \begin{align*} \frac{f(x;\mu_1)}{f(x;\mu_0)} &= \exp[-\frac{n}{2}\{(\bar{x}-\mu_1)^2 - (\bar{x}-\mu_0)^2\}] \\ &= \exp[-\frac{n}{2}(-2\bar{x}(\mu_1-\mu_0)+\mu_1^2-\mu_0^2)] \\ &= \exp[\frac{n}{2}(\mu_1-\mu_0) (2\bar{x}-\mu_1-\mu_0)] \end{align*} $$

$$\mu_1 \geq \mu_0$$임으로 위 가능도비는 $$\bar{x}$$에 대한 증가함수이다. 즉, $$\bar{x}$$에 대해 $$\phi^{\text{UMP}}$$를 $$I[\bar{X}\geq c]$$로 다시 쓸 수 있다. 이제 유의수준 $$\alpha$$를 만족하도록 다음과 같이 $$c$$를 구한다.


$$
\begin{align*} \gamma_{\phi^{\text{UMP}}}(\mu_0) &= \mathbb{E}_{\bar{X} \sim \mathcal{N}(\mu_0, \frac{1}{n})}[I(\bar{X}>c)] \\ &= Pr(Z>\sqrt{n}(c-\mu_0)) = \alpha \\ \therefore c&= \frac{z_{\alpha}}{\sqrt{n}}+\mu_0 \end{align*}
$$




Note. 기각역이 대립가설의 모수에 의존하지 않는다.

또한 검정력 함수는 $$\gamma_{\phi^{\text{UMP}}}(\mu) = 1-\Phi(\sqrt{n}(c-\mu))$$로 주어짐으로 $$\mu$$에 대한 증가함수이다. 그러므로,  $$H_0:\mu \leq \mu_0 \text{ v.s. } H_1: \mu \geq \mu_0$$에 대한 전역최강력 검정에 대한 기각역은 아래와 같이 주어진다.

$$ \sqrt{n}(\bar{X} - \mu_0) \geq z_{\alpha} $$

더 나아가, $$H_0:\mu = \mu_0 \text{ v.s. } H_1: \mu \neq \mu_0$$를 유의수준 $$\alpha$$로 검정할 때 전역 최강력 검정이 존재하지 않음을 알 수 있다. 만약 그러한 검정이 존재한다면, $$\mu_1 < \mu_0 < \mu_2$$를 만족하는 $$\mu_1,\mu_2$$에 대한 대립가설 $$(\mu=\mu_1),(\mu=\mu_2)$$에 대해 유의수준 $$\alpha$$의 전역 최강력 검정을 만족해야한다.

즉, 기각역이 $$\bar{X}  \leq c_1, \bar{X} \geq c_2$$를 동시에 만족하는 꼴을 가져야 함으로, 이는 모순이다.

# 네이만-피어슨 보조정리 (단순 가설의 전역 최강력 검정의 존재성)

위의 예시에서 분산이 알려진 정규모집단에 대해 단순 가설  $$H_0:\theta=\theta_0 \text{ v.s. } H_1:\theta = \theta_1$$꼴의 유의수준 $$\alpha$$에 대한 가설검정에 대한 UMP는 다음과 같은 기각역이 주어짐을 보였다.

$$ \begin{align*} \phi^{*}(x)  &=I[\frac{f(x;\theta_1)}{f(x;\theta_0)}> k] \text{ where } k\geq 0 \text { satisfying }\gamma_{\phi^*}(\theta_0) = \alpha \end{align*} $$

(이산형인 경우, 등호가 성립하는 경우에 대한 고려를 위해 $$0<\gamma<1$$을 사용하여 $$\phi^{*}(x)  =I[\frac{f(x;\theta_1)}{f(x;\theta_0)}\geq k]+ \gamma I[\frac{f(x;\theta_1)}{f(x;\theta_0)}\geq k]$$이고, $$\gamma_{\phi^*}(\theta_0)=\alpha$$를 만족하는 검정)

또한, 이를 보이는 과정에서 정규분포에 대한 성질을 사용하지 않았음으로 일반적인 단일 모수를 갖는 $$f(x;\theta)$$에 대해서도 이가 성립함을 보일 수 있다. 사실, 이는 유의수준 $$\alpha$$의 전역최강력검정이기 위한 필요충분 조건이다.

1. 충분조건: $$\phi^*$$는 유의수준 $$\alpha$$의 전역 최강력 검정(UMP)이다.
2. 필요조건: $$\alpha \in (0,1)$$에 대해 $$k>0$$이면서 위를 만족하는 $$\phi^*$$가 존재한다면, 모든 유의수준 $$\alpha$$의 전역 최강력 검정 $$\phi^{\text{UMP}}$$에 대해 $$\gamma_{\phi^{\text{UMP}}}(\theta_0) = \alpha$$를 만족한다. 또한, 모든 유의수준 $$\alpha$$의 전역 최강력 검정 $$\phi^{\text{UMP}}$$는 $$\mathcal{A}=\cup\{A:Pr_{\theta_0}(X\in A) = Pr_{\theta_1}(X\in A)=0\}$$을 제외하고 $$\phi^{\text{UMP}}=\phi^{*}(x)$$ 이다. $$\phi^{\text{UMP}}=\phi^{*}(x) \ a.s.$$

연속형인 경우에 대해서만 보여보자.

먼저 충분조건에 대해서 증명해보자. 임의의 유의수준 $$\alpha$$인 검정 $$\phi$$에 대해 다음과 같은 함수를 정의하자.

$$ G(\phi) = (\gamma_{\phi^*}(\theta_1) - k\gamma_{\phi^*}(\theta_0))-(\gamma_{\phi}(\theta_1) - k\gamma_{\phi}(\theta_0)) $$

만약 $$G(\phi)\geq 0, \ \forall \phi$$를 만족한다면, 다음과 같이 $$\phi^*$$가 유의수준 $$\alpha$$의 전역 최강력 검정이다.

$$ \gamma_{\phi^*}(\theta_1) - \gamma_{\phi}(\theta_1) \ge  k(\gamma_{\phi^*}(\theta_0) - \gamma_{\phi}(\theta_0)) = k(\alpha-\gamma_{\phi}(\theta_0))\geq0 $$

이제 $$G$$가 항상 음이 아님을 보이자.


$$
\begin{align*} G(\phi) &= \mathbb{E}_{\theta_1}[(\phi^*-\phi)] - k\mathbb{E}_{\theta_0}[(\phi^*-\phi)] \\ &= \mathbb{E}_{\theta_1}[(\phi^*-\phi)I[f(x;\theta_0)=0]] + \mathbb{E}_{\theta_0}[(\phi^*-\phi)\{\frac{f(x;\theta_1)}{f(x;\theta_0)}-k\}] \end{align*}
$$




여기서 첫번째 항은 $$f(x;\theta_0)=0$$이면, 가능도비의 분모가 0이 되기에 $$\phi^*=1$$이 되고, $$0\leq \phi \leq 1$$임으로, 항상 음이 아니다. 또한 두번째 항에 대해서도 가능도비가 $$k$$보다 크면 $$(\phi^*-\phi)=(1-\phi)$$로 항상 음이 아니고, 반대는 $$-\phi$$로 항상 양이 아님으로, $$G(\phi)$$는 항상 0보다 크거나 같다.

이제 필요조건에 대해서 보여보자. 임의의 유의수준 $$\alpha$$의 전역 최강력 검정 $$\phi^{\text{UMP}}$$에 대해서도 $$G(\phi^{\text{UMP}}) \geq 0$$을 만족해야 한다. 또한 충분조건에 의해 검정력이 $$\gamma_{\phi^*}(\theta_1) =\gamma_{\phi^{\text{UMP}}}(\theta_1)$$ 를 만족해야 함으로 $$k(\alpha -\gamma_{\phi^{\text{UMP}}}(\theta_0))=0$$이어야 하고, 가정에 의해 $$k>0$$임으로 $$\gamma_{\phi^{\text{UMP}}}(\theta_0)=\alpha$$를 만족해야 한다. 즉, $$G(\phi^{\text{UMP}})=0$$이다. 따라서 $$\mathcal{A}$$를 제외하고 다음을 만족해야 한다.

$$ \begin{align*} 0 &=\mathbb{E}_{\theta_0}[(\phi^*-\phi^{\text{UMP}})\{\frac{f(x;\theta_1)}{f(x;\theta_0)}-k\}]

\end{align*} $$

위가 0이 되려면 피적분함수가 $$\mathcal{A}$$를 제외하고 0이여야 한다. 즉, 가능도비가 $$k$$보다 크면 $$\phi^{\text{UMP}}=1$$을, $$k$$보다 작으면 $$\phi^{\text{UMP}}=0$$을 만족해야 한다.

Note. $$\theta$$에 대한 충분통계량을 $$T(x)$$이고 확률밀도함수가 $$g(t;\theta)$$로 주어진다고 하자. 분해정리에 의해 $$f(x;\theta) = g(t;\theta)h(x)$$가 성립한다. 즉, 위 가능도비 꼴의 검정은 $$g(t;\theta_1)/g(t;\theta_0) > k$$로 충분통계량에 대한 기각역을 활용하여 다시 쓸 수 있다.

Note. 일반적으로 UMP의 존재성을 보이는데에는 위 보조정리의 충분조건을, UMP가 존재하지 않음을 보일 때에는 필요조건을 활용한다.

# 칼린-루빈 정리 (전역 최강력 한쪽 검정)

$$T \sim g(t;\theta), \theta \in \Omega$$라고 할 때, 임의의 $$\theta_0 < \theta_1$$에 대하여, 다음을 만족하면 모수 $$\theta$$에 대해 단조가능도비(MRT)를 만족한다고 한다.

$$ \forall t \in \{t:g(t;\theta_0)>0 \text{ or } g(t;\theta_1)>0 \}, \ \frac{g(t;\theta_1)}{g(t;\theta_0)} \text{ is monotone} $$

Note. 단일 모수 지수족 $$f(x;\theta) = \exp[g(\theta)T(x) - B(\theta) + S(x)\}, \theta \in \Omega \subset \mathbb{R}$$로부터의 랜뎜표본 $$X_1,..,X_n$$에 대하여  $$\theta_0<\theta_1$$를 고정하면 가능도비는 다음과 같다.

$$ \frac{f(x;\theta_1)}{f(x;\theta_0)}  = \exp[(g(\theta_1)-g(\theta_0))\sum_{i=1}^nT(x_i)-nB(\theta_1)+nB(\theta_0)] $$

즉, $$g(\theta)$$가 단조 증가함수이면, 충분통계량 $$\sum_{i=1}^nT(x_i)$$에 대한 단조 증가함수이다. 즉, 네이만 피어슨 보조정리의 필요조건에 의해 가설 $$H_0:\theta= \theta_0 \text{ v.s. } H_1:\theta = \theta_1$$에 대한 유의수준 $$\alpha$$의 전역최강력 검정은 $$\phi(x)=\sum_{i=1}^nT(x_i) > c$$의 꼴로 주어지며 $$c$$는 $$\gamma_{\phi}(\theta_0)=\alpha$$를 만족한다. 이는 모수 $$\theta_1$$의 값에 관계없이 $$\theta_0$$보다 크기만 하면 성림함으로, 이는  가설 $$H_0:\theta= \theta_0 \text{ v.s. } H_1:\theta > \theta_0$$에 대한 유의수준 $$\alpha$$의 전역최강력 검정이기도 하다.

Note. 만약 $$g(\theta)$$가 단조 감소한다면, 위 가능도비 꼴의 검정은 $$\phi(x)= \sum_{i=1}^nT(x_i)<c$$로 주어지고 이는 가설 $$H_0:\theta= \theta_0 \text{ v.s. } H_1:\theta > \theta_0$$에 대한 유의수준 $$\alpha$$의 전역최강력 검정이다.

가설 $$H_0: \theta \leq \theta_0 \text{ v.s. } H_1: \theta > \theta_0$$을 유의수준 $$\alpha \in(0,1)$$로 검정한다고 하자. $$T$$를 $$\theta \in \Omega$$에 대한 충분통계량이라고 하고, $$T$$의 확률밀도함수가 모수 $$\theta$$에 대한 함수 $$g(t;\theta)$$로 나타낼 수 있고, $$\theta$$에 대해 단조 우도비(MLR)을 갖는다고 하자. 이 때 다음을 만족하는 검정 $$\phi^*$$는 유의수준 $$\alpha$$의 전역 최강력 검정(UMP)이다.

$$ \phi^*(t) = I[t>c] \text{ where } \gamma_{\phi^*}(\theta_0) = \alpha $$

증명은 네이만-피어슨 보조정리를 사용해 단순가설에서의 UMP 존재성을 보이고 이러한 검정이 가능도비의 단조성에 의해 한쪽검정에서의 UMP이기도 함을 보일것이다.

$$\theta_1 >\theta_0$$을 만족하는 $$\theta_1$$를 하나 고정하면, MRT가정에 의해 $$g(t;\theta_1)/g(t;\theta_0)$$은 $$t$$에 대한 단조함수이다. 다시 말해서 일대일 대응이므로 $$\phi^*$$는 가능도비 검정 꼴로 다시 쓸 수 있다.

만약 $$T$$의 가능도비가 $$t$$에 대해 단조증가한다면, 다음과 같이 가능도비꼴의 검정과 동치이다.

$$ (t'>c) \Leftrightarrow (\frac{g(t';\theta_1)}{g(t';\theta_0) } >k') \text{ where }  \inf_{t > c}\frac{g(t;\theta_1)}{g(t;\theta_0) } $$

가능도비가 단조감소할 경우에는 $$-T$$를 이용하여, 위와 같이 만들 수 있다.

따라서, 네이만-피어슨 보조정리에 의해 단순가설 $$H_0:\theta=\theta_0 \text{ v.s. } H_1:\theta =\theta_1$$의 유의 수준 $$\alpha$$에 대한 전역최강력 검정이다. 또한, $$\phi^*$$는 $$\theta_1$$에 의존하지 않기에, 이는 $$H_0:\theta=\theta_0 \text{ v.s. } H_1:\theta >\theta_0$$에 대한 전역 최강력 검정이다.

Remark. $$\theta_1$$은 기각역의 방향만을 알려준다. 즉, $$\theta_1 < \theta_0$$이었다면, UMP는 $$t<c$$의 꼴로 주어질 것이다. 더 나아가, $$H_0:\theta=\theta_0 \text{ v.s. } H_1:\theta  \neq\theta_0$$에 대한 전역 최강력 검정이기도 할까? 만약 그러한 검정 $$\phi^*$$가 존재한다면, 이는 $$\theta_1 < \theta_0 < \theta_2$$에 대해 단순가설 $$H_0:\theta=\theta_0 \text{ v.s. } H_1:\theta =\theta_1, \ H_0:\theta=\theta_0 \text{ v.s. } H_1:\theta =\theta_2$$ 각각에 대해서도 UMP여야 한다. 네이만-피어슨 보조정리의 필요조건에 의하면, 각각의 단순가설의 기각역은 $$(t<c_1), (t>c_2)$$꼴이어야 하고, 이 둘을 만족하는것은 불가능함으로 UMP는 존재하지 않는다.

한편 이러한 $$\phi^*$$에 대한 검정력 함수 $$\gamma_{\phi^*}(\theta)$$가 $$\theta$$에 대한 단조증가함수임을 밝히면, $$\sup_{\theta \leq \theta_0} \gamma_{\phi^*}(\theta) = \gamma_{\phi^*}(\theta_0)=\alpha$$를 만족함으로 $$H_0:\theta \leq \theta_0$$에 대한 유의 수준 $$\alpha$$인 검정을 만족한다.

또한 검정력을 비교하는 검정 집합이 $$H_0:\theta = \theta_0$$인 가설이 $$H_0:\theta \leq \theta_0$$를 포함한다.

$$ \{ \phi: \sup_{\theta \leq \theta_0} \gamma_{\phi}(\theta) \leq \alpha \} \subset \{\phi:\gamma_{\phi}(\theta_0)\leq \alpha\} $$

그러므로, 가설 $$H_0: \theta \leq \theta_0 \text{ v.s. } H_1: \theta > \theta_0$$을 검정할 때에도 유의수준 $$\alpha$$의 전역 최강력 검정이다.

이제 $$\gamma_{\phi^*}(\theta)$$가 단조증가함수임을 보이자. 임의의 $$\theta' < \theta{''}$$를 하나 정하여, 가설 $$H_0:\theta=\theta^{'} \text{ v.s. } H_1:\theta  =\theta^{''}$$에 대한 유의 수준 $$\alpha' = \gamma_{\phi^*}(\theta')$$에 대한 UMP를 $$\phi^{\text{UMP}}*{\alpha^{'}}$$라고 하자. 네이만 피어슨 보조정리의 필요조건에 의해 $$\gamma*{\phi^{\text{UMP}}*{\alpha^{'}}}(\theta^{'}) = \alpha' = \gamma*{\phi^*}(\theta')$$임으로,  $$\cup\{A:\int_{A}g(t;\theta^{'})=0 \text{ or }\int_{A}g(t;\theta^{''})=0 \}$$를 제외하고 검정 $$\phi^*$$와 일치한다.

한편, 랜덤표본과 독립으로 항상 $$\alpha'$$의 확률로 기각하는 검정을 $$\phi^{'}$$라고 하면, 이또한 유의수준 $$\alpha'$$ 검정이며, 검정력은 $$\alpha'$$이기에, $$\alpha ' \leq \gamma_{\phi^{\text{UMP}}_{\alpha^{'}}}(\theta^{''})$$를 만족한다. 따라서 다음이 성립한다.


$$
\gamma_{\phi^*}(\theta')= \alpha'\leq \gamma_{\phi^{\text{UMP}}_{\alpha^{'}}}(\theta^{''})=\gamma_{\phi^*}(\theta^{''})
$$




즉, $$\gamma_{\phi^*}(\theta)$$는 $$\theta$$에 대한 단조증가함수이다.



# 균등분포에서의 양쪽검정에 대한 UMP 존재성

$$U[0,\theta]$$으로부터의 랜덤표본을 통해 $H_0:\theta=\theta_0 \text{ v.s. } H_1: \theta \neq \theta_0$에 대한 유의수준 $\alpha\in(0,1)$의 검정을 고려해보자.

먼저, $\theta_1 < \theta_0 < \theta_2$을 만족하는 $(\theta_1,\theta_2)$를 고정하여, 각각의 단순가설에 대한 UMP의 형태를 알아보자.

$H_1:\theta = \theta_1$의 단순 가설의 경우, 가능도비는 다음과 같이 주어진다.

$$ (\frac{\theta_0}{\theta_1})^n \cdot  \frac{I[x_{(n)} \leq \theta_1]}{I[x_{(n)} \leq \theta_0]} = \begin{cases} (\frac{\theta_0}{\theta_1})^n &\text{if } x_{(n)} \leq \theta_1 \\ 0 &\text{if } x_{(n)} > \theta_1  \end{cases} \text{ where } x_{(1)} \leq ... \leq x_{(n)} $$

즉, 네이만 피어슨 보조정리의 필요조건에 의해 UMP는 $X_{(n)} \leq c$의 꼴의 기각역이 주어지며, 다음을 만족해야 한다.

$$ Pr_{\theta_0}[X_{(n)} \leq c] = (\frac{c}{\theta_0})^n = \alpha $$

즉, $\phi^*(x) = I[x_{(n)} \leq \alpha^{1/n}\theta_0]$은 해당 단순가설에 대한 UMP이다. 또한, 이에 대한 검정력은 $\theta_1$에 의존하지 않음으로, $H_1: \theta < \theta_0$에 대한 검정의 UMP이기도 하다.

$H_1:\theta = \theta_2$의 단순 가설의 경우, 가능도비는 다음과 같이 주어진다.

$$ (\frac{\theta_0}{\theta_2})^n \cdot  \frac{I[x_{(n)} \leq \theta_2]}{I[x_{(n)} \leq \theta_0]} = \begin{cases} (\frac{\theta_0}{\theta_2})^n &\text{if } x_{(n)} \leq \theta_0 \\ \infty &\text{if } \theta_0 < x_{(n)} \leq \theta_2
 \\ 0 &\text{if } x_{(n)} > \theta_2
 \end{cases} \text{ where } x_{(1)} \leq ... \leq x_{(n)} $$

이 경우, 검정 $\phi^{*}(x)$는 마찬가지로 유의수준 $\alpha$ 검정이며 $$\phi^{**}(x) = I[x_{(n)} \leq \alpha^{1/n}\theta_0 \text{ or } x_{(n)} > \theta_0]$$ 또한 마찬가지이다.  $\gamma_{\phi^{**}}(\theta)$는 $\theta>\theta_0$에서 $Pr_{\theta}[X_{(n)} > \theta_0]$만큼이 더해지기에 검정력의 향상이 있다.

$$ \begin{align*} \gamma_{\phi^{**}}(\theta) &= Pr_{\theta}[X_{(n)} \leq \alpha^{1/n}\theta_0] + Pr_{\theta_2}[X_{(n)} > \theta_0] \\ &= \alpha (\frac{\theta_0}{\theta})^n + 1 -  (\frac{\theta_0}{\theta})^n, \   \forall \theta > \theta_0 \end{align*} $$

더 나아가 $$\phi^{**}$$는 가설 $$H_1:\theta < \theta_0$$에서는 $$\gamma_{\phi^{**}} =\gamma_{\phi^*}$$이기에 대립가설 $H_0:\theta <\theta_0$에 대한 UMP이다.

이제 대립가설 $H_1: \theta = \theta_2$에서 유의수준 $\alpha$를 만족하는 임의의 검정 $\phi$에 대해 다음이 성립한다.


$$
\begin{align*} \gamma_{\phi}(\theta_2) &= \mathbb{E}_{\theta_2}[\phi(X)] \\ &= \mathbb{E}_{\theta_0}[\phi(X)\frac{pdf_{X_{(n)}}(x;\theta_2)}{pdf_{X_{(n)}}(x;\theta_0)}] + \mathbb{E}_{\theta_2}[\phi(X)I[pdf_{X(n)}(x;\theta_0)=0] ]\\ &= \mathbb{E}_{\theta_0}[\phi(X) (\frac{\theta_0}{\theta_2})^nI[X_{(n)} \leq \theta_0] ]+ \mathbb{E}_{\theta_2}[\phi(X)I[\theta_0 <X_{(n)} \leq \theta_2]] \\ &\leq \mathbb{E}_{\theta_0}[\phi(X) (\frac{\theta_0}{\theta_2})^n] +\mathbb{E}_{\theta_2}[I[\theta_0 <X_{(n)} \leq \theta_2]] \\ &= \alpha (\frac{\theta_0}{\theta_2})^n +Pr_{\theta_2}[X_{(n)} > \theta_0] = \gamma_{\phi^{**}}(\theta_2)

\end{align*}
$$




즉, $\phi^{**}$는  대립가설 $H_1: \theta = \theta_2$에서 유의수준 $\alpha$ 검정에 대한 UMP이고, 이에 대한 검정력은 $\theta_2$에 의존하지 않으므로,  대립가설 $H_0:\theta >\theta_0$에 대한 UMP이다.

그러므로, $\phi^{**}$는 양쪽 검정에 대한 UMP이다.



# 전역 최강력 불편검정

단일 모수 지수족 $$f(x;\theta) = \exp[g(\theta)T(x)-A(\theta)+B(x)]$$으로부터의 $$n$$개의 랜덤표본을 고려하자. 여기서 $$g(\theta)$$는 증가함수이고, 미분가능하다고 하자.

양쪽검정에 전역최걍력 검정이 존재하지 않는 경우, 대안으로 유의수준 $$\alpha$$인 양쪽 검정에서 가능한 검정의 종류를 다음을 만족하는 검정으로 제한한다.

$$ \gamma_{\phi}(\theta_0) = \alpha, \ \frac{\partial}{\partial\theta}\gamma_{\phi}(\theta)\mid_{\theta=\theta_0}=0 $$

즉, 유의수준은 $$\alpha$$이면서 귀무가설 $$\theta =\theta_0$$에서 검정력 함수는 국소적 최솟값을 갖는다. 이를 만족하는 검정을 유의수준 $$\alpha$$의 불편검정이라고 하며, 이러한 불편 검정중에 대립가설 하의 검정력을 가장 크게 하는 검정을  유의수준 $$\alpha$$의 전역최강력 불편검정(UMPU)라고 한다.

단일 모수 지수족에서 UMPU의  기각역의 꼴은 다음과 같이 주어진다.

$$ \sum_{i=1}^nT(x_i)<c_1 \text{ or } \sum_{i=1}^nT(x_i)>c_2 $$

이는 다음과 같이 보일 수 있다.

상수 $$\theta_1 \neq \theta_0,k_1,k_2$$을 고정하여 다음의 최적화 문제를 고려한다.


$$
\begin{align*} g(\phi;\theta_1,k_1,k_2)&=\gamma_{\phi}(\theta_1) - k_1\gamma_{\phi}(\theta_0) -k_2 \gamma'_{\phi}(\theta_0)\\ &= \mathbb{E}_{\theta_0}[\phi(X)\{\frac{f(x;\theta_1)}{f(x;\theta_0)}-k_1-k_2\frac{f'(x;\theta_0)}{f(x;\theta_0)}\}] \end{align*}
$$




이를 최대화 하는 $$\phi^* = I[f(x;\theta_1)/f(x;\theta_0)\geq k_1 +k_2f'(x;\theta_0)/f(x;\theta_0)]$$이 유의수준 $$\alpha$$의 불편성을 만족하도록 $$k_1,k_2$$를 정한다. 제약조건이 2개임으로 이는 유일하게 결정된다.

즉, 임의의 $$\phi$$에 대해 $$g(\phi^*)\geq g(\phi)$$임으로 다음의 부등식이 성립한다.

$$ \gamma_{\phi^*}(\theta_1) - \gamma_{\phi}(\theta_1) \geq k_1(\alpha - \gamma_{\phi}(\theta_0)) + k_2 \gamma'_{\phi}(\theta_0) $$

즉, $$\phi^*$$는 $$\gamma_{\phi}(\theta_0) = \alpha, \ \frac{\partial}{\partial\theta}\gamma_{\phi}(\theta)\mid_{\theta=\theta_0}=0$$를 만족하는 $$\phi$$에 대해 $$\theta_1$$에서 가장 큰 검정력을 갖는다. 이제 $$\phi^*$$에 단일모수지수족을 대입하여 $$T(X_i)$$에 관한 항만 확인하면 다음과 같다.

$$ \begin{align*} \frac{f(x;\theta_1)}{f(x;\theta_0)} &= \exp[(g(\theta_1)-g(\theta_0))\sum_{i=1}^nT(x_i) +C_1] \\ \frac{f'(x;\theta_0)}{f(x;\theta_0)} &= g'(\theta_0)\sum_{i=1}^nT(x_i)+C_2 \end{align*} $$

$$g(\theta)$$는 증가함수임을 가정하였음으로, 기각역은 다음과 같은 꼴이다.

$$ \exp(a\sum_{i=1}^n T(x_i)) - b + c\sum_{i=1}^nT(x_i) \geq 0 \text{ where } a \neq 0 $$

좌변에 대해 $$T=\sum_{i=1}^nT(x_i)$$에 대한 이계도 함수는 $$a^2\exp(aT)  >0$$로 볼록함수 꼴이므로 $$c_1 < c_2$$에 대해 $$\sum_{i=1}^nT(x_i) \leq c_1$$ 또는 $$\sum_{i=1}^nT(x_i) \geq c_2$$ 꼴의 기각역을 갖고, 이는 $$\theta_1$$에 의존하지 않는다.

따라서, UMPU $$\phi^{\text{UMPU}}$$는 다음과 같이 주어진다.

$$ \begin{align*} \phi^{\text{UMPU}}(x) &= I[\sum_{i=1}^nT(x_i) \leq c_1 \text{ or } \sum_{i=1}^nT(x_i) \geq c_2 ] \text{ where } c_1,c_2 \text{ satisfy } \\ &\gamma_{\phi^{\text{UMPU}}}(\theta_0)=\alpha, \ \frac{\partial}{\partial \theta}\gamma_{\phi^{\text{UMPU}}}(\theta)\mid_{\theta=\theta_0}=0

\end{align*} $$

# 비모수적 검정

모집단의 분포가 확률밀도 함수 $$f(x;\theta), \ \theta \in \Omega$$중의 하나인 랜덤표본 $$X_1,...,X_n$$을 활용하여 모수에 대한 가설에서 $$f(x;\theta)$$를 특정한 형태를 가정하지 않고 검정을 수행해볼 수 있다.

여기서는 위치모수 모형으로 확률밀도함수가 모수 $$\theta$$에 대해 대칭이고 누적확률밀도함수 $$F$$가 순증가함수인 확률모형에 대한 위치모수 $$\mu$$의 비모수적 검정을 알아보자.

## 한쪽 검정에서의 부호 검정

가설 $$H_0:\theta=\theta_0 \text{ v.s. } H_1:\theta > \theta_0$$의 유의수준 $$\alpha \in (0,1)$$에 대한 검정으로 다음의 검정통계량 $$S_n$$을 이용한 검정 $$\phi_S$$를 고려해보자.

$$ \begin{align*} S_n &= \sum_{i=1}^nI[X_i-\theta>\theta_0-\theta] \\ &= \sum_{i=1}^nI[Z_i>\theta_0-\theta] \text{ where } Z_i \overset{d}{=}X_i-\theta \\ &= \#\{1\leq i \leq n:Z_i>\theta_0-\theta\} \end{align*} $$

즉, $$Z_i$$는 0에 대해 대칭이고 이의 누적밀도함수를 $$F$$라고 하자. $$S_n$$은 서로 독립인 $$n$$번의 $$\mathbb{E}[I[Z_i>\theta_0-\theta]] = 1-F(\theta_0-\theta)$$의 성공확률을 갖는 베르누이 시행에 대한 성공횟수임으로 $$Bin(n,1-F(\theta_0-\theta))$$를 따른다. 즉, 귀무가설 하에서 $$S_n \sim Bin(n, 1/2)$$를 따른다.

$$\theta_0-\theta <0$$인 대립가설에 대해 $$1-F(-\infty) = 1$$임으로 대립가설에 가까울수록 $$S_n$$은 큰 값을 갖기에, 기각역의 방향은 $$S_n>c$$의 꼴로 주어진다.

따라서, 검정 $$\phi_S$$가 다음을 만족하면 유의수준 $$\alpha$$의 검정이다.

$$ \begin{align*} \phi_S &:= I[S_n > c] + \gamma I[S_n=c]  \\\text{ where } \gamma_{\phi_S}(\theta_0) &= [\sum_{k=c+1}^n \binom{n}{k} + \gamma \binom{n}{c} ]()^n = \alpha, \ 0<\gamma<1

\end{align*} $$

또한 검정력 함수는 $$Z_i$$에 대한 증가함수임으로, 이는 $$\sup_{\theta\leq \theta_0}\gamma_{\phi_S}(\theta) = \gamma_{\phi_S}(\theta_0)=\alpha$$이다. 즉, $$\phi_S$$는 가설  $$H_0:\theta\leq \theta_0 \text{ v.s. } H_1:\theta > \theta_0$$에 대한 유의수준 $$\alpha \in (0,1)$$에 대한 검정이기도 하다.

Note. 중심극한정리에 의해 $$S_n/n$$은 평균이 $$p(\theta)=1-F(\theta_0-\theta)$$이고, 분산이 $$p(\theta)(1-p(\theta))/n$$인 정규분포로 수렴한다.

$$ \frac{S_n-np(\theta)}{\sqrt{np(\theta)(1-p(\theta))}} \overset{d}{\rightarrow} \mathcal{N}(0,1) $$

즉, 충분히 큰 랜덤 표본에서의 검정에 대해 임계값은 $$c \approx z_{\alpha}np(\theta_0) + \sqrt{np(\theta_0)(1-p(\theta_0)}$$으로 근사된다.

## 한쪽 검정에서의 부호 순위 검정

가설 $$H_0:\theta=\theta_0 \text{ v.s. } H_1:\theta > \theta_0$$의 유의수준 $$\alpha \in (0,1)$$에 대한 검정으로 검정통계량은 귀무가설 하에서 0에 대칭인 $$Z_i \overset{d}{=}X_i-\theta_0$$을 이용하여 다음과 같이 주어진다.

$$ \begin{align*} W_n &= \sum_{i=1}^n sgn(Z_i)R(|Z_i|) \text{ where } R(|Z_i|) = \sum_{j=1}^nI[|Z_j| \leq |Z_i|] \\ W_n^+ &= \sum_{i=1}^n I[Z_i>0]R(|Z_i|)

\end{align*} $$

위 검정통계량을 이용한 대립가설 $$H_1:\theta>\theta_0$$의 기각역은 $$W_n^+>c^+(W_n>c)$$의 꼴로 주어짐을 알 수 있다.

먼저 $$W_n$$가 귀무가설하에서 어떤 분포를 따르는지 알아보자.  $R(\|Z_i\|)$는 $$Z_i$$을 크기순으로 정렬한 순위임으로 이러한 순위를 $$j$$라고 하면 해당 통계량의 인덱스 $$i$$와의 일대일 매핑인 $$R$$이 존재한다. 부호 부분도 순위 $$j$$를 이용하여 $$W_n$$을 나타내면 $$W_n  = \sum_{j=1}^n sgn(Z_{R^{-1}(j)})j$$이다.

또한 귀무가설 하에서 $$Z_i$$는 0을 기준으로 대칭임으로 다음이 성립한다.

$$ \begin{align*} Pr_{\theta_0}(|Z_i|\leq x, sgn(Z_i)=1) &= Pr(0\leq Z_i \leq x) \\ &= \frac{1}{2}Pr(|Z_i|\leq x) \\ &= Pr(sgn(Z_i)=1)Pr(|Z_i| \leq x)

\end{align*} $$

즉 $$(sgn(Z_i))_{i=1}^n$$과 $(\|Z_i\|)_{i=1}^n$이 독립이다. 따라서 귀무가설 하에서  $$(S_{R^{-1}(j)})_{i=1}^n:=(sgn(Z_{R^{-1}(j)}))_{i=1}^n$$에 대해 이러한 독립성과 총확률의 법칙을 사용하면 다음이 성립한다.

$$ \begin{align*} Pr_{\theta_0}(S_{R^{-1}(1)} = s_1,..., S_{R^{-1}(n)} = s_n) &= \sum_{\pi \in \Pi} Pr_{\theta_0}(S_{R^{-1}(1)} = s_1,..., S_{R^{-1}(n)} = s_n, R=\pi) \\ &= \sum_{\pi \in \Pi} Pr_{\theta_0}(S_{\pi^{-1}(1)} = s_1,..., S_{\pi^{-1}(n)} = s_n)Pr_{\theta_0}(R=\pi) \\ &=  Pr_{\theta_0}(S_{1} = s_1,..., S_{n} = s_n)\sum_{\pi \in \Pi} Pr_{\theta_0}(R=\pi) \\ &= Pr_{\theta_0}(S_1=s_1,...,S_n=s_n)

\end{align*} $$

즉, 귀무가설 하에서  $$W_n \overset{d}{=} \sum_{j=1}^N jsgn(Z_j)$$이고 $$sgn(Z_j) \overset{d}{=} S(j), S(j) \overset{\text{i.i.d}}{\sim} U\{-1,1\}$$이 성립한다. 또한 $$W_n^+$$에 대해서는 $$sgn(Z_i) =2I[Z_i>0]-1$$이 성립함으로 부호 부분이 $$U\{0,1\}$$ 즉, 성공확률이 1/2인 베르누이 시행이다. 따라서, 귀무가설 하에서 $$W_n, W_n^+$$의 분포는 다음과 같다.

$$ \begin{align*} W_n &\overset{d}{=} \sum_{i=1}^njS(j) \text{ where } S(j) \overset{\text{i.i.d}}{\sim} U\{-1,1\} \\ W_n^+ &\overset{d}{=} \sum_{j=1}^njB_j \text{ where } B_j \overset{\text{i.i.d}}{\sim} Ber(\frac{1}{2}) \end{align*} $$

이제 부호검정에서와 같이 검정력함수가 $$X_i$$에 대해 단조증가하여 $$H_0:\theta = \theta_0 \text{ v.s. } H_1:\theta > \theta_0$$에서의 유의 수준 $$\alpha$$의 검정이 가설 $$H_0:\theta \leq \theta_0 \text{ v.s. } H_1:\theta > \theta_0$$의 유의수준 $$\alpha$$ 검정이기도 함을 보이자. 먼저 $$W_n^+$$을 사용한 검정을 고려해보자.

검정력이 $$\gamma_{\phi_{W^+}}(\theta) = Pr_{\theta}[W^+_n \geq c]$$임으로, $$\theta$$에 대해 단조증가하기 위해서는 $$W_n^+$$가 $$\theta$$에 대해 단조증가해야한다.


$$
\begin{align*} W_n^+&= \sum_{i=1}^n I[Z_i>0]\sum_{j=1}^nI[|Z_j| \leq |Z_i| ]\\ &= \sum_{i=1}^n\sum_{j=1}^nI[Z_i>0, |Z_j| \leq Z_i] \\ &= \sum_{i=1}^n\sum_{j=1}^iI[Z_{(i)}>0, Z_{(i)} +Z_{(j)} >0] \text{ where } Z_{(1)} < ... < Z_{(n)} \\ &= {\sum\sum}_{1 \leq i \leq j \leq n}I[Z_{(i)}+Z_{(j)} > 0] \\ &= {\sum\sum}_{1 \leq i \leq j \leq n}I[Z_{i}+Z_{j} > 0] =  {\sum\sum}_{1 \leq i \leq j \leq n}I[X_{i}+X_{j} > 2\theta_0]

\end{align*}
$$




이는 $$Z_i$$가 연속형 확률변수로 서로 같은 값을 갖지 않을때 성립한다. 한편 $$X_i$$들은 $$\theta$$를 기준으로 대칭인 분포임으로 $$\theta$$가 증가할수록 $$X_i+X_j$$도 증가함으로, $$W_n^+$$는 $$\theta$$에 대해 단조증가한다.

이제 $$W_n$$을 사용한 검정의 검정력 함수 또한 $$\theta$$에 대해 단조증가임을 보이자. 마찬가지로 $$Z_i$$가 연속형으로서 한점인 $$(Z_i=0)$$인 경우를 고려하지 않으면 $$W_n$$은 다음과 같이 나타낼 수 있다.

$$ W_n = W_n^+ - W_n^- \text{ where } W_n^- = \sum_{i=1}^n I[Z_i<0]R(|Z_i|) $$

또한 $$W_n^+ + W_n^-$$은 전체 순위의 합임으로 $$n(n+1)/2$$이다. 즉, $$W_n = 2W_n^+ - n(n+1)/2$$임으로 마찬가지로 $$\theta$$에 대해 단조증가한다. 따라서 검정력인 $$Pr_{\theta}[W_n\geq c]$$또한 $$\theta$$에 대해 단조증가한다.

# 점근상대효율성 (피트만 효율성)

모수 $$\theta \in \Omega \subset \mathbb{R}$$에 관한 가설 $$H_0:\theta=\theta_0 \text{ v.s. } H_1:\theta > \theta_0$$을 유의수준 $$\alpha\in(0,1)$$에서 검정할 때 크기 $$n$$의 랜덤표본에 기초한 검정 $$\phi_{1n},\phi_{2n}$$의 점근상대효율성을 비교해보자.

1. 각각의 검정에 대한 검정통계량 $$T_{in}(i=1,2)$$에 대해 미분가능하고 연속인 함수 $$\mu_i(\theta),\sigma_i(\theta)$$이 각각 존재하여 각각의 기각역은 $$\sqrt{n}(T_{in} - \mu_i(\theta_{0}))/\sigma_i(\theta_0) \geq t_{in}$$의 꼴로 주어진다.
2. 귀무가설하의 모수 $$\theta_0$$의 근방에서 $$\sqrt{n}(T_{in} - \mu_i(\theta))/\sigma_i(\theta) \xrightarrow{d} \mathcal{N}(0,1)$$의 점근 정규성이 성립한다.

여기서 효율성의 비교는 가장 힘든 상황인 귀무가설에 가까운 대립에서 하는 것이 자연스럽다. 표본의 수가 고정된 상태에서 귀무가설 모수에서 대립으로 살짝 벗어났을 때, 검정력이 더 빨리 증가하는 검정이 더 효율이 좋은 검정일 것이다. 따라서, 위 조건은 가장 검정이 어려운 $$\theta_{1n}$$에서의 검정력이 계산 가능하여 비교할 수 있도록 하는 조건이다.

대립가설하의 모수를 $$\theta_{1n} = \theta_0 + \frac{K}{\sqrt{n}}$$이라고 하면, $$\phi_{1n},\phi_{2n}$$는 마찬가지로 단순가설 $$H_0:\theta=\theta_0 \text{ v.s. } H_1:\theta =\theta_{1n}$$의 유의수준 $$\alpha$$인 검정이다. 이러한 단순가설에 대해 목표 검정력 $$\gamma$$를 달성하기 위해 필요한 표본의 수를 $$N(T_{in};\gamma,\theta_{1n})$$이라고 하여, 표본크기의 역수값 $$N^{-1}(T_{1n};\gamma,\theta_{1n})/N^{-1}(T_{2n};\gamma,\theta_{1n})$$을 검정 $$\phi_{1n}$$의 검정 $$\phi_{2n}$$에 대한 점근상대효율성이라고 한다.

또한, 각각의 검정력은 다음과 같이 근사할 수 있다.

$$ \begin{align*} \gamma_{\phi_{in}}(\theta_{1n}) &= Pr_{\theta_{1n}}[\sqrt{n}\frac{(T_{in} - \mu_i(\theta_{0}))}{\sigma_i(\theta_0)} \geq t_{in}] \\ &\approx Pr[Z \geq \{t_{in} \sigma_i(\theta_0) + \sqrt{n}(\mu_i(\theta_0)-\mu_i(\theta_{1n}))\}/\sigma_i(\theta_{1n})] \ (\because \sqrt{n}(T_{in} - \mu_i(\theta_{1n}))/\sigma_i(\theta_{1n}) \xrightarrow{d} \mathcal{N}(0,1))\\ &= 1- \Phi[z_{\alpha}\frac{\sigma_i(\theta_{0})}{\sigma_i(\theta_{1n})} - \frac{\sqrt{n}(\mu_i(\theta_{1n}) - \mu_i(\theta_0))}{\sigma_i(\theta_{1n})}] \\ &\approx 1- \Phi[z_{\alpha} - \frac{\sqrt{n}(\theta_{1n} - \theta_0)\mu'(\theta_0)}{\sigma_i(\theta_{0})}] \ (\because \mu_i(\theta_{1n})-\mu_i(\theta_0) \approx  (\theta_{1n}-\theta_0)\mu'(\theta_0), \ \sigma_i(\theta_{1n})\approx \sigma_i(\theta_{0})) \end{align*} $$

또한 $$\theta_{1n} = \theta_0+\frac{K}{\sqrt{n}}(K>0)$$에서 다음의 수렴성이 성립한다.

$$ \frac{\sqrt{n}(\theta_{1n} - \theta_0)\mu'(\theta_0)}{\sigma_i(\theta_{0})} \approx \frac{\sqrt{n}\frac{K}{\sqrt{n}}\mu'(\theta_0)}{\sigma_i(\theta_0)} = \frac{K\mu'(\theta_0)}{\sigma_i(\theta_0)} < \infty $$

따라서 $$\theta_{1n}$$을 통한 단순가설 검정에서 목표 검정력 $$\gamma$$를 달성하기 위해 필요한 표본의 수 $$N(T_{in};\gamma,\theta_{1n})$$에 대한 다음의 근사식이 성립한다.

$$ N(T_{in};\gamma,\theta_{1n}) \approx (\frac{(z_{1-\gamma} + z_{\alpha})\sigma_i(\theta_0)}{(\theta_{1n} - \theta_0)\mu'_i(\theta_0)})^2 $$

즉 $$\phi_{1n}$$의 $$\phi_{2n}$$에 대한 점근상대 효율성은 다음과 같다.


$$
 \lim_{n \rightarrow \infty}\frac{N(T_{2n};\gamma,\theta_{1n})}{N(T_{1n};\gamma,\theta_{1n})} =( \frac{\mu_1'(\theta_0)/\sigma_1(\theta_0)}{\mu_2'(\theta_0)/
\sigma_2(\theta_0)})^2
 
$$
