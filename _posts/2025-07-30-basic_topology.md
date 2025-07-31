---
layout: single
title: "Basic Topology"
categories: Mathematic
tag: [Mathematical Analysis, Topology]
typora-root-url: ..\
author_profile: false
use_math: true
toc: true
---

루딘의 PMA 2단원을 정리한 내용이다. 열린집합과 닫힌 집합, 컴팩트 집합, 연결집합을 소개하고, 유클리드 공간을 넘어 일반적인 거리공간의 성질을 간략히 다룬다.



# 거리 공간 (metric space)

집합 $X$의 원소를 점(point)이라고 하자. 함수 $d:X \times X \rightarrow \mathbb{R}$가 다음을 만족하도록 정의하자.

1. $d(p,q) \geq 0 \ \forall p,q \in X$ (단, 등호는 $p=q$일 때만 성립)
2. $d(p,q) = d(q,p) \ \forall p,q\in X$
3. $d(p,q) \leq d(p,r) + d(r,q) \ \forall p,q,r \in X$

위 세 조건을 만족하는 함수 $d$를 거리(distance, metric)라고 정의하고 $(X,d)$를 **거리 공간(metric space)**이라고 한다.

다음의 함수가 거리 함수임은 쉽게 보일 수 있다. 이를 **유클리드 거리(Euclidean distance**)라고 한다.

$$
d(x,y) = |x - y| \   (x,y \in \mathbb{R}^k)
$$

이를 부여한 $(\mathbb{R}^k, d)$를 **유클리드 공간(Euclidean space)**라고 한다. 이후에는 단순히 $\mathbb{R}^k$로 표기한다.

- Segment $(a,b)$: $\{x \in \mathbb{R}: a < x < b\}$
- Interval $[a,b]$: $\{x \in \mathbb{R}: a \leq x \leq b\}$
- $k$-cell $[a_1,b_1]\times ... \times [a_k,b_k]$



# 열린 집합과 닫힌 집합 (open set and closed set)

이제 아래와 같이 $x \in X$ 주변의 거리 $\epsilon >0$ 안에 포함되는 점들의 집합을 **$x$의 근방(neighborhood**)이라고 하자.

$$
N_{\epsilon}(x) = \{y\in X: d(x,y) < \epsilon\}
$$

거리공간상의 부분집합 $E(\subset X)$를 정의하자.

1. 임의의 점 $x \in X$에 대해 $\forall \epsilon >0 , \ (N_{\epsilon}(x) -\{x\}) \cap E \neq \emptyset$이면 점 $x$를 $E$의 **극한점(limit point)**라고 한다. (Note. $x$가 $E$의 원소가 아닐 수도 있다.)
   1. $E$의 극한점들의 집합을 일반적으로 $E'$라고 표기한다. 만약 $E' \subset E$이면, 즉 $E$의 모든 극한점이 $E$에 속할 경우 $E$를 **닫힌 집합(closed set)**이라고 한다. 더 나아가 $E = E'$이면, $E$를 **완전 집합(perfect set)**이라고 한다.
   2. $\bar{E} = E \cup E'$를 $E$의 **폐포(closure)**라고 한다. 즉, $\bar{E}$는 $X$상에서 $E$를 포함하는 가장 작은 닫힌 부분집합이다. 또한 $\bar{E} = E$이면, $E$는 닫힌 집합이다.
2. $x \in E$이고, $x$가 $E$의 극한점이 아닌 경우, $x$를 $E$의 **고립점(isolated point)**라고 한다.
3. 임의의 점 $x \in X$ $\exists \epsilon >0 \ s.t. \ N_{\epsilon}(x) \subset E$인 경우, $x$를 $E$의 **내부점(interior point)**이라고 한다.
   1. $E$의 모든 점들이 내부점일 경우, 즉, $\forall x \in E, \ \exists \epsilon >0 \ s.t. \ N_{\epsilon}(x) \subset E$인 경우, $E$를 **열린 집합 (open set)**이라고 한다.
   2. 자연스럽게, $E$가 열린집합이면, 여집합 $E^c$는 닫힌 집합임을 알 수 있다.
4. $\exists y \in X, M \in \mathbb{R} \ s.t. \ \forall x \in E, \ d(x,y) < M$인 경우, $E$를 **유계 집합(bounded set)**이라고 한다.
5. 임의의 점 $x\in X$가 $E$에 속하거나, $E$의 극한점이거나, 또는 두가지를 모두 만족할 경우, $E$를 $X$상에서 **조밀한 집합(dense set)**이라고 한다.

**Note**. 점 $x$가 $E$의 극한점이면, $x$상의 어떤 근방도 $E$의 무한히 많은 점을 포함한다. (만약 유한하다면, 그러한 $\epsilon$에 대해, $E$의 유한한 점을 $p_1,...,p_n$이라고 하면, $\epsilon' = \min_id(x,p_i)$에 대한 근방에서 $x$를 제외하면 $E$의 어떤 점도 포함하지 않는다. 즉, $x$는 극한점이 아니다.)

**Note**. $N_{\epsilon}(x)$는 열린 집합이다. ($\forall y \in N_{\epsilon}(x)$에 대해 $\epsilon(y) = (\epsilon - d(x,y))/2$이면, $N_{\epsilon(y)}(y) \subset N_{\epsilon}(x)$이다.)

**Note**. $(a,b)$는 $\mathbb{R}^1$에서는 열린 집합(임의의 $a<x<b$에 대해 $\epsilon(x) = \min(\|a-x\|,\|b-x\|)/2$)이지만, $\mathbb{R}^2$에서는 열린 집합이 아니다.

열린 집합들을 모아놓은 집합족 $\{G_{\alpha}\}$에 대해 $\cup_{\alpha}G_{\alpha}$는 열린집합이다. 또한 $\cap_{i=1}^nG_{i}$도 열린집합이다. 드모르간의 법칙을 통해 임의의 닫힌 집합들의 교집합은 닫힌집함이며, 유한한 합집합은 닫힌 집합임을 알 수 있다.

e.g. $G_n=(-1/n, 1/n)$은 $\mathbb{R}$에서 열린 집합이지만, $\cap_{n=1}^{\infty}G_n = \{0\}$은 열린 집합이 아니다. 

즉, 열린집합의 교집합인 경우 근방의 반경이 각각의 열린집합들의 반경들의 하한인데, 열린집합이기 위해서는 이러한 하한이 양수가 되어야 한다.



Theorem 1.

거리 공간 $X$에 대해 $E \subset Y \subset X$인 경우를 생각해보자. 마찬가지로 $Y$도 거리 공간이 된다. 이 경우, 다음은 동치이다.

1. $E$는 $Y$에서 열린 집합이다.
2. $E = Y \cap G$를 만족하는 $X$에서 열린 집합 $G$가 존재한다.

증명 과정은 다음과 같다.

(a) → (b):  $x\in E$이면, $Y$상에서 $N_{Y,\epsilon(x)}(x) \subset E$인 $\epsilon(x) > 0$가 존재한다. 즉, 모든 점들의 근방에 대한 합집합을 정의하면, 이는 위에서 보였듯이, 마찬가지로 합집합이므로, $G$를 다음과 같이 정의한다.

$$
G =\cup_{x \in E}N_{X,\epsilon(x)}(x) =\cup_{x \in E}\{y \in X: d(x,y) < \epsilon(x)\}
$$

모든 $x,\epsilon$에 대해$x, \epsilon$$N_{Y,\epsilon}(x) \subset N_{X,\epsilon}(x)$이므로, $x \in G$이다. 즉, $E \subset G \cap Y$이다. 반대로 $x\in G \cap Y$이면, 이는 $E$에 속하는 점의 $Y$상의 근방 중 하나에 포함되며, $E$가 $Y$에서 열린집합이므로, $x\in E$이다. 따라서 $G \cap Y \subset E$이다.

(b) → (a): 즉,  $E = \cup_{x \in E}N_{Y,\epsilon(x)}(x)$를 만족하는 $\epsilon(x)$들이 존재하는 경우이다. 따라서, $E$는 열린 집합이다.



# 콤팩트 집합 (compact set)

거리공간 $X$에서 대해 $E \subset X$이고, $\{G_{\alpha}\}$를 $X$에서 열린 집합들의 집합족이라고 하자. 이 집합족으로 $E$를 덮을 수 있으면, 즉 $E \subset \cup_{\alpha}G_{\alpha}$이면, $\{G_{\alpha}\}$를 **열린 덮개(open cover)**라고 한다.

더 나아가, $E$의 모든 열린 덮개들이 유한한 부분 덮개(finite subsetcover)를 포함하면, $E$를 **콤팩트 집합(compace set)**이라고 한다. (즉 $E$의 모든 열린 덮개 $\{G_{\alpha}\}$에 대해  $E \subset \cup_{i=1}^n G_{\alpha_i}$를 만족하는 $\alpha_1,...,\alpha_n$이 항상 존재한다.)

다음 정리들은 Compactness의 이해를 돕기 위한 성질들이다.



Theorem 2.

$K \subset Y \subset X$에 대해 다음은 동치이다.

1. $K$는 $Y$에서 compact set이다.
2. $K$는 $X$에서 compact set이다.

증명 과정은 다음과 같다.

(a) → (b): $X$에서의 열린 덮개를 $\{G_{\alpha}\}$라고 하자. $G_{\alpha}$마다 Theorem 1에 의해 $V_{\alpha} = G_{\alpha} \cap Y$는 열린 집합이며, $\{V_{\alpha}\}$는 $K$를 덮는다. 따라서, $K \subset V_{\alpha_1} \cup ... \cup V_{\alpha_n}$을 만족하는 $\alpha_1,...,\alpha_n$이 존재한다. 또한 각각의 $V_{\alpha_i} \subset G_{\alpha_i}$임으로, $K \subset G_{\alpha_1} \cup ... \cup G_{\alpha_n}$를 만족한다.

(b) → (a): $Y$에서의 열린 덮개를 $\{V_{\alpha}\}$라고 하자. 이 경우에는 $K \subset G_{\alpha_1} \cup ... \cup G_{\alpha_n}$와 $V_{\alpha_i} = G_{\alpha_i} \cap Y$

를 만족하는 열린집합 $G_{\alpha_1},...,G_{\alpha_n}$이 항상 존재하고, $K \subset Y$이므로, $K \subset V_{\alpha_1} \cup ... \cup V_{\alpha_n}$이다.



Theorem 3.

콤팩트 집합 $K \subset X$는 $X$에서 **유계 닫힌 집합**이다.

아래와 같이 $G$를 정의하면, 이는 $K$의 열린 덮개이다.

$$
G = \cup_{x \in K} N_{\epsilon(x)}(x)
$$

따라서, 유한개의 $x_1,...,x_n$의 근방으로 $K$를 덮을 수 있다. 이제,

$$
r_1 = \max_i d(x_1,x_i)+\max_i\epsilon(x_i)
$$

라고 하면, $K \subset N(x_1, r_1)$임으로, $K$는 유계이다.

이제 닫힌 집합임을 보이기 위해 임의의 $y \in K^c$에 대해 다음과 같은 열린 덮개를 정의하자.

$$
G(y) = \cup_{x \in K}N_{\epsilon_y(x)}(x), \ \epsilon_y(x) < d(x,y)/2
$$

마찬가지로, 유한개의 $x_1,...,x_n$의 근방으로 $K$를 덮을 수 있다. 이제 다음과 같이 $r$을 정의하자.

$$
r = \min_{1\leq i \leq n} d(y, x_i)/2
$$

즉 $N_r(y)$는 $x_1,...,x_n$의 근방과 겹치지 않으므로, $K$와도 겹치지 않는다. 따라서 $K^c$에 속한다. 즉, $K^c$가 열린 집함임으로, $K$는 닫힌 집합이다.



Theorem 4.

콤팩트 집합 $K \subset X$에 대해 닫힌 부분 집합 $E \subset K$는 콤팩트 집합이다. 즉, 콤팩트 집합의 닫힌 부분집합은 콤팩트 집합이다.

증명 과정은 다음과 같다.

$E^c$가 열린 집합임으로 $\{G_{\alpha}\}$를 $E$의 열린 덮개라고 하면,$\
\{G_{\alpha}\} \cup E^c$는 $K$의 열린 덮개이다. $K$가 콤팩트 집합이므로 이는 유한개의 부분 덮개를 갖는다. 따라서 $E \subset K$임으로,  $\{G_{\alpha}\}$의 유한개의 부분 덮개로 $E$를 덮을 수 있다.

**Note**. 콤팩트 집합은 닫힌 집합이며, 닫힌 집합의 교집합도 닫힌 집합이다. 즉, Theorem 4에 의해  **닫힌 집합 $F$와 콤팩트 집합 $K$에 대해 $F \cap K$는 콤팩트 집합이다.**



Theorem 5. 칸토어의 교차 정리 (Cantor's Intersection Theorem)

포함관계가 작아지는 공집합이 아닌 콤팩트 집합 열 $\{K_n: K_{n+1} \subset K_{n}, K_n\neq \emptyset\}$에 대해 

$$
\lim_{n \rightarrow \infty} K_n = \cap_{n=1}^{\infty}K_n \neq \emptyset
$$

더 일반적인 경우, 콤팩트 집합열 $\{K_{\alpha}\}$에 대해 모든 유한개의 부분 집합들의 교집합이 공집합이 아닐 때 $\cap_{\alpha}K_{\alpha}$도 공집합이 아님을 보여보자.

만약 $\cap_{\alpha}K_{\alpha} = \emptyset$라고 가정하자. 즉, $K_1$는 다른 부분집합 $K_{\alpha}$와 교집합을 갖지 않는다고 하자. Theorem 3에 의해 $K_{\alpha}^c$는 열린 집합이고, 이에 대한 합집합은 $K_1$을 덮는다. 또한 $K_1$이 콤팩트 집합임으로, 이를 덮는 유한개의 부분 덮개 $K_{\alpha_1}^c \cup ... \cup K_{\alpha_n}^c$이 존재한다. 즉, $K_1 \cap K_{\alpha_1} \cap ... \cap K_{\alpha_n} = \emptyset$으로 공집합인 유한개의 부분집합족이 존재함으로 이는 모순이다. 그러므로, $\cap_{\alpha} K_{\alpha} \neq \emptyset$이다.

포함관계가 작아지는 공집합이 아닌 콤팩트 집합열의 유한개의 부분집합  $K_{n_1},...,K_{n_m}(n_1<...<n_m)$들의 교집합은 $K_{n_m} \neq \emptyset$임으로 위 조건을 만족하므로, 마찬가지로 성립한다.





### Theorem 6. 하이네-보렐 정리 (Heine-Borel Theorem)

**유클리드 공간** 상의 부분집합 $E \subset \mathbb{R}^k$에 대해 다음은 동치이다.

TFAE

(a) $E$는 **유계 닫힌 집합**이다.

(b) $E$는 **콤팩트 집합**이다.

(c) $E$의 무한 부분 집합은 $E$에서 극한점을 갖는다.



증명과정은 다음과 같다.

(a) → (b): 

**Lemma**. 포함관계가 작아지는 $k$-cell의 열 $\{I_n\}$에 대해 $\cap_{n=1}^{\infty}I_n$은 공집합이 아니다.

먼저, 각 축마다 $[a_{nj}, b_{nj}]$ ( $j=1,...,k$ ) 에 대해 고려해보자. $x_j^* = \sup \{a_{nj}:n=1,2,...\}$라고 하자. 

1. $a_{nj} \leq x_{j}^*, \ \forall n$
2. $b_{1j},b_{2j},...$들은 모두 $\{a_{nj}: n=1,2,...\}$의 상계이므로, $x_j^* \leq b_{nj}, \ \forall n$

이제 $$x^* = (x_1^*,...,x_k^*)$$라고 정의하면, 모든 $n$에 대해 $x^* \in I_n$이다.

유계 닫힌 집합임으로, $E$를 포함하는 $k$-cell $I$가 존재한다. $I$가 콤팩트 집합이면, Theorem 4에 의해 닫힌 부분 집합 $E$도 콤팩트이기에 $k$-cell이 콤팩트임을 확인하면 충분하다.

$I = [a_1,b_1] \times ... \times [a_k, b_k]$이면, 모든 $x,y\in I$에 대해 $\|x-y\| \leq \delta$이다.

$$
\delta = [\sum_{j=1}^k (b_j - a_j)^2]^{\frac{1}{2}}
$$

만약 $I$가 콤팩트가 아니라고 가정해보자. 즉, 유한개의 부분덮개가 존재하지 않는 열린 덮개 $\{G_{\alpha}\}$가 있다고 가정하자. 이제 이 열린 덮개에 대해 $I$를 각 축마다 절반으로 쪼갠 $2^k$개의 부분에서 유한개로 덮을 수 없는 부분을 $I_1$이라고 하자.  다시 $I_1$을 각 축마다 절반으로 쪼개어 마찬가지로 유한개로 덮을 수 없는 부분을 $I_2$로 하고, 이를 반복하여 $\{I_n\}$을 얻는다. 

이는 포함관계가 작아지는 $k$-cell의 열이다. 즉, lemma에 의해 $$x^* \in \cap_{n=1}^{\infty} I_n$$인 $$x^*$$가 존재한다. 열린 덮개 $\{G_{\alpha}\}$에서 $$x^{*}$$를 포함하는 열린 집합 $G_{\alpha}$에 대해 $N_r(x^*) \subset G_{\alpha}$를 만족하는 $r>0$이 존재한다.

또한, 임의의 $x,y \in I_n$에 대해 $\|x-y\| \leq \delta/2^n$이 성립하고, 아르키메데스의 정리에 의해 $\delta/2^n < r$을 만족하는 $n$이 존재한다. 즉, 충분히 큰 $n$에 대해  $I_n \subset N_r(x^*) \subset G_{\alpha}$을 만족하므로, 유한개의 부분덮개로 $I_n$을 덮을 수 있으므로, 모순이다. 따라서 $I$는 콤팩트 집합이다.

(b) → (c): 무한개의 점을 갖는 콤팩트 집합 $E$의 부분 집합을 $S$라고 하자.

만약 $E$에서 극한점을 갖지 않는 경우를 고려해보자. 즉, 모든 $x \in E$에 대해 $N_{\epsilon(x)}(x) - \{x\}$가 $S$와 겹치지 않도록 하는 $\epsilon(x) >0$가 반드시 존재한다. 즉, $N_{\epsilon(x)}(x)$는 기껏해야 한개의 $S$ 내부의 점을 갖는다. 

이러한 근방들을 모아놓은 열린 집합족 $\{N_{\epsilon(x)}(x): x \in E\}$는 $E$를 덮기에, $S \subset E \subset N_{\epsilon_1}(x_1)\cup...\cup N_{\epsilon_n}(x_n)$를 만족하는 $x_1,...,x_n$이 존재하고, 각각의 근방은 최대 1개의 $S$ 내부의 점을 갖고 있기에, $S$는 최대 $n$개의 점을 갖는다. 따라서 모순이다.

(c) → (a): 만약 유계가 아니라면, $S = \{x_n:\|x_n\| \geq n\}$과 같은 부분 집합을 잡을 수 있고, 이는 무한개의 점을 갖지만, $\mathbb{R}^k$에서 극한점을 갖지 않으므로, $E$에서도 극한점을 갖지 않는다. 따라서 모순이다.

만약 닫힌 집합이 아니라면, $E$의 극한점인 $x \notin E$가 존재한다. 즉, 임의의 $r>0$에 대해 $\|x-y\| < r$인 $y \in E$가 항상 존재하고, 아르키메데스의 정리에 의해 $r <n$인 $n$이 항상 존재하기에 $S = \{x_n \in E :  \|x_n-x\| < 1/n\}$의 부분집합을 정의하면, $S$는 무한개의 점을 갖는 $E$의 부분집합니다.

또한, $y \in \mathbb{R}^k-\{x\}$에 대해서 다음을 만족한다.

$$
\begin{align*}
|x_n - y| &\geq |x - y| - |x_n - x| \\ &> |x-y| - \frac{1}{n}  > \frac{1}{2} |x-y|

\end{align*}
$$

마지막 부등식은 아르키메데스의 정리에 의해 성립한다. 즉, $2r = \|x-y\|$를 만족하도록, $N_{r}(y)$를 잡으면, $S$에 속하지 않는다. 즉, $S$는  $\mathbb{R}^k$에서 $x$외에는 극한점을 갖지 않으므로, $E$에 대해서도 마찬가지이며, 극한점 $x$는 $E$에 속하지 않는다. 따라서 모순이다.

**Note**. (a)→(b), (c)→(a)의 증명 과정에서 **아르키메데스 정리와 같은** 실수의 조밀함; 즉 거리 함수가 아닌 $\mathbb{R}^k$의 성질을 이용하여 보였다. 모든 거리공간에 대해서 (b)와 (c)는 동치이지만, (a)는 그렇지 않다. (즉, (c) → (b)도 $\mathbb{R}^k$의 성질을 사용하지 않고 보일 수 있다.)

**Note.** Theorem 3에서 (b) → (a)임을 보였었다. 즉, 일반적인 거리공간에서는 컴팩트 집합이 더 걍력한 성질이다.

(c)→(b)를 보이는 과정은 다음과 같다.

먼저, 거리공간 $X$에 대해 열린 부분집합족 $\{V_{\alpha}\}$에 대해 $X$의 모든 열린 부분 집합을 $\{V_{\alpha}\}$의 부분 집합족의 합집합으로 나타낼 수 있으면, $\{V_{\alpha}\}$를 $X$의 **기저(base)**라고 한다. (다시 말해서, 임의의 $x \in X$에 대해 $x$를 포함하는 임의의 열린집합 $G(x)$에 대해 $x \in V_{\alpha} \subset G$를 만족하는 $V_{\alpha}$가 존재한다.)

가산의 조밀한 부분집합을 포함하는 거리공간을 **분리가능(separable)**하다고 한다. 

이제, $X$에 대해 $E \subset X$도 하나의 거리공간으로 볼 수 있다. 위 정의와 다음의 명제들을 각각 보여 (c)→(b)를 증명할 수 있다.

1. 모든 분리가능한 거리 공간은 **가산인 기저(countable base)**를 갖음을 보인다. 
2. 거리공간 $E$가 (c)를 만족하면 분리가능함을 보인다.
3. 가산 기저를 활용해 거리공간의 모든 열린 덮개는 유한한 부분덮개를 가짐을 보여 $E$가 콤팩트임을 보인다.





Theorem 6.1.

거리공간 $X$가 분리 가능하면, 가산인 기저를 갖는다.

즉, 가산의 조밀한 부분집합 $D$를 갖는다. $D$가 가산임으로 $D$의 원소들을 $d_1,d_2,...$와 같이 정의 할 수 있다. 

임의의 $x \in X$를 잡아서 고정하고, 다음과 같은 열린 집합족 $\mathbb{B}$를 정의하자. 이는 $\mathbb{N} \times \mathbb{Q}^+$와 일대일 대응임으로, 가산이다.

$$
\mathbb{B} = \{N_{q}(d_n):n=1,2,..., q \in \mathbb{Q}^+\}
$$

이제 $x$를 포함하는 임의의 열린 집합을 $G(x)$라고하면, $N_{\epsilon}(x) \subset G(x)$를 만족하는 $\epsilon>0$가 존재한다. 

또한, $D$가 조밀한 집합임으로 $d =d(x,d_n) < \epsilon/2$를 만족하는 $d_n$이 존재한다. 

즉, 이러한 $d_n$을 중심으로 $d$보다는 크고 (점 $x$를 포함) , $\epsilon - d$보다는 작은 ($N_{\epsilon(x)}(x)$내부에 속함)  $q \in \mathbb{Q}^+$를 잡아 근방 $N_{q}(d_n)$을 정의하면, 이는 $N_q(d_n) \subset \mathbb{B}$이고, $x \in N_q(d_n) \subset N_{\epsilon(x)}(x) \subset G(x)$이기에, $\mathbb{B}$가 $X$의 기저임을 알 수 있다. 

이는 $d < \epsilon -d$; 즉, $d < \epsilon/2$이도록 $d$를 정의했으므로, 유리수의 조밀성에 의해 $q \in (\epsilon/2, \epsilon - d)$가 항상 존재한다.

그러므로, $\mathbb{B}$는 $X$의 가산 기저이다.





Theorem 6.2.

거리공간 $X$에 대해 임의의 무한부분집합 $E \subset X$가 $X$에서 극한점을 갖는다면, $X$는 분리 가능하다. 

$x_1 \in X$를 하나 고정하고, 임의의 $\delta >0$에 대해서 $d(x_n,x_{n+1}) > \delta$가 되도록, $x_2,x_3,...$들을 잡으면 유한개이다. 만약, 무한개라면, $\{x_n\}\subset X$은 무한 부분 집합이고, 해당 부분 집합은 극한점을 갖지 않음으로 모순이다. 즉, $N_{\delta}(x_1),...,N_{\delta}(x_N)$은 $X$를 덮는다. 

이제, $x \in X$를 고정한다. 임의의 $\epsilon>0$에 대해 $\delta = 1/n < \epsilon$을 잡으면, $\min_n d(x,x_n) < \delta < \epsilon$ 임으로, $x_n \in N_{\epsilon}(x)$가 되는 $x_n$이 반드시 존재한다.

사실 이경우의, 가산의 조밀한 부분집합은 $\cup_{n=1}^{\infty}\{x_i:\delta=1/n\}$이다.





Theorem 6.3.

거리공간 $X$에 대해 임의의 무한부분집합 $E \subset X$가 $X$에서 극한점을 갖는다면, $X$는 콤팩트이다.

만약 유한한 부분덮개를 갖지않는 $X$의 열린 덮개 $\{G_{\alpha}\}$가 존재한다고 가정하자.

Theorem 6.2에 의해 $X$는 분리가능하며, Theorem 6.1에 의해 가산인 기저를 갖는다. 이러한 가산의 기저를 $\{V_n\}$라고 하자. 임의의 $x \in X$에 대해 $x \in G_{\alpha}$인 $G_{\alpha}$가 존재하고, $x \in V_n \subset G_{\alpha}$를 만족하는 $V_n$도 존재한다. 이제 각각의 $V_n$을 완전히 포함하는 $G_{\alpha}$들을 모아놓은 부분 덮개는 $X$를 덮는 가산의 열린 덮개이다. 이를 $\{G_n\}$라고 하자.

이제 $E = \{x_n \in X: x_n \notin \cup_{i=1}^nG_i\}$라고 하자. $\{G_{\alpha}\}$는 유한한 부분덮개를 갖지 않으므로, 가산 부분덮개 $\{G_n\}$도 마찬가지이기에, $E$는 무한 부분집합이다. 이제 $E$가 극한점이 존재하지 않음을 보이면 충분하다.

 만약 $E$가 $X$에서 극한점 $x_0$가 존재한다고 하자. 가산 부분 덮개를 이용하여, $x_0 \in G_{N}$인 $N$이 존재한다. $G_N$은 열린집합임으로, $N_{\epsilon_0}(x_0) \subset G_N$을 만족하는 $\epsilon_0>0$이 존재한다. $x_0$은 $E$의 극한점이므로, 임의의 $\epsilon>0$에 대해 $N_{\epsilon}(x_0)$과 $E$의 교집합은 반드시 $x_0$이외의 점을 하나 이상 포함해야 한다. 하지만 $E$의 정의에 의해 $x_N, x_{N+1},...$은 $G_N$에 포함되지 않는다. 그러므로, $2\epsilon = \min( \min_{1\leq i<N}d(x_i,x_0), \epsilon)$이 되는 $N_{\epsilon}(x_0)$은 $x_0$을 제외하고는 $E$의 어떠한 점도 포함하지 않으므로 모순이다.

**Note**. 증명과정에서, 가산인 기저를 갖는 거리공간에서는 임의의 열린 덮개가 가산인 부분 덮개를 가짐을 보였다. 임의의 열린 덮개가 가산인 부분 덮개가 항상 존재한다는 성질을 린델로프 속성(Lindelöf property)이라고 하며, 이를 만족하는 거리공간을 린델로프 공간이라고 한다. 이는, 컴팩트 공간보다 약한 조건이며, 위 명제의 필요조건은 성립하지 않는다.





### Theorem 7. **볼차노-바이어슈트라스** 정리 (Bolzano–Weierstrass theorem)

유클리드 공간상에서 무한개의 점을 갖는 임의의 부분집합 $E \subset \mathbb{R}^k$이 유계이면, $\mathbb{R}^k$에서 극한점을 갖는다.

$E$가 유계이므로, $E$를 포함하는 k-cell $I \subset \mathbb{R}^k$가 존재한다. $I$는 하이네-보렐 정리에서 보였듯이, 콤팩트 집합이기에 (c)에 의해 부분 집합 $E$는 $I$에서 극한점을 갖는다.





Theorem 8.

유클리드 공간에서 공집합이 아닌 완전 집합$P \subset \mathbb{R}^k$는 비가산 집합이다.

칸토어의 교차 정리와 귀류법을 이용하여 보일 수 있다.

만약 $P$가 가산 집합이라면 $P$의 점들을 $x_1,x_2,...$과 같이 자연수로 인덱싱이 가능하다. 또한 이들은 모두 $P$의 극한점이다. $r_1>0$을 고정하여, $V_1 = N_{r_1}(x_1)$라고 하자. $V_1$의 폐포(closure) $\bar{V}_1$은 유계 닫힌 집합임으로 하이네 보렐 정리에 의해 콤팩트 집합이다.

극한점의 성질을 이용하여 아래의 조건을 만족하는 $V_2 = N_{r_2}(x_2), V_3 = N_{r_3}(x_2),...$들을 귀납적으로 정의할 수 있다. (이전 근방에는 포함되면서, 중점은 포함하지 않는 근방을 정의)

$$
\begin{align*}
&\bar{V}_{n+1} \subset V_n \\
&x_n \notin \bar{V}_{n+1} \\
&\bar{V}_{n+1} \cap P \neq \emptyset, \ n=1,2,...
\end{align*}
$$

이제 $$K_n = \bar{V}_n \cap P$$( $n=1,2,...$) 라고 하자. Theorem 4에 의해 이는 컴팩트 집합이고, 위의 세번째 조건에 의해 공집합이 아니다. 또한 첫번째 조건에 의해 포함관계가 작아지는 집합열이므로, 칸토어의 교차정리에 의해 $\cap_{n=1}^{\infty}K_n$은 공집합이 아니여야 한다. 하지만, $y \in\cap_{n=1}^{\infty}K_n$라면, $y\in P$이고, $y \in \bar{V}_1 \cap \bar{V}_2 \cap...$인데 이는 두번째 조건에 의해 $P$의 어떤 점도 포함할 수 없다. 즉, $y \in P$이고, $y \notin P$이여야 함으로 모순이다.

Corollary. **임의의 구간 $[a,b]$ ($a<b$)은 비가산**이다. 특히, $\mathbb{R}$은 비가산이다.

### 칸토어 집합 (The Cantor set)

$\mathbb{R}^1$상에서 어떠한 segment $(a,b)$도 포함하지 않는 완전 집합으로 다음과 같이 정의할 수 있다.

$$
\begin{align*}
E_0 &= [0,1] \\
E_1 &= [0,\frac{1}{3}] \cup [\frac{2}{3},1] \\
E_2 &= [0,\frac{1}{9}] \cup [\frac{2}{9},\frac{3}{9}] \cup [\frac{6}{9},\frac{7}{9}] \cup [\frac{8}{9},1] \\ \vdots
\end{align*}
$$

각 $E_n$은 유계 닫힌 구간의 $n$개의 합집합임으로, 마찬가지로 유계 닫힌 구간이고, 하이네 보렐 정리에 의해 콤팩트 집합이다. 또한, $E_1 \supset E_2 \supset ...$로 포함관계가 작아지는 집합열이다. 이제 이러한 집합열에 대해 다음과 같이 칸토어 집합을 정의한다.

$$
P = \cap_{n=1}^{\infty}E_n
$$

칸토어의 교차정리에 의해 $P$는 공집합이 아니다.

이제 $P$가 완전 집합임을 보이자. 닫힌 집합들의 교집합은 마찬가지로 닫힌 집합이므로 $P$는 닫힌 집합이다. 따라서, $P$의 모든 점이 극한점임을 보이면 충분하다. 

임의의 $x_0 \in P$에 대해서, $x_0$을 포함하는 구간 $I_n=[\frac{p}{3^n}, \frac{p+1}{3^n}]$이 반드시 존재한다. 따라서, 아르키메데스의 정리에 의해 모든 $\epsilon>0$에 대해, $n$이 충분히 크면 $I_n \subset (x_0-\epsilon, x_0 + \epsilon)$이고, 구간의 끝점 $\frac{p}{3^n}, \frac{p+1}{3^n}$중 하나는 $x_0$이 아니므로, $x_0$이 아닌 끝점을 $x_n$으로 잡는다.

정리하면, 모든 $\epsilon>0$에 대해, $x_n \in (x_0-\epsilon, x_0+\epsilon ) \cap P$가 항상 존재함으로, $x_0$은 $P$의 극한점이다. 즉, $P$는 공집합이 아닌 완전집합임으로, Theorem 8에 의해 비가산 집합이다. 

이제 $P$가 어떤 segment $(a,b)$도 포함하지 않음을 보이자. $E_n$의 각 구간의 길이는 $3^{-n}$임으로, 아르키메데서의 정리에 의해  segment의 길이 $d=b-a$보다 작게하는 $E_n$이 항상 존재해서, 이에 대한 교집합 $P$에 속할 수 없다.

이처럼 칸토어 집합은 점의 개수는 셀 수 없이 많지만, 길이는 0인 매우 특별하고 중요한 집합의 예시이다.



## 연결 집합 (Connected set)

거리공간 $X$의 부분집합 $A,B$에 대해 $A \cap \bar{B} = \emptyset$이고, $\bar{A} \cap B = \emptyset$이면, $A$와 $B$를 **분리된(separated)** 집합이라고 한다. 즉, $A$의 어떤 점도 $B$에서 극한점이 아니며 $B$에 속하지 않고, $B$의 어떤 점도 $A$의 폐포에 속하지 않는다.

집합 $E \subset X$가 공집합이 아닌 분리된 집합의 합집합이 아닌 집합일 때, **$E$를 연결 집합(connected set)**이라고 한다.

$\mathbb{R}^1$의 경우, 부분 집합 $E \subset \mathbb{R}^1$에 대해 다음은 동치이다.

(a) $E$는 연결집합이다.

(b) $\forall x,y \in E, \ \forall z \in (x,y), \ z \in E$



증명과정은 다음과 같다.

(a)→(b): 만약 $E$에 속하지 않는 $z \in (x,y)$가 존재하는 $x,y$가 있다고 가정하자. 이제 이를 이용해 $A_z = (-\infty, z) \cap E, \ B_z = (z, \infty) \cap E$를 정의하면, $x\in A_z, y\in B_z$로 각각은 공집합이 아니며, $A_z \cup B_z = E$이다. 또한 $(-\infty, z)$와 $(z,\infty)$는 분리된 집합임으로, 각각의 부분집합 $A_z$와 $B_z$도 분리된 집합이다. 따라서, $E$는 연결 집합이 아님으로 이는 모순이다.

(b) → (a):

Lemma. $E \subset \mathbb{R}^1$가 공집합이 아닌 위로 유계인 집합이면, $\sup E \in \bar{E}$. (또는 아래로 유계이면, $\inf E \in \bar{E}$)

$y = \sup E$라고 하면, 임의의 $\epsilon>0$에 대해 $x \in(y-\epsilon, y)$인 $x \in E$가 항상 존재한다. (만약 존재하지 않으면, $y-\epsilon$이 가장 작은 상계임으로 가정에 모순이다.) 즉 $y$는 $E$에서 극한점을 갖는다.

 $A\cup B = E$를 만족하며 공집합이 아닌 분리된 집합 $A,B$가 있다고 가정하자. 여기서 임의의 $x \in A, y \in B$를 뽑아 $x<y$라고 하자. $z = \sup(A \cap [x,y])$라고 정의하면, Lemma에 의해 $z \in \bar{A}$이다. 즉, $z \notin B$이다.

만약 $z \notin A$라면, 이는 $x<z<y$인 경우이고, $z \notin B$였기에 $z \notin E$임으로 마찬가지로 모순이다.

만약 $z \in A$라면, $z \notin \bar{B}$이다. 즉 $B$에서 극한점이 아님으로, $(z,z+r) \cap B = \emptyset ( z+r < y)$인 $r>0$가 반드시 존재한다. 또한, $z$가 $[x,y]$내의 $A$의 상한임으로 이 segment는 $A$와도 겹치지 않는다. 따라서, $w \in (z, z+r)$은 $w \notin E$이고 $x \in (x,y)$임으로 모순이다.
