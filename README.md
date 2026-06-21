# Variational Divergence Minimization — Week 1

**Mathematical Foundation for GenAI** · Quiz questions with worked answers and explanations.

Topics: deep generative models, $f$-divergences, conjugate (Fenchel) duality, the $f$-GAN min–max objective, and the reparameterization gradient.

> 📄 A typeset PDF version is in [`pdf/GenAI_Week1_Reference.pdf`](pdf/GenAI_Week1_Reference.pdf). The LaTeX source is in [`tex/week1.tex`](tex/week1.tex) and is rebuilt automatically on every push (see the Actions tab).

---

## Background in one page

**The goal.** We have data $D=\{x_1,\dots,x_n\}$ drawn i.i.d. from an unknown distribution $P_X$, with each $x_i\in\mathbb{R}^d$. We want to *learn to sample* from $P_X$. The plan: assume a parametric family $P_\theta$ (a model), measure how far $P_\theta$ is from $P_X$ using a divergence, and minimize that divergence over $\theta$.

**The sampler.** Draw a latent $z\sim\mathcal{N}(0,I)$ and push it through a neural network $g_\theta$. The induced distribution of $x=g_\theta(z)$ is $P_\theta$ — the *pushforward* of the latent distribution under $g_\theta$. At the optimum $P_\theta\approx P_X$, so $g_\theta(z)$ becomes an approximate sampler for the data.

**The $f$-divergence.** For a convex, left-semicontinuous $f$ with $f(1)=0$,

$$D_f(P_X\|P_\theta)=\int p_\theta(x)\,f\!\left(\frac{p_X(x)}{p_\theta(x)}\right)dx=\mathbb{E}_{x\sim P_\theta}\!\left[f\!\left(\frac{p_X(x)}{p_\theta(x)}\right)\right].$$

Different choices of $f$ give different divergences (KL, reverse KL, JS, total variation).

**The variational trick.** We don't know the densities, only samples. Using the convex conjugate $f^*(t)=\sup_u\big(tu-f(u)\big)$ and the identity $f(u)=\sup_t\big(tu-f^*(t)\big)$, the divergence becomes a *lower bound expressed purely with expectations*:

$$D_f(P_X\|P_\theta)\;\ge\;\sup_{T\in\mathcal{T}}\Big(\mathbb{E}_{x\sim P_X}[T(x)]-\mathbb{E}_{x\sim P_\theta}\big[f^*(T(x))\big]\Big).$$

Both expectations can be estimated from samples (law of large numbers). Combined with the generator we get the $f$-GAN min–max problem

$$\min_\theta\ \max_{T\in\mathcal{T}}\ \Big(\mathbb{E}_{x\sim P_X}[T(x)]-\mathbb{E}_{x\sim P_\theta}\big[f^*(T(x))\big]\Big),$$

where $T$ is the discriminator/critic and $g_\theta$ is the generator.

---

## Question 1

The $f$-divergence between two distributions ($p_X,p_\theta$ are the density functions) is defined as:

- **A.** $D_f(P_X\|P_\theta)=\int p_\theta(x)\,f\!\left(\frac{p_X(x)}{p_\theta(x)}\right)dx$
- **B.** $D_f(P_X\|P_\theta)=\int p_X(x)\,f\!\left(\frac{p_X(x)}{p_\theta(x)}\right)dx$
- **C.** $D_f(P_X\|P_\theta)=\int p_\theta(x)\,f\!\left(\frac{p_\theta(x)}{p_X(x)}\right)dx$
- **D.** $D_f(P_X\|P_\theta)=\int p_X(x)\,f\!\left(\frac{p_\theta(x)}{p_X(x)}\right)dx$

> ### ✅ Answer: A
> The $f$-divergence weights by the *second* distribution $p_\theta$ (the one we measure divergence *to*) and applies $f$ to the ratio $\frac{p_X}{p_\theta}$. Sanity check with the KL generator $f(u)=u\log u$:
>
> $$\int p_\theta\cdot\frac{p_X}{p_\theta}\log\frac{p_X}{p_\theta}\,dx=\int p_X\log\frac{p_X}{p_\theta}\,dx=D_{\mathrm{KL}}(P_X\|P_\theta),$$
>
> recovering the forward KL exactly. Options B and D put the wrong density in front; C and D invert the ratio. Only A reproduces a valid $f$-divergence.

---

## Question 2

The $f$ function used in the divergence should be at least:

- **A.** Left Continuous
- **B.** Right Continuous
- **C.** Left Semi Continuous
- **D.** Right Semi Continuous

> ### ✅ Answer: C — Left Semi Continuous
> The defining conditions on $f$ are: convex, *left- (lower-) semicontinuous*, and $f(1)=0$. Lower semicontinuity is what makes the convex conjugate $f^*$ (and the biconjugate $f^{**}=f$) well behaved, which is exactly what the variational lower bound relies on. Full continuity is stronger than needed; semicontinuity is the minimal requirement.

---

## Question 3

What is the $f$ function for KL-divergence?

- **A.** $\log(u+1)$
- **B.** $u\log(u)$
- **C.** $0.5\,|u-1|$
- **D.** $(u+1)\log(u+1)$

> ### ✅ Answer: B — $u\log(u)$
> Substituting $f(u)=u\log u$ gives $\int p_\theta\cdot\frac{p_X}{p_\theta}\log\frac{p_X}{p_\theta}\,dx=\int p_X\log\frac{p_X}{p_\theta}\,dx$, the forward KL. For reference, option C, $0.5\,|u-1|$, is the generator of the **total variation** distance.

---

## Question 4

Is Forward KL the same as Reverse KL?

- **A.** Yes
- **B.** No

> ### ✅ Answer: B — No
> Forward KL is $D_{\mathrm{KL}}(P_X\|P_\theta)$ and reverse KL is $D_{\mathrm{KL}}(P_\theta\|P_X)$; in general $D_{\mathrm{KL}}(P\|Q)\ne D_{\mathrm{KL}}(Q\|P)$ — KL is **not symmetric**. They also fit differently: forward KL is mass-covering / mean-seeking (it penalizes putting zero model mass where the data has mass), while reverse KL is mode-seeking (it lets the model concentrate on a subset of modes).

---

## Question 5

If $f(u)$ is a convex function, then the conjugate function $f^*(t)$ is a:

- **A.** Concave function
- **B.** Convex function
- **C.** Quasi-Convex function
- **D.** Quasi-Concave function

> ### ✅ Answer: B — Convex function
> The convex conjugate $f^*(t)=\sup_u\big(tu-f(u)\big)$ is a pointwise supremum of *affine* functions of $t$ (each fixed $u$ gives the line $t\mapsto tu-f(u)$). A pointwise supremum of affine/convex functions is always convex — so $f^*$ is convex regardless of whether $f$ was convex.

---

## Question 6

If the divergence is the JS-divergence, what is the $f$ function used?

- **A.** $0.5\big(u\log(u+1)-(u+1)\log(0.5(u+1))\big)$
- **B.** $0.5\big(u\log(u)-(u-1)\log(0.5(u-1))\big)$
- **C.** $0.5\big(u\log(u-1)-(u+1)\log(0.5(u+1))\big)$
- **D.** $0.5\big(u\log(u)-(u+1)\log(0.5(u+1))\big)$

> ### ✅ Answer: D
> The JS generator is $f(u)=\tfrac12\big(u\log u-(u+1)\log\tfrac{u+1}{2}\big)$, and $0.5(u+1)=\tfrac{u+1}{2}$. Only D has both the correct $u\log(u)$ term and the correct $(u+1)\log(0.5(u+1))$ term. A has $u\log(u+1)$ (wrong first term), B uses $(u-1)$, and C has $u\log(u-1)$.

---

## Question 7

Is there a function $f$ such that the resulting divergence can take negative values?

- **A.** Maybe for some values
- **B.** Yes
- **C.** No

> ### ✅ Answer: C — No
> An $f$-divergence is always non-negative for any valid $f$. By Jensen's inequality (convexity of $f$):
>
> $$D_f(P_X\|P_\theta)=\mathbb{E}_{p_\theta}\!\left[f\!\left(\frac{p_X}{p_\theta}\right)\right]\ge f\!\left(\mathbb{E}_{p_\theta}\!\left[\frac{p_X}{p_\theta}\right]\right)=f(1)=0,$$
>
> since $\mathbb{E}_{p_\theta}\!\big[\tfrac{p_X}{p_\theta}\big]=\int p_\theta\cdot\tfrac{p_X}{p_\theta}\,dx=\int p_X\,dx=1$ and $f(1)=0$. So the divergence can never be negative.

---

## Question 8

Given the variational representation

$$D_f(P_X\|P_\theta)=\sup_{T\in\mathcal{T}}\big(\mathbb{E}_{x\sim P_X}[T(x)]-\mathbb{E}_{x\sim P_\theta}[f^*(T(x))]\big),$$

which statements correctly describe how this leads to a min–max problem? *(Select all that apply.)*

- **A.** The inner supremum over $T$ corresponds to the discriminator maximizing the difference between real and generated data.
- **B.** The generator minimizes the divergence by pushing $P_\theta$ closer to $P_X$.
- **C.** The convex conjugate $f^*$ ensures the $f$-divergence is symmetric and always tractable.
- **D.** The objective defines a saddle-point problem.
- **E.** The generator maximizes the divergence by generating more distinguishable samples.

> ### ✅ Answer: A, B, D
> The full objective is $\min_\theta\max_{T\in\mathcal{T}}\big(\mathbb{E}_{P_X}[T]-\mathbb{E}_{P_\theta}[f^*(T)]\big)$.
>
> - **A** — the inner $\sup_T$ is the discriminator/critic; maximizing it amounts to best distinguishing real ($P_X$) from generated ($P_\theta$) samples.
> - **B** — the outer $\min_\theta$ trains the generator $g_\theta$ to push $P_\theta\to P_X$.
> - **D** — a $\min\max$ objective *is* a saddle-point problem by definition.
>
> **C is false:** $f^*$ buys variational *tractability*, not symmetry — most $f$-divergences (e.g. KL) are asymmetric. **E is false:** roles reversed — the generator *minimizes* the divergence to make samples *less* distinguishable; the discriminator is the one that maximizes.

---

## Question 9

Which of the following are benefits of variational divergence minimization? *(Select all that apply.)*

- **A.** Allows divergence estimation via samples instead of density functions.
- **B.** Requires access to exact density ratios between model and data.
- **C.** Can be implemented using neural networks for both the generator and discriminator.
- **D.** Provides flexibility to choose different divergence measures.

> ### ✅ Answer: A, C, D
> - **A** — the central benefit: the variational form uses only *samples* from $P_X$ (the dataset) and $P_\theta$ (outputs of $g_\theta$), so densities are never needed.
> - **C** — both $g_\theta$ and the critic $T$ are neural networks (the $f$-GAN architecture).
> - **D** — varying the convex function $f$ yields different divergences (KL, reverse KL, JS, TV).
>
> **B is false** — and it is the *opposite* of a benefit. The whole point is to *avoid* needing exact density ratios by working with samples and the variational lower bound.

---

## Question 10

With conjugate $f^*(t)=\sup_{u>0}(tu-f(u))$, in the variational representation of the $f$-divergence, what does the optimal $T(x)$ represent?

- **A.** An estimate of the likelihood ratio $\frac{p_X(x)}{p_\theta(x)}$.
- **B.** The output of a neural network discriminator $T_\vartheta(x)$.
- **C.** The true divergence value at each sample point.

> ### ✅ Answer: A
> The inner supremum is tight where $T^*(x)=f'\!\big(\tfrac{p_X(x)}{p_\theta(x)}\big)$ — a function of the density ratio. So the optimal variational function *encodes (an estimate of) the likelihood ratio* $\tfrac{p_X}{p_\theta}$. Option B describes the *parameterization* we optimize over, not what the optimum *represents*. (See Q14 for the precise functional form.)

---

## Question 11

You observe a dataset $D=\{x_i\}_{i=1}^n$ with $x_i\in\mathbb{R}^d$ and assume $x_i\overset{\text{i.i.d.}}{\sim}p_X$. Which statement is correct?

- **A.** The coordinates inside each $x_i$ are independent.
- **B.** The samples $x_i$ are independent draws from the same $p_X$, but coordinates within each $x_i$ may be dependent.
- **C.** $p_X(x)=\prod_{j=1}^d p_X(x_j)$ must hold.
- **D.** $p_X$ must be Gaussian.

> ### ✅ Answer: B
> "i.i.d." is a statement about the *samples* (data points): each $x_i$ is an independent draw from the same distribution $p_X$. It says nothing about the internal structure of a single $x_i$ — the coordinates within one vector can be strongly dependent (e.g. neighboring pixels in an image). A and C wrongly assert coordinate-wise independence; D wrongly assumes a Gaussian form.

---

## Question 12

Let $z\sim p_z$ and $x=g_\theta(z)$. The induced distribution $p_\theta$ on $x$ is best described as:

- **A.** $p_\theta(x)=p_z(g_\theta(x))$.
- **B.** $p_\theta$ is the pushforward distribution of $p_z$ under $g_\theta$.
- **C.** $p_\theta(x)=\int p_z(z)\,dz$ (a constant).
- **D.** $p_\theta(x)=p_X(x)$ by construction.

> ### ✅ Answer: B
> Sampling $z\sim p_z$ and applying the deterministic map $x=g_\theta(z)$ produces, by definition, the *pushforward* measure of $p_z$ under $g_\theta$ (written $p_\theta=(g_\theta)_\#\,p_z$). A misapplies the map and omits the Jacobian (a valid change-of-variables density would be $p_z(g_\theta^{-1}(x))\,|\det J|$). C is a meaningless constant ($\int p_z\,dz=1$). D holds only *at the optimum*, not by construction.

---

## Question 13

Using the variational form with $f(u)=u\log u$ and $f^*(t)=e^{t-1}$, the inner objective becomes:

- **A.** $\sup_T\big(\mathbb{E}_{p_X}[T]-\mathbb{E}_{p_\theta}[\log(1+e^{T})]\big)$
- **B.** $\sup_T\big(\mathbb{E}_{p_X}[T]-\mathbb{E}_{p_\theta}[e^{T-1}]\big)$
- **C.** $\sup_T\big(\mathbb{E}_{p_\theta}[T]-\mathbb{E}_{p_X}[e^{T-1}]\big)$
- **D.** $\inf_T\big(\mathbb{E}_{p_X}[T]-\mathbb{E}_{p_\theta}[e^{T-1}]\big)$

> ### ✅ Answer: B
> Substitute $f^*(T(x))=e^{T(x)-1}$ into $\sup_T\big(\mathbb{E}_{p_X}[T]-\mathbb{E}_{p_\theta}[f^*(T)]\big)$. The first expectation is over $p_X$, the second over $p_\theta$ — option B. (Check: $f^*(t)=\sup_u(tu-u\log u)$; setting the derivative $t-\log u-1=0$ gives $u=e^{t-1}$, hence $f^*(t)=e^{t-1}$.) A uses the *JS* conjugate $\log(1+e^T)$; C swaps the two measures; D uses $\inf$ instead of $\sup$.

---

## Question 14

For a differentiable convex $f$, the tightness (Fenchel–Young equality) condition implies the optimal $T^*(x)$ satisfies:

- **A.** $T^*(x)=f\!\left(\frac{p_X(x)}{p_\theta(x)}\right)$
- **B.** $T^*(x)=f'\!\left(\frac{p_X(x)}{p_\theta(x)}\right)$
- **C.** $T^*(x)=\left(\frac{p_X(x)}{p_\theta(x)}\right)^{-1}$
- **D.** $T^*(x)=\log\frac{p_\theta(x)}{p_X(x)}$

> ### ✅ Answer: B
> The Fenchel–Young inequality $f(u)+f^*(t)\ge ut$ holds with equality exactly when $t=f'(u)$. In the bound the ratio plays the role of $u=\tfrac{p_X}{p_\theta}$, so the optimal critic is $T^*(x)=f'\!\big(\tfrac{p_X(x)}{p_\theta(x)}\big)$. This refines Q10: $T^*$ *encodes* the likelihood ratio (Q10), and its precise form is $f'$ *of* that ratio. Option A applies $f$ instead of $f'$; C and D coincide with $f'$ only for particular $f$, not in general.

---

## Question 15

Let $L(\theta)=\mathbb{E}_{z\sim p_z}[\ell(g_\theta(z))]$ with $z$ independent of $\theta$. Which gradient expression is correct?

- **A.** $\nabla_\theta L=\mathbb{E}_z[\ell(g_\theta(z))\,\nabla_\theta\log p_z(z)]$
- **B.** $\nabla_\theta L=\mathbb{E}_z\big[J_{g_\theta}(z)^\top\nabla_x\ell(x)\,\big|\,x=g_\theta(z)\big]$
- **C.** $\nabla_\theta L=\nabla_\theta\int p_z(z)\,dz$
- **D.** $\nabla_\theta L=\mathbb{E}_{x\sim p_X}[\nabla_\theta\ell(x)]$

> ### ✅ Answer: B
> Because the latent distribution $p_z$ does not depend on $\theta$, the gradient moves inside the expectation and the chain rule gives the Jacobian of $g_\theta$ times the gradient of $\ell$:
>
> $$\nabla_\theta L=\mathbb{E}_z\big[J_{g_\theta}(z)^\top\,\nabla_x\ell(x)\big|_{x=g_\theta(z)}\big].$$
>
> This is the **pathwise / reparameterization** gradient. A is a score-function (REINFORCE) term, but here $\nabla_\theta\log p_z(z)=0$ since $p_z\perp\theta$, so it vanishes. C is zero ($\int p_z\,dz=1$, constant). D differentiates over the data measure $p_X$, which has no $\theta$ dependence.

---

## Answer key at a glance

| Q  | Ans | Q   | Ans     | Q   | Ans       |
|----|-----|-----|---------|-----|-----------|
| 1  | A   | 6   | D       | 11  | B         |
| 2  | C   | 7   | C       | 12  | B         |
| 3  | B   | 8   | A, B, D | 13  | B         |
| 4  | B   | 9   | A, C, D | 14  | B         |
| 5  | B   | 10  | A       | 15  | B         |

### A note on Q10 vs Q14

These two can look contradictory but are not. Q10 asks, among coarse options, what the optimal $T^*$ *represents* — "the likelihood ratio" (A). Q14 asks for the *precise* functional form under Fenchel–Young tightness — $f'$ *of* that ratio (B). The optimal critic is $T^*(x)=f'\!\big(\tfrac{p_X}{p_\theta}\big)$: a monotone transform of the density ratio, so it "encodes" the ratio without being equal to it for a general $f$.

---

*Reference notes for personal study. Math rendered via GitHub's native LaTeX support.*
