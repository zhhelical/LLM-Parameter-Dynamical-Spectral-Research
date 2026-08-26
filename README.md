# LLM Knowledge–Parameter Dynamics Research

## A Unified Research Framework for Knowledge–Parameter Topology, Jacobian Spectral Analysis, and Continual Parameter Plasticity

**Research program:** LLM 与人脑探索  
**Date:** August 2026  
**Status:** Theoretical research program / pre-publication research record

---

## Abstract

This repository presents a unified theoretical research program for understanding how knowledge may be organized, identified, and dynamically updated inside Large Language Model (LLM) parameter spaces.

The research consists of three closely related works:

1. **RL-KPT — Reinforcement-Learned Knowledge–Parameter Topology**  
   Proposes that semantic knowledge can be represented in an explicit knowledge space and that reinforcement learning can learn a reusable mapping from knowledge states to parameter-update regions.

2. **RL-KPT-SI — Jacobian-Based Spectral Analysis of LLM Parameter Systems**  
   Investigates the mathematical feasibility of using Jacobian matrices and related spectral methods to analyze the structure of LLM parameter systems.

3. **LLM Parameter Dynamical–Spectral Theory**  
   Provides the theoretical foundation connecting the preceding two works by treating the LLM parameter state as a high-dimensional dynamical system:
   \[
   \theta \rightarrow \theta(t),
   \]
   and by studying the coupled evolution of knowledge state, parameter state, Jacobian state, and spectral state.

The unified research hypothesis is:

\[
\boxed{
K(t)\longleftrightarrow\theta(t)\longleftrightarrow J(t),S(t)
}
\]

The ultimate objective is to establish a mathematically grounded route toward **continual parameter-level learning**, in which an LLM can identify where newly acquired knowledge affects its internal parameter system and selectively modify relevant parameter subspaces while limiting interference with previously learned knowledge.

---

# 1. Research Motivation

Current LLMs are generally trained according to

\[
\theta^{*}=\arg\min_\theta L(\theta;D),
\]

after which the learned parameter state is largely treated as fixed.

For continual learning, however,

\[
\theta_{t+1}=\theta_t+\Delta\theta_t.
\]

The fundamental question is:

> **Which part of the parameter system should change when a particular piece of knowledge is acquired?**

A complete continual-learning system should solve:

1. **Knowledge identification** — What kind of knowledge has arrived?
2. **Parameter-state diagnosis** — Which internal parameter regions are associated with that knowledge?
3. **Controlled plasticity** — Which regions should actually be modified?

The research program addresses these through:

\[
\boxed{
\text{Knowledge Space}
\rightarrow
\text{Parameter Space}
\rightarrow
\text{Spectral/Dynamical State}
\rightarrow
\text{Parameter Routing}
\rightarrow
\text{Continual Plasticity}
}
\]

---

# 2. The Three Research Works

## 2.1 RL-KPT

### Reinforcement-Learned Knowledge–Parameter Topology

RL-KPT proposes that an LLM parameter space may contain structured regions associated with different knowledge domains.

The conceptual mapping is:

\[
x\rightarrow K\rightarrow\text{RL Router}\rightarrow P\rightarrow\Delta\theta.
\]

Define the knowledge space:

\[
\mathcal K=\{K_1,K_2,\ldots,K_m\}.
\]

A knowledge item is encoded as

\[
z_x=f_K(x).
\]

The parameter space is

\[
\mathcal P=\{P_1,P_2,\ldots,P_q\}.
\]

A knowledge region does not need to correspond to one individual parameter or one contiguous tensor block. It may instead be a distributed subspace:

\[
P_K=\operatorname{span}\{v_1,v_2,\ldots,v_r\}.
\]

For example:

\[
P_{\mathrm{robotics}}
\approx
P_{\mathrm{mechanics}}
\cup
P_{\mathrm{control}}
\cup
P_{\mathrm{physics}}
\cup
P_{\mathrm{mathematics}}.
\]

The RL controller generates a parameter-update mask:

\[
M_t=\pi_\phi(S_t).
\]

The localized update is

\[
\boxed{
\theta_{t+1}=\theta_t+M_t\odot\Delta\theta_t.
}
\]

The key distinction is:

> **Gradient descent determines how selected parameters should change; RL-KPT determines which parameter regions should be allowed to change.**

MoE routes computation:

\[
x\rightarrow\text{Router}\rightarrow\text{Expert}.
\]

RL-KPT routes parameter plasticity:

\[
x\rightarrow\text{Knowledge State}\rightarrow\text{RL Router}\rightarrow\text{Parameter Update Region}.
\]

---

# 3. RL-KPT Training Concept

For each training episode:

1. sample a knowledge instance;
2. encode its semantic knowledge state;
3. run the Transformer;
4. observe internal state;
5. select parameter regions using RL;
6. compute the local gradient update;
7. apply the selected mask;
8. evaluate new-knowledge acquisition;
9. evaluate retention;
10. evaluate transfer and interference;
11. calculate reward;
12. update the RL policy.

Formally:

\[
z_t=f_K(x_t),
\qquad
h_t=f_\theta(x_t),
\]

\[
S_t=(h_t,z_t,c_t,\theta_t),
\qquad
M_t=\pi_\phi(S_t),
\]

\[
\Delta\theta_t=-\eta\nabla_\theta L(x_t),
\]

\[
\theta_{t+1}
=
\theta_t+M_t\odot\Delta\theta_t.
\]

A possible reward is

\[
R_t=
\lambda_1R_{\mathrm{learn}}
+\lambda_2R_{\mathrm{retain}}
+\lambda_3R_{\mathrm{transfer}}
-\lambda_4R_{\mathrm{interference}}
-\lambda_5R_{\mathrm{cost}}.
\]

The learned policy is expected to approximate

\[
\boxed{
\pi_\phi:(K,S)\rightarrow M.
}
\]

---

# 4. RL-KPT-SI

## Jacobian-Based Spectral Analysis of LLM Parameter Systems

RL-KPT introduces the hypothesis that knowledge can be associated with parameter regions.

RL-KPT-SI asks:

> **How can the internal structure of the enormous parameter space be measured mathematically?**

The proposed route is

\[
\theta\rightarrow J(\theta)\rightarrow\sigma(J),
\]

where \(J(\theta)\) is an appropriate Jacobian and \(\sigma(J)\) represents spectral observables.

These may include:

- eigenvalues;
- singular values;
- eigenvectors;
- dominant invariant subspaces;
- spectral radius;
- spectral moments;
- eigengaps;
- singular-value distributions.

The point is not that an individual eigenvalue directly represents one fact. Rather, the spectrum may provide a mathematical coordinate system for studying parameter-system structure.

---

# 5. Why Spectral Analysis Matters

An LLM contains parameter matrices

\[
W_l\in\mathbb R^{d_l\times d_l'}.
\]

The complete parameter state can be represented as

\[
\theta=\operatorname{vec}(W_1,\ldots,W_L).
\]

For each matrix,

\[
W_l=U_l\Sigma_lV_l^\top.
\]

The singular values are

\[
\sigma_{l,1}\geq\sigma_{l,2}\geq\cdots\geq0.
\]

For the parameter dynamics,

\[
J_\theta=\frac{\partial F}{\partial\theta}.
\]

Its eigenvalues

\[
\lambda_1,\ldots,\lambda_N
\]

describe local dynamical properties.

The spectral radius is

\[
\rho(J_\theta)=\max_i|\lambda_i|.
\]

For continuous-time dynamics, local stability is related to

\[
\max_i\operatorname{Re}(\lambda_i).
\]

Therefore, Jacobian spectra can describe **how the parameter system responds to perturbations**, not merely its static matrix structure.

---

# 6. The Missing Theoretical Layer

RL-KPT and RL-KPT-SI naturally lead to:

> If parameter space has structure and that structure can be analyzed spectrally, how does it evolve when knowledge changes?

This requires

\[
\theta\rightarrow\boxed{\theta(t)}.
\]

The parameter system becomes a dynamical system.

---

# 7. LLM Parameter Dynamical–Spectral Theory

## 7.1 Parameter State

Let

\[
\theta(t)\in\Theta
\]

represent the internal parameter state.

A continual learning process is

\[
\theta_{t+1}=\theta_t+\Delta\theta_t.
\]

For localized plasticity:

\[
\theta_{t+1}
=
\theta_t+M_t\odot\Delta\theta_t.
\]

A continuous approximation is

\[
\boxed{
\frac{d\theta}{dt}=F(\theta,x,r,u,t).
}
\]

Here:

- \(x\): knowledge-bearing input;
- \(r\): feedback or reward;
- \(u\): control;
- \(F\): parameter evolution field.

---

# 8. Jacobian as the Local Dynamical Operator

Define

\[
\boxed{
J_\theta=\frac{\partial F}{\partial\theta}.
}
\]

Around a local state \(\theta^*\):

\[
\dot{\delta\theta}\approx J_\theta\delta\theta.
\]

Therefore,

\[
\delta\theta(t)\approx e^{J_\theta t}\delta\theta(0).
\]

The Jacobian thus describes how perturbations propagate through the parameter dynamical system.

This provides the theoretical foundation for the Jacobian spectral analysis proposed by RL-KPT-SI.

---

# 9. Three Coupled Spaces

The unified theory introduces:

### Knowledge space

\[
\mathcal K
\]

representing semantic knowledge.

### Parameter space

\[
\mathcal P\subseteq\Theta
\]

representing internal parameter state and functional subspaces.

### Spectral state space

\[
\mathcal S
\]

representing measurable spectral characteristics.

The central relation is

\[
\boxed{
K(t)\longleftrightarrow\theta(t)\longleftrightarrow S(t).
}
\]

More explicitly:

\[
K(t)=\Phi(\theta(t))
\]

and

\[
S(t)=\Psi(\theta(t),J_\theta(t)).
\]

---

# 10. Dynamic Knowledge–Parameter Mapping

If

\[
K(t)=\Phi(\theta(t)),
\]

then

\[
\boxed{
\dot K(t)=D\Phi(\theta(t))\dot\theta(t).
}
\]

If

\[
S(t)=\Psi(\theta(t),J_\theta(t)),
\]

then

\[
\boxed{
\dot S(t)
=
D\Psi
\begin{bmatrix}
\dot\theta(t)\\
\dot J_\theta(t)
\end{bmatrix}.
}
\]

Therefore:

\[
\boxed{
\text{Knowledge Change}
\leftrightarrow
\text{Parameter Change}
\leftrightarrow
\text{Spectral Change}.
}
\]

The research question becomes:

> **Can the evolution of spectral and parameter structures dynamically track the evolution of knowledge?**

---

# 11. Knowledge Regions as Parameter Subspaces

For knowledge class \(K_i\), define

\[
G_i
=
\mathbb E_{x\sim K_i}
[g_xg_x^\top],
\qquad
g_x=\nabla_\theta\ell(\theta;x).
\]

The dominant eigenspace is

\[
\mathcal U_i=\operatorname{TopEig}(G_i).
\]

Thus a knowledge-responsive region can be defined as

\[
\boxed{
P_i\approx\operatorname{TopEig}(G_i).
}
\]

This region may be distributed across:

- layers;
- attention heads;
- MLP channels;
- matrix blocks;
- low-rank directions;
- cross-layer subspaces.

---

# 12. Knowledge Overlap and Principal Angles

For two knowledge domains \(K_i,K_j\), let their dominant subspaces be \(\mathcal U_i,\mathcal U_j\).

Principal angles

\[
0\leq\alpha_1\leq\cdots\leq\alpha_r\leq\frac\pi2
\]

measure their geometric relationship.

A possible similarity measure is

\[
\operatorname{Sim}(i,j)
=
\frac1r\sum_{k=1}^{r}\cos^2\alpha_k.
\]

This provides a mathematical route to studying interdisciplinary knowledge.

For example:

\[
P_{\mathrm{robotics}}
\approx
P_{\mathrm{mechanics}}
\cup
P_{\mathrm{control}}
\cup
P_{\mathrm{physics}}
\cup
P_{\mathrm{mathematics}}.
\]

---

# 13. Spectral Perturbation and Knowledge Change

Suppose

\[
W\rightarrow W+\Delta W.
\]

For singular values,

\[
|\sigma_i(W+\Delta W)-\sigma_i(W)|
\leq
\|\Delta W\|_2.
\]

Define parameter displacement:

\[
D_\theta=\|\theta'-\theta\|_2.
\]

Define spectral displacement:

\[
D_S(W,W')
=
\|\Sigma(W)-\Sigma(W')\|_2.
\]

For multiple layers:

\[
D_S^{(L)}
=
\sum_{l=1}^{L}
w_l
\|\Sigma_l(W_l')-\Sigma_l(W_l)\|_2.
\]

The empirical chain is:

\[
\boxed{
\Delta K\rightarrow\Delta\theta\rightarrow\Delta S.
}
\]

---

# 14. Stability Means Controlled Plasticity

The topology itself may evolve:

\[
\mathcal P(t+1)\neq\mathcal P(t).
\]

Therefore, the desired property is not absolute stationarity but

\[
\boxed{\text{controlled plasticity}.}
\]

Define parameter-subspace drift:

\[
D_P(i;t,t+\Delta t)
=
d(\mathcal U_i(t),\mathcal U_i(t+\Delta t)).
\]

Define spectral drift:

\[
D_S(t,t+\Delta t)
=
d_S(S(t),S(t+\Delta t)).
\]

Preserved knowledge should ideally exhibit limited drift, while newly acquired knowledge may intentionally change relevant regions.

---

# 15. Complementary Mathematical Tools

## Hessian

\[
H(\theta)=\nabla_\theta^2L(\theta).
\]

Knowledge-conditioned curvature:

\[
H_i
=
\mathbb E_{x\sim K_i}
[
\nabla_\theta^2\ell(\theta;x)
].
\]

## Fisher Information

\[
F(\theta)
=
\mathbb E
[
\nabla_\theta\log p_\theta
\nabla_\theta\log p_\theta^\top
].
\]

Knowledge-conditioned Fisher matrix:

\[
F_i
=
\mathbb E_{x\sim K_i}
[
\nabla_\theta\log p_\theta
\nabla_\theta\log p_\theta^\top
].
\]

## Gradient Covariance

\[
G_i=\mathbb E[g_xg_x^\top].
\]

Its dominant eigenspace identifies major learning directions associated with a knowledge class.

## Mutual Information

Let \(Z_S\) be a spectral representation and \(Z_K\) a knowledge representation:

\[
I(Z_S;Z_K).
\]

A useful spectral representation should contain measurable information about knowledge classes.

---

# 16. Existence, Identifiability, and Stability

The framework distinguishes:

### Existence

Does a relationship exist?

\[
K\leftrightarrow S
\]

### Identifiability

Can knowledge classes be distinguished from spectral observables?

\[
S(K_i)\neq S(K_j)
\]

in a statistically meaningful sense?

### Stability

Does the relationship remain sufficiently consistent under:

- different examples;
- perturbations;
- model checkpoints;
- sequential learning;
- domain changes?

The theory does **not** assume that a single eigenvalue uniquely identifies a fact.

---

# 17. Why Reinforcement Learning Becomes Important

The dynamical theory does not require RL to define the parameter dynamics.

RL becomes the **active control layer**.

Let

\[
o_t=\Omega(h_t,z_t,S_t,J_t).
\]

Then

\[
u_t=\pi_\phi(o_t).
\]

The parameter dynamics become

\[
\boxed{
\dot\theta=F(\theta,x,r,\pi_\phi(o)).
}
\]

The complete active loop is:

\[
\boxed{
\text{Observe}
\rightarrow
\text{Analyze}
\rightarrow
\text{Decide}
\rightarrow
\text{Adapt}
\rightarrow
\text{Re-observe}.
}
\]

The roles are:

| Component | Function |
|---|---|
| Knowledge space | Semantic state representation |
| Parameter dynamics | Internal state evolution |
| Jacobian | Local dynamic sensitivity |
| Spectral analysis | Structural observation |
| RL | Active plasticity control |
| Gradient optimization | Numerical parameter update |

---

# 18. Unified Continual-Learning Architecture

```text
Incoming experience / new knowledge
                |
                v
      Semantic knowledge state
                |
                v
       Transformer parameter state
                |
                v
       Parameter matrices W_l(t)
                |
                v
 +--------------------------------+
 | Jacobian / Spectral / Hessian  |
 | Fisher / Gradient analysis     |
 +--------------------------------+
                |
                v
        Spectral state S_t
                |
                v
      Knowledge-region inference
                |
                v
          RL controller
                |
                v
       Parameter update mask
                |
                v
      Localized parameter plasticity
                |
                v
             theta_(t+1)
                |
                +---------------> repeat
```

The theoretical cycle is

\[
\boxed{
K_t
\rightarrow
\theta_t
\rightarrow
J_t,S_t
\rightarrow
\widehat K_t
\rightarrow
\theta_{t+1}.
}
\]

With RL:

\[
\boxed{
K_t
\rightarrow
\theta_t
\rightarrow
S_t
\rightarrow
\pi_\phi
\rightarrow
M_t
\rightarrow
\theta_{t+1}.
}
\]

---

# 19. Relationship Between the Three Works

| Work | Main question | Role |
|---|---|---|
| **RL-KPT** | Where should new knowledge be incorporated? | Knowledge–parameter topology and RL routing |
| **RL-KPT-SI** | How can parameter-system structure be mathematically analyzed? | Jacobian and spectral analysis |
| **LLM Parameter Dynamical–Spectral Theory** | How does parameter/spectral structure evolve as knowledge changes? | Mathematical foundation and dynamical framework |

The logical relationship is:

\[
\boxed{
\text{Parameter Dynamical Theory}
\rightarrow
\text{Jacobian/Spectral Analysis}
\rightarrow
\text{Knowledge–Parameter Topology}
\rightarrow
\text{Active Continual Learning}
}
\]

Viewed from knowledge:

\[
\boxed{
K(t)
\rightarrow
\theta(t)
\rightarrow
J(t)
\rightarrow
S(t)
\rightarrow
P_K(t)
\rightarrow
\Delta\theta(t).
}
\]

---

# 20. Static LLM vs Continually Plastic LLM

### Conventional static LLM

\[
\text{Data}
\rightarrow
\text{Training}
\rightarrow
\text{Fixed Parameters}
\]

### Proposed continually plastic LLM

\[
\text{Experience}
\rightarrow
\text{Knowledge Representation}
\rightarrow
\text{Parameter-State Diagnosis}
\rightarrow
\text{Parameter Routing}
\rightarrow
\text{Local Plasticity}
\rightarrow
\text{Updated Model}
\]

The proposed architecture treats the parameter space as a **structured and dynamically evolving substrate for knowledge**.

---

# 21. Core Research Hypotheses

### H1 — Knowledge–Parameter Topology

Semantically related knowledge should produce statistically distinguishable parameter-response or parameter-subspace patterns.

### H2 — Spectral Knowledge Separability

Knowledge classes should exhibit statistically distinguishable spectral signatures.

### H3 — Dynamic Knowledge Tracking

Knowledge acquisition should induce measurable trajectories in parameter and spectral states.

### H4 — Subspace Composition

Interdisciplinary knowledge should correspond to overlapping or compositional parameter subspaces.

### H5 — Controlled Plasticity

Preserved knowledge regions should exhibit lower parameter/subspace/spectral drift than regions responsible for new learning.

### H6 — Routing Stability

Knowledge-to-parameter routing should remain sufficiently stable across examples within the same domain.

### H7 — Localized Learning

Localized updates should achieve competitive new-knowledge acquisition while modifying substantially fewer parameters than full fine-tuning.

### H8 — Reduced Interference

Localized plasticity should reduce degradation of unrelated knowledge.

### H9 — Continual Accumulation

Sequential knowledge should accumulate without complete retraining.

### H10 — Predictive Spectral State

Spectral state variables should improve prediction of future parameter changes or interference relative to appropriate baselines.

---

# 22. Experimental Program

## Experiment A — Static Parameter Topology

Use a small Transformer with controlled knowledge domains such as:

- mathematics;
- physics;
- biology;
- history;
- engineering.

Measure:

- weight spectra;
- singular-value distributions;
- parameter-response patterns;
- gradient covariance;
- Fisher spectrum;
- Hessian spectrum;
- layer-wise representations.

Test whether knowledge classes form distinguishable structures.

## Experiment B — Jacobian Spectral Analysis

Estimate Jacobian-vector products and leading spectral directions.

Compare model states before and after knowledge acquisition.

Measure:

\[
\Delta\lambda,
\qquad
\Delta\sigma,
\qquad
\Delta\rho(J),
\qquad
\Delta\mathcal U.
\]

## Experiment C — Knowledge-Induced Parameter Dynamics

Sequentially introduce

\[
K_1\rightarrow K_2\rightarrow K_3\rightarrow K_4.
\]

Record

\[
\theta_t,\quad W_l(t),\quad J_t,\quad S_t.
\]

Measure

\[
\Delta K_t,\quad\Delta\theta_t,\quad\Delta S_t.
\]

## Experiment D — Knowledge Subspace Classification

Construct

\[
\mathcal U_i=\operatorname{TopEig}(G_i)
\]

and compare principal-angle topology with semantic similarity.

## Experiment E — Continual Learning

Sequentially add knowledge and evaluate previously learned domains.

Define forgetting:

\[
F_i(t)
=
A_i(t_{\mathrm{before}})
-
A_i(t_{\mathrm{after}}).
\]

Compare forgetting with

\[
D_P(i),\qquad D_S(i),\qquad\rho(J_\theta).
\]

## Experiment F — RL-KPT Integration

Provide spectral state to the RL controller:

\[
o_t=(h_t,z_t,S_t,J_t).
\]

Generate

\[
M_t=\pi_\phi(o_t)
\]

and update:

\[
\theta_{t+1}
=
\theta_t+
M_t\odot\Delta\theta_t.
\]

---

# 23. Computational Scalability

A full Jacobian for an LLM with \(N\) parameters is

\[
J_\theta\in\mathbb R^{N\times N},
\]

which is generally impossible to materialize for frontier-scale models.

The framework therefore relies on approximate methods:

- Jacobian-vector products (JVP);
- vector-Jacobian products (VJP);
- Lanczos iteration;
- randomized low-rank eigensolvers;
- blockwise Jacobian analysis;
- layer-wise spectral summaries;
- low-rank tangent-space approximations;
- Kronecker-factored approximations.

The target is often the leading spectral subspace:

\[
\operatorname{TopEig}(J),
\qquad
\operatorname{TopSVD}(J).
\]

---

# 24. Falsifiability and Limitations

The framework remains explicitly falsifiable.

### No one-to-one knowledge encoding

It does not assert

\[
K_i\leftrightarrow W_j.
\]

Knowledge may be distributed, superposed, or represented through nonlinear combinations.

### Spectrum alone is not assumed to recover knowledge

It does not require

\[
K=\Phi(\sigma(W))
\]

to be globally invertible.

A more defensible hypothesis is

\[
S=\Psi(W,J)
\]

contains measurable information about knowledge organization and evolution.

### Nonstationary knowledge topology

The topology may change:

\[
\mathcal P(t+1)\neq\mathcal P(t).
\]

### Causality

Correlation between spectral change and knowledge change does not prove causal representation. Causal intervention and ablation are required.

### Scale

Full Jacobian analysis is currently impractical for very large models. Approximate methods are essential.

---

# 25. Broader Theoretical Significance

The central conceptual change is:

### Conventional view

\[
\boxed{
\theta=\text{optimization result}
}
\]

### Proposed view

\[
\boxed{
\theta(t)=\text{dynamic internal substrate}
}
\]

This changes the scientific question from:

> How do we optimize parameters?

to:

> **How is knowledge organized within the parameter system, how does that organization evolve, and how can the evolution be actively controlled?**

This creates three layers:

\[
\boxed{
\text{Dynamical Theory}=\text{State Model}
}
\]

\[
\boxed{
\text{Spectral Analysis}=\text{Observation Model}
}
\]

\[
\boxed{
\text{Reinforcement Learning}=\text{Control Model}
}
\]

---

# 26. Unified Research Roadmap

```text
Stage 1
Knowledge–Parameter Topology
        |
        v
RL-KPT
        |
        v
Stage 2
Jacobian / Spectral Analysis
        |
        v
RL-KPT-SI
        |
        v
Stage 3
LLM Parameter Dynamical Theory
        |
        v
K(t) <-> theta(t) <-> J(t), S(t)
        |
        v
Stage 4
Knowledge-region identification
        |
        v
Stage 5
Active RL parameter control
        |
        v
Stage 6
Continually plastic LLM
```

Long-term objective:

\[
\boxed{
\text{Knowledge-aware, dynamically observable, actively plastic LLM}
}
\]

---

# 27. Research Program in One Equation

\[
\boxed{
\begin{aligned}
z_t &= f_K(x_t),\\
\dot\theta_t &= F(\theta_t,x_t,r_t,u_t),\\
J_t &= \frac{\partial F}{\partial\theta_t},\\
S_t &= \Psi(\theta_t,J_t),\\
P_t &= \Gamma(z_t,S_t,J_t),\\
u_t &= \pi_\phi(z_t,S_t,J_t),\\
\theta_{t+1}
&=
\theta_t+
M_t\odot\Delta\theta_t.
\end{aligned}
}
\]

The complete transition is:

\[
\boxed{
K_t
\rightarrow
\theta_t
\rightarrow
J_t,S_t
\rightarrow
P_t
\rightarrow
\pi_\phi
\rightarrow
\theta_{t+1}.
}
\]

---

# 28. Current Scientific Position

This repository represents a **theoretical research hypothesis and mathematical framework**, not an experimentally established theory.

The current strongest claims are:

1. LLM parameters form mathematically analyzable matrices and tensors.
2. Jacobian, Hessian, Fisher, gradient covariance, singular-value, eigenvalue, and subspace methods provide legitimate mathematical tools for studying neural parameter systems.
3. Continual learning naturally produces a sequence of parameter states.
4. Treating the parameter state as \(	heta(t)\) provides a coherent dynamical-systems formulation.
5. Knowledge-conditioned parameter subspaces can be explicitly defined and experimentally tested.
6. Spectral-state trajectories can be investigated as candidate observables of knowledge organization and knowledge change.
7. RL provides a natural active-control mechanism for selecting parameter plasticity once such internal state representations are available.

The central unresolved empirical question is:

\[
\boxed{
\text{Does a stable and useful knowledge–parameter–spectral mapping actually exist in trained LLMs?}
}
\]

---

# 29. Related Works by the Author

## RL-KPT

**Reinforcement-Learned Knowledge–Parameter Topology for Continual Language Model Adaptation**

Core contribution:

\[
\boxed{
\text{Knowledge State}
\rightarrow
\text{RL Routing}
\rightarrow
\text{Parameter Region}
}
\]

## RL-KPT-SI

**RL-KPT-SI: Jacobian-Based Spectral Analysis of Large Language Model Parameter Systems**

Core contribution:

\[
\boxed{
\theta
\rightarrow
J(\theta)
\rightarrow
\text{Spectrum}
}
\]

## Present Theory

**A Dynamical–Spectral Theory of Knowledge Organization in Large Language Model Parameter Spaces**

Core contribution:

\[
\boxed{
K(t)
\leftrightarrow
\theta(t)
\leftrightarrow
J(t),S(t)
}
\]

Together:

\[
\boxed{
\text{RL-KPT}
+
\text{RL-KPT-SI}
+
\text{Parameter Dynamical Theory}
}
\]

form a unified research program for **knowledge organization, parameter dynamics, spectral structure, and continual parameter plasticity in LLMs**.

---

# 30. Future Work

The next major stage is empirical validation:

1. build a small Transformer testbed;
2. construct controlled knowledge domains;
3. measure parameter and spectral states;
4. test knowledge-region separability;
5. measure temporal spectral dynamics;
6. test causal parameter interventions;
7. integrate spectral observations into RL-KPT;
8. evaluate continual learning and interference;
9. scale the validated method toward larger language models.

The ultimate objective is to investigate whether an LLM can:

\[
\boxed{
\text{observe its internal knowledge structure}
\rightarrow
\text{identify where new knowledge belongs}
\rightarrow
\text{actively modify the appropriate parameter subspace}
\rightarrow
\text{retain and reorganize prior knowledge}.
}
\]

---

## Citation

The individual works should be cited according to their respective Zenodo records.

> **TODO:** Insert the final author name, Zenodo DOI links, publication dates, and license before public repository release.

---

## License

To be determined by the author before public repository release.
