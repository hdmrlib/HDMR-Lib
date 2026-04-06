# Fundamentals

This page introduces the main mathematical ideas behind **High-Dimensional Model Representation (HDMR)** and **Enhanced Multivariate Products Representation (EMPR)** in **HDMR-Lib**.

## Why HDMR and EMPR?

High-dimensional functions and tensors are difficult to analyze directly because the number of possible interactions grows rapidly with dimension. HDMR and EMPR address this by representing a multivariate object through lower-order terms, which makes approximation and interaction analysis more manageable.

## High-Dimensional Model Representation (HDMR)

Let $f(\mathbf{x}) = f(x_1, x_2, \dots, x_d)$ be a multivariate function.

HDMR represents $f$ as a hierarchy of lower-order terms:

```{math}
f(\mathbf{x}) = f_0
+ \sum_{i=1}^{d} f_i(x_i)
+ \sum_{1 \le i < j \le d} f_{ij}(x_i, x_j)
+ \cdots
+ f_{12\ldots d}(x_1, \ldots, x_d)
```

Here, $f_0$ is a constant term, $f_i(x_i)$ are first-order terms, and $f_{ij}(x_i, x_j)$ are second-order interaction terms.

The key idea is that many systems are dominated by low-order effects, so a truncated expansion often provides a useful approximation.

## Enhanced Multivariate Products Representation (EMPR)

EMPR follows the same general idea: a multivariate object is expressed through constant, univariate, bivariate, and higher-order terms.

Like HDMR, EMPR organizes information hierarchically by interaction order. Its main distinction is the use of **support functions**, which define how lower-order terms are extended across the remaining variables.

This makes EMPR a support-based representation of multivariate structure rather than only a purely additive decomposition.

## Terms and Components

Both HDMR and EMPR are organized by term order.

### Zeroth-order term

The zeroth-order term $f_0$ is the constant baseline. It captures the global reference level of the function or tensor.

### First-order terms

A first-order term depends on a single variable, for example $f_i(x_i)$. It describes the isolated contribution of one variable, independent of interactions with other variables.

### Second-order terms

A second-order term depends on two variables, for example $f_{ij}(x_i, x_j)$. It captures pairwise interactions that cannot be explained by the corresponding first-order terms alone.

### Higher-order terms

Higher-order terms involve three or more variables, such as $f_{ijk}(x_i, x_j, x_k)$. These terms represent increasingly complex interactions.

### Truncation

A decomposition truncated at order $m$ keeps only terms up to that order:

```{math}
f^{(m)}(\mathbf{x}) = \sum_{\lvert u \rvert \le m} f_u(\mathbf{x}_u)
```

This is the mathematical basis for lower-order approximation and reconstruction.

## Support Vectors and Weights

In **HDMR-Lib**, both HDMR and EMPR are implemented for tensor-valued data using per-dimension support vectors.

Let the tensor have $d$ dimensions. Then each dimension is associated with a support vector $s_i \in \mathbb{R}^{n_i}$, where $n_i$ is the size of the $i$-th mode.

### HDMR in the library

In the current HDMR implementation, the constant term is obtained by successive contractions of the tensor with weighted support vectors:

```{math}
g_0
=
G \times_1 (s_1 \odot w_1)^\top
\times_2 (s_2 \odot w_2)^\top
\cdots
\times_d (s_d \odot w_d)^\top
```

Here:

- $G$ is the input tensor
- $s_i$ is the support vector for dimension $i$
- $w_i$ is the corresponding weight vector
- $\odot$ denotes elementwise multiplication
- $\times_k$ denotes contraction along mode $k$

Higher-order HDMR terms are then obtained recursively after removing lower-order contributions.

### EMPR in the library

In the current EMPR implementation, the constant term is also obtained by successive contractions, but the construction uses support vectors together with per-dimension scalar normalization factors:

```{math}
g_0
=
\left(
\cdots
\left(
G \times_1 s_1^\top
\right)\alpha_1
\times_2 s_2^\top
\right)\alpha_2
\cdots
\times_d s_d^\top \alpha_d
```

In the current implementation,

```{math}
\alpha_i = \frac{1}{n_i}
```

for a mode of size $n_i$.

This reflects the current code path in **HDMR-Lib**: EMPR uses support vectors directly, while HDMR uses support vectors together with explicit weight vectors.

## HDMR and EMPR in Tensor Form

For tensor-valued data, the decomposition is not interpreted only as a function expansion, but also as a structured representation of a multiway array.

In the library implementation, a component term defined on a subset of dimensions is expanded back to the full tensor shape by combining it with support vectors along the remaining dimensions. This gives a tensor-level analogue of lower-order functional decomposition.

As a result, both methods support the same high-level ideas:

- decompose a high-dimensional object into lower-order terms
- retain only terms up to a chosen order
- reconstruct an approximation from the retained terms

## HDMR vs EMPR

HDMR and EMPR are closely related, but they are not identical.

### Common structure

Both methods:

- organize information hierarchically by order
- represent a multivariate object through lower-order terms
- support truncation to obtain lower-order approximations

### Main difference

The main conceptual difference is the role of support functions.

- **HDMR** is most naturally introduced as a hierarchical functional decomposition
- **EMPR** uses support functions explicitly as part of the representation

In the current **HDMR-Lib** implementation, both methods operate on tensors and both rely on support vectors at the implementation level. The difference appears in how the component terms are computed:

- HDMR uses weighted support vectors
- EMPR uses support vectors together with per-dimension scalar normalization

## References

- H. Rabitz and Ö. F. Aliş, *General foundations of high-dimensional model representations*, Journal of Mathematical Chemistry, 1999.
- B. Tunga and M. Demiralp, *The influence of the support functions on the quality of enhanced multivariance product representation*, Journal of Mathematical Chemistry, 2010.
