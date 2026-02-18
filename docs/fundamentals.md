# Fundamentals

## What are HDMR and EMPR?

**HDMR (High-Dimensional Model Representation)** decomposes a multivariate function into a sum of terms with increasing interaction order:

- 1st-order terms: effects of individual variables
- 2nd-order terms: pairwise interactions
- ...
- up to the selected truncation **order**

This allows you to trade off expressiveness vs. complexity.

**EMPR (Enhanced Multivariate Products Representation)** is a closely related representation that emphasizes structured interaction components and is often used as an efficient feature representation in downstream tasks.

## What does `order` mean?

`order = 1`  
Only main effects (no interactions).

`order = 2`  
Main effects + pairwise interactions.

Higher orders include higher-order interactions but increase computation and the number of components.

## What is the `components()` output?

Most APIs in this style return a dictionary-like structure where keys identify the component and values store the corresponding factor/term.

A typical structure is:

- keys for main effects, e.g. `("x0",)`, `("x1",)` ...
- keys for interactions, e.g. `("x0","x3")` ...

The exact key type (tuple of indices vs. names) depends on the implementation, but the intent is consistent:
**each key corresponds to one contribution term** in the representation.

