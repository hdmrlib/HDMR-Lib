# User Guide

Use this section to prepare data, run decompositions, inspect components, and reconstruct lower-order approximations.

## Main workflow

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Prepare data
Format tensors, choose a backend, and validate input assumptions.

+++
[Prepare Input Data](prepare_data)
:::

:::{grid-item-card} Run decomposition
Compute HDMR or EMPR representations through the main workflow.

+++
[Run a Decomposition](run_a_decomposition)
:::

:::{grid-item-card} Inspect components
Examine lower-order terms and interaction structure.

+++
[Inspect Components](inspect_components)
:::

:::{grid-item-card} Reconstruct data
Build approximations from selected component orders.

+++
[Reconstruct Data](reconstruct_data)
:::
::::

## Additional topics

- [Quick Start](quick_start)
- [Backend support](backends)
- [Troubleshooting](troubleshoot)

```{toctree}
:maxdepth: 1
:hidden:

quick_start
prepare_data
run_a_decomposition
inspect_components
reconstruct_data
backends
troubleshoot