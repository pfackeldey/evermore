# evermore for CMS

If you are coming from the CMS experiment, you are probably familiar with the
[{math}`\Combine`](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/) project.

In the following, you will find a brief comparison how modifier types of
{math}`\Combine` can be implemented in evermore.

## Simple Example (Combine)

This is a simple one-bin example of a signal process scaled by an unconstrained modifier {math}`\mu` and a background process with a normalization uncertainty (lnN).

::::{tab-set}
:::{tab-item} Combine

```{code-block}
:caption: datacard.txt {octicon}`file;1em`
# run this datacard with the combine tool
imax 1  number of channels
jmax 1  number of processes -1
kmax *  number of nuisance parameters (sources of systematical uncertainties)
-------
bin                   bin1
observation           51
-------
bin                   bin1    bin1
process               signal  background
process               0       1
rate                  12      50
-------
bkg_norm    lnN       -       0.9/1.1
```

:::

:::{tab-item} evermore <img src="../assets/favicon.png" height="1.5em">

```{code-block} python
import jax
import jax.numpy as jnp
import evermore as evm


jax.config.update("jax_enable_x64", True)

params = {"mu": evm.Parameter(value=1.0), "bkg_norm": evm.NormalParameter(value=0.0)}

hists = {"signal": jnp.array([12.0]), "background": jnp.array([50.0])}

data = jnp.array([51.0])


def model(params: dict, hists: dict) -> jnp.ndarray:
    mu_modifier = params["mu"].scale()
    syst_modifier = params["bkg_norm"].scale_log(up=1.1, down=0.9)
    return mu_modifier(hists["signal"]) + syst_modifier(hists["background"])


# eval model to get expectation
model(params, hists)
# -> Array([62.], dtype=float64)

model({"mu": evm.Parameter(value=0.5), "bkg_norm": evm.NormalParameter(value=1.12)}, hists)
# -> Array([61.63265822], dtype=float64)
```

:::
::::

## Modifier/Effect Types

For a more detailed overview of modifier/effect types in {math}`\Combine`, please refer to
the [{math}`\Combine`](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/what_combine_does/model_and_likelihood/#likelihoods-implemented-in-combine) documentation.

### Normalization Effects (lnN)

See [normalization effects](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/what_combine_does/model_and_likelihood/#normalization-effects).

::::{tab-set}
:::{tab-item} Combine

```{code-block}
:caption: datacard.txt {octicon}`file;1em`
[...]
-------
norm_sys   lnN   0.9/1.1
```

:::

:::{tab-item} evermore <img src="../assets/favicon.png" height="1.5em">

```{code-block} python
import jax.numpy as jnp
import evermore as evm


param = evm.NormalParameter()

norm_sys = evm.Modifier(
    parameter=param,
    effect=evm.effect.AsymmetricExponential(up=1.1, down=0.9),
)

# or short-hand:
norm_sys = param.scale_log(up=1.1, down=0.9)
```

:::
::::

### Shape Morphing Effects (shape)

See [shape morphing effects](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/what_combine_does/model_and_likelihood/#shape-morphing-effects).

::::{tab-set}
:::{tab-item} Combine

```{code-block}
:caption: datacard.txt {octicon}`file;1em`
shapes * * shapes.root $PROCESS $PROCESS_$SYSTEMATIC
[...]
-------
shape_sys   shape   1.0
```

:::

:::{tab-item} evermore <img src="../assets/favicon.png" height="1.5em">

```{code-block} python
import jax.numpy as jnp
import evermore as evm


param = evm.NormalParameter()

shape_sys = evm.Modifier(
    parameter=param,
    effect=evm.effect.VerticalTemplateMorphing(
        up_template=[...],  # histogram from `shapes.root`, matching `$PROCESS $PROCESS_$SYSTEMATIC` ("Up") pattern
        down_template=[...],  # histogram from `shapes.root`, matching `$PROCESS $PROCESS_$SYSTEMATIC` ("Down") pattern
    )
)

# or short-hand:
shape_sys = param.morphing(
    up_template=[...],  # histogram from `shapes.root`, matching `$PROCESS $PROCESS_$SYSTEMATIC` ("Up") pattern
    down_template=[...],  # histogram from `shapes.root`, matching `$PROCESS $PROCESS_$SYSTEMATIC` ("Down") pattern
)
```

:::
::::

### Statistical Uncertainties (autoMCstats)

See [autoMCstats](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/part2/bin-wise-stats/).

::::{tab-set}
:::{tab-item} Combine

```{code-block}
:caption: datacard.txt {octicon}`file;1em`
# run this datacard with the combine tool
imax 1  number of channels
jmax 1  number of processes -1
kmax *  number of nuisance parameters (sources of systematical uncertainties)
-------
bin                   bin1
observation           51
-------
bin                   bin1        bin1        bin1
process               signal      bkg1        bkg2
process               0           1           2
rate                  12          50          30
-------
bin1 autoMCStats 10 [include-signal = 0] [hist-mode = 1]
```

:::

:::{tab-item} evermore <img src="../assets/favicon.png" height="1.5em">

```{code-block} python
from operator import itemgetter
import jax.numpy as jnp
import evermore as evm


hists = {"signal": jnp.array([12]), "bkg1": jnp.array([50]), "bkg2": jnp.array([30])}

# `histsw2` corresponds to sumw2 of the TH1 histograms
histsw2 = {"signal": jnp.array([12]), "bkg1": jnp.array([50]), "bkg2": jnp.array([30])}

# Additional `Combine` options:
# if `[hist-mode 2]`: <not available in evermore>
# if `[include-signal 0]`:
#     hists.pop("signal")
#     histsw2.pop("signal")

staterrors = evm.staterror.StatErrors(hists, histsw2, threshold=10.0)

# Create a modifier for the qcd process, `getter` is a function
# that finds the corresponding parameter from `staterrors.params_per_process`
getter = itemgetter("bkg1")
mod = staterrors.modifier(getter=getter)
# apply the modifier
mod(getter(hists))
```

:::
::::

### Rate Parameters (rateParam)

See [rateParam](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/part2/settinguptheanalysis/#rate-parameters).

::::{tab-set}
:::{tab-item} Combine

```{code-block}
:caption: datacard.txt {octicon}`file;1em`
[...]
-------
sys   rateParam   bin1   process1  1.0   [0.0,2.0]
```

:::

:::{tab-item} evermore <img src="../assets/favicon.png" height="1.5em">

```{code-block} python
import jax.numpy as jnp
import evermore as evm


param = evm.Parameter(value=1, lower=0, upper=2)

sys = evm.Modifier(
    parameter=param,
    effect=evm.effect.Linear(slope=1.0, offset=0.0),
)

# or short-hand:
sys = param.scale()
```

:::
::::
