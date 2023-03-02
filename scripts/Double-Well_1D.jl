#' ---
#' title : 1D Langevin Dynamics in a Double Well Potential
#' author : Yang-yang Tan
#' date : `j import Dates; Dates.Date(Dates.now())`
#' ---

#' # Introduction

#' This an example of a julia script that can be published using
#' [Weave](http://weavejl.mpastell.com/dev/usage/).
#' The script can be executed normally using Julia
#' or published to HTML or pdf with Weave.
#' Text is written in markdown in lines starting with "`#'` " and code
#' is executed and results are included in the published document.

#' Notice that you don't need to define chunk options, but you can using
#' `#+`. just before code e.g. `#+ term=True, caption='Fancy plots.'`.
#' If you're viewing the published version have a look at the
#' [source](FIR_design.jl) to see the markup.


#' # FIR Filter Design

#' We'll implement lowpass, highpass and ' bandpass FIR filters. If
#' you want to read more about DSP I highly recommend [The Scientist
#' and Engineer's Guide to Digital Signal
#' Processing](http://www.dspguide.com/) which is freely available
#' online.

#' ## Calculating frequency response

#' DSP.jl package doesn't (yet) have a method to calculate the
#' the frequency response of a FIR filter so we define it:


#' ## Design Lowpass FIR filter

#' Designing a lowpass FIR filter is very simple to do with DSP.jl, all you
#' need to do is to define the window length, cut off frequency and the
#' window. We will define a lowpass filter with cut off frequency at 5Hz for a signal
#' sampled at 20 Hz.
#' We will use the Hamming window, which is defined as:
#' $w(n) = \alpha - \beta\cos\frac{2\pi n}{N-1}$, where $\alpha=0.54$ and $\beta=0.46$

#' ## Plot the frequency and impulse response

#' The next code chunk is executed in term mode, see the [script](FIR_design.jl) for syntax.

#' And again with default options

#' Load package
using DrWatson, Plots, DifferentialEquations
@time @quickactivate :LangevinDynamics


saved_values = SavedValues(Float32, Any)

cb = SavingCallback(
    (u, t, integrator) -> mean(u),
    saved_values;
    saveat=0.0:5.0:8000.0,
)


@time sol_1D_SDE = solve(
    langevin_1d_SDE_prob(;
        η=50.0,
        σ0=0.2,
        h=0.0,
        g0=0.01f0,
        tspan=(0.0f0, 8000.0f0),
        u0fun=x ->
            CUDA.fill(1.0f0, LangevinDynamics.N, LangevinDynamics.M),
        # u0fun=x ->
        #     CUDA.randn(Float32, LangevinDynamics.N, LangevinDynamics.M),
    ),
    [SOSRA(),ImplicitEM(),SImplicitMidpoint(),ImplicitRKMil(),SKSROCK()][5],
    dt=0.05f0,
    EnsembleSerial();
    trajectories=1,
    # dt=0.1,
    # saveat=0.0:0.2:10.0,
    save_everystep=false,
    save_start=true,
    save_end=true,
    dense=false,
    callback=cb,
    abstol=1e-1,
    reltol=1e-1,
)
plot(sol_1D_SDE[:,1,2,1])
plot(saved_values.saveval)
