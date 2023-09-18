using FFTW
using DSP
using Plots
function unifun(x::T) where {T}
    if x > 0
        return x
    else
        return T(0)
    end
end
N = 2^10 - 1

# Sample period
Ts = 0.02
# Start time
t0 = 0.0f0
tmax = t0 + 10000 * Ts
# time coordinate
t = t0:Ts:tmax

rhot = [
    mean(tempv[2][t][2] .* tempv[2][1][1] - tempv[2][1][2] .* tempv[2][t][1]) for
    t in 1:10001
]

rhot = [mean(tempv3[1][t][1][1]) for t in 1:10001]

rhot =
    -128^2 * [
        mean(
            (tempv3[1][t][2] .* tempv3[1][1][1] - tempv3[1][1][2] .* tempv3[1][t][1])[1:end]
        ) for t in 1:10001
    ] ./ 0.1f0

[
    mean(
        (tempv3[1][t + dt][2] .* tempv3[1][dt][1] - tempv3[1][dt][2] .* tempv3[1][t + dt][1])[1:end],
    ) for t in 1:10001
]
# signal
tempv3[1]
Array(tempv3[1][1])
pia = stack(tempv_T05[1])[:, 2, :]
phia = stack(tempv_T05[1])[:, 1, :]

pia[:, t + tp] .* phia[:, tp] - pia[:, tp] .* phia[:, t + tp]

rhot_T05 =
    mean.([
        vec(
            mean(
                pia[:, (t + 1):(t + 10001)] .* phia[:, 1:10001] -
                pia[:, 1:10001] .* phia[:, (t + 1):(t + 10001)];
                dims=2,
            ),
        ) for t in 0:1:10000
    ])

let t = 1
    vec(
        mean(
            pia[:, (t + 1):(t + 10001)] .* phia[:, 1:10001] -
            pia[:, 1:10001] .* phia[:, (t + 1):(t + 10001)];
            dims=2,
        ),
    )
end

mean([
    pia[:, 1 + tp] .* phia[:, tp] - pia[:, tp] .* phia[:, 1 + tp] for tp in 1:(10001 - 1)
])

plot(rhot[1:8000])

signal = -128^2 *rhot_T100[1:10001]./10.0 # sin (2π f t)

# Fourier Transform of it
F = fftshift(fft(100.0 * signal))
freqs = fftshift(fftfreq(length(t), 1.0 / Ts))

J, N = 8, 16
k = range(-0.4; stop=0.4, length=J)  # nodes at which the NFFT is evaluated
f = randn(ComplexF64, J)             # data to be transformed
p = plan_nfft(k, N; reltol=1e-9)     # create plan
fHat = adjoint(p) * f                # calculate adjoint NFFT
y = p * fHat                         # calculate forward NFFT

plot!(
    0.01:0.01:4,
    abs.(
        nufft1d3(
            collect(t)[1:10001], signal[1:10001] .+ 0.0im, 1, 1e-11, collect(0.01:0.01:4)
        )[:, 1]
    );
    yaxis=:log,
    xlabel="ω",ylabel="ρ(ω)",label="T=10.0"
)

# plots
time_domain = plot(t,signal[1:10001];title="T=2",xlabel="t",ylabel="ρ(t)")
freq_domain = plot(freqs, abs.(F); title="Spectrum", xlims=(0.01, 1.0), yaxis=:log)
plot(time_domain, freq_domain; layout=2)
phia = stack(tempv_T50[1])[:, 1, :]
plot(mean(phia; dims=1)[1, :])
