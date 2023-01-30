using FFTW
using DSP
using Plots
function unifun(x::T) where {T}
    if x>0
        return x
    else
        return T(0)
    end
end
N = 2^14-1

# Sample period
Ts = 5.0 / (1.1 * N)
# Start time
t0 = 0
tmax = t0 + N * Ts
# time coordinate
t = -tmax:Ts:tmax

# signal
signal = @. Sin(10*t)*exp(-t/10)* # sin (2π f t)

# Fourier Transform of it
F = fft(signal) |> fftshift
freqs = fftfreq(length(t), 1.0/Ts) |> fftshift

# plots
time_domain = plot(t, signal, title = "Signal")
freq_domain = plot(freqs, abs.(F), title = "Spectrum",xlims=(-50,50))
plot(time_domain, freq_domain, layout = 2)
