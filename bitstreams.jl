using Pkg # hideall
Pkg.activate("./Project.toml")
Pkg.instantiate()

using BitSAD

x = SBitstream(0.3)

float(x)

xt = pop!(x)

push!(x, xt)
x

generate!(x) # add a single sample
@show length(x)
generate!(x, 1000)
x

abs(estimate(x) - float(x))

y = SBitstream(0.5)
z = x * y

float(z) == float(x) * float(y)

multiply_sbit(x, y) = SBit((pos(x) * pos(y), neg(x) * neg(y)))

num_samples = 1000
for t in 1:num_samples
    xbit, ybit = pop!(x), pop!(y)
    zbit = multiply_sbit(xbit, ybit)
    push!(z, zbit)
end

abs(estimate(z) - float(z))

Pkg.activate(".") # hideall

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

