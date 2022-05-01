lfsr_power_model(n) = 10.89 * n
lfsr_area_model(n) = 354.31 * n
dp_power_model(n) = 0.2624 * n + 1.5944
dp_area_model(n) = 14.717 * n + 56.324

##
function dp_cost(m::Dense, input_size, cost_fn)
    output_size = Flux.outputsize(m, input_size) # wxhxcinxcout
    # compute the number of nonzero elements in weight array
    total_cost = 0
    for w in eachrow(m.weight)
        mul_dp = size(m.weight)[2] - count(iszero, w)
        channel_cost = cost_fn(mul_dp)
        total_cost += channel_cost
    end
    return total_cost, output_size
end
##
function dp_cost(m::Conv, input_size, cost_fn)
    # compute the number of nonzero elements in weight array
    total_cost = 0.0
    output_size = Flux.outputsize(m, input_size) # WxHxCinxCout
    num_output_patches = output_size[1] * output_size[2] # height of im2col matrix
    # how to iterate over
    for w in eachslice(m.weight, dims=4) # iterate over output channels
        # println(size(w))
        muls = (prod(size(w)) - count(iszero, w))
        channel_cost = cost_fn(muls) * num_output_patches
        total_cost += channel_cost
    end
    return total_cost, output_size
end
##
function dp_cost(model::Chain, input_size, cost_fn)
    total_cost = 0
    intermediate_size = input_size

    for i in 1:length(model)
        layer_cost, output_size = dp_cost(model[i], intermediate_size, cost_fn)
        total_cost += layer_cost
        # println("cost in layer ", i, ": ", layer_cost)
        # println("output_size: ", output_size)  
        intermediate_size = output_size
    end
    return total_cost, intermediate_size
end

# default case
function dp_cost(m, input_size, cost_fn)
    output_size = Flux.outputsize(m, input_size) # wxhxcinxcout
    return 0, output_size
end

lfsr_cost(m, cost_fn) = 0
lfsr_cost(m::Union{Dense, Conv}, cost_fn) =
    cost_fn(count(iszero, m.weight) + count(iszero, m.bias))
lfsr_cost(m::Chain, cost_fn) = sum(layer -> lfsr_cost(layer, cost_fn), m)

power_consumption(model, inputsize) =
    dp_cost(model, inputsize, dp_power_model)[1] + lfsr_cost(model, lfsr_power_model)

area_consumption(model, inputsize) =
    dp_cost(model, inputsize, dp_area_model)[1] + lfsr_cost(model, lfsr_area_model)
