function evaluate_submission(model, data, simulation_length)
    @info "Calculating HW cost..."
    power = power_consumption(model, (96, 96, 3, 1))
    area = area_consumption(model, (96, 96, 3, 1))

    @info "Evaluating simulated model performance..."
    model_scaled, scalings = prepare_bitstream_model(model)
    total_scaling = prod(prod.(scalings))
    add_conversion_error!(model_scaled, simulation_length)
    model_rescaled = Chain(model_scaled, x -> x .* total_scaling)
    acc = accfn(data, model_rescaled)

    @info """
    Evaluation complete!

    Area consumption = $area mm²
    Energy consumption = $(power * simulation_length) uW * cycles
    Accuracy = $(round(100 * acc; digits = 2))% correct

    Please submit these results on the website.
    """

    return area, power, acc
end
