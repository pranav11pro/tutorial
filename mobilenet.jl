using Pkg # hideall
Pkg.activate("./Project.toml")
Pkg.instantiate()

include("./src/setup.jl");

artifacts = "./Artifacts.toml"
ensure_artifact_installed("mobilenet", artifacts)
mobilenet = artifact_hash("mobilenet", artifacts)
modelpath = joinpath(artifact_path(mobilenet), "mobilenet.bson")
model = BSON.load(modelpath, @__MODULE__)[:m];

ensure_artifact_installed("vww", artifacts)
vwwdata = artifact_hash("vww", artifacts)
dataroot = joinpath(artifact_path(vwwdata), "vww-hackathon")
valdata = VisualWakeWords(dataroot; subset = :val)
valaug = map_augmentation(ImageToTensor(), valdata)
valloader = DataLoader(BatchView(valaug; batchsize = 32), nothing; buffered = true)

accfn(ŷ::AbstractArray, y::AbstractArray) = mean((ŷ .> 0) .== y)
accfn(data, model) = mean(accfn(model(x), y) for (x, y) in data)

accfn(valloader, model)

model_scaled, scalings = prepare_bitstream_model(model)
@show total_scaling = prod(prod.(scalings))
model_scaled

simulation_length = 1000
add_conversion_error!(model_scaled, simulation_length);

model_rescaled = Chain(model_scaled, x -> x .* total_scaling)
accfn(valloader, model_rescaled)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

