include("src/setup.jl");

# modelpath = joinpath(artifact"mobilenet", "mobilenet.bson")
# m = BSON.load(modelpath)[:m]

m = MobileNet(relu, 0.25; fcsize = 64, nclasses = 2)
mults, adds, output_size = compute_dot_prods(m, (96, 96, 3, 1)) # height and weight are 96, input channels are 3, batch size = 1
println("MobileNet Mults ", mults, " Adds ", adds)
