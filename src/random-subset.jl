function random_subset(trainset, subset_size)
    subset = ObsView(trainset, shuffle(1:numobs(trainset))[1:subset_size])
end