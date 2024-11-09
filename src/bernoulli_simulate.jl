function simulate_Bernoulli(loc::Matrix, time::Matrix, params::NamedTuple, m)

    local n = size(loc, 1) 
    local nb = getNeighbors(loc, m)

    local B, F, Border = getNNGPmatsST(nb, loc, time, params.rangeS, params.rangeT)

    local gzmu = params.beta .+ params.sw * ( LowerTriangular(B) \ (sqrt.(F) .* randn(n)) )
    
    local zmu = softmax.(gzmu)

    local z = 1 .* (rand(n) .< zmu)

    return z
end