function simulate_Bernoulli(loc::Matrix, time::Matrix, params::NamedTuple, m)

    gtg = haskey(params, :sw) & haskey(params, :rangeS) & haskey(params, :rangeT)
    if !gtg
        error("bad 'params': The expected fields are 'sw, rangeS, rangeT'.") 
    end

    local n = size(loc, 1) 
    local nb = getNeighbors(loc, m)

    local B, F, Border = getNNGPmatsST(nb, loc, time, params.rangeS, params.rangeT)

    local gzmu = params.beta .+ params.sw * ( LowerTriangular(B) \ (sqrt.(F) .* randn(n)) )
    
    local zmu = softmax.(gzmu)

    local z = 1 .* (rand(n) .< zmu)

    return z
end