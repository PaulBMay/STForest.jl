function simulate_Bernoulli2(loc::Matrix, time::Matrix, params::NamedTuple, m)

    gtg = haskey(params, :sw1) & haskey(params, :rangeS1) & haskey(params, :rangeT1) & haskey(params, :sw2) & haskey(params, :rangeS2) & haskey(params, :beta)
    if !gtg
        error("bad 'params': The expected fields are 'sw1, rangeS1, rangeT1, sw2, rangeS2, beta'.") 
    end

    local n = size(loc, 1) 

    locUnq = unique(loc, dims = 1)
    nUnq = size(locUnq, 1)
    map2unq = indexin(loc2str(loc), loc2str(locUnq))
    P = sparse(1:n, map2unq, 1)


    local nb1 = getNeighbors(loc, m)
    local nb2 = getNeighbors(locUnq, m)

    local B1, F1, B1order = getNNGPmatsST(nb1, loc, time, params.rangeS1, params.rangeT1)

    local B2, F2, B2order = getNNGPmatsS(nb2, locUnq, params.rangeS2)

    local gzmu = params.beta .+ ( params.sw1 * ( LowerTriangular(B1) \ (sqrt.(F1) .* randn(n)) ) ) +  P*( params.sw2 * ( LowerTriangular(B2) \ (sqrt.(F2) .* randn(nUnq)) ) )
    
    local zmu = softmax.(gzmu)

    local z = 1 .* (rand(n) .< zmu)

    return z
end