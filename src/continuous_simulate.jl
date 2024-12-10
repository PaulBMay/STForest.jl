function simulate_Continuous(loc::Matrix, time::Matrix, nKnots::Int, params::NamedTuple, m)

    gtg = haskey(params, :beta) & haskey(params, :sw1) & haskey(params, :rangeS1) & haskey(params, :rangeT1) & haskey(params, :sw2) & haskey(params, :rangeT2) & haskey(params, :tSq)
    if !gtg
        error("bad 'params': The expected fields are 'beta, sw1, rangeS1, rangeT1, sw2, rangeT2, tSq'.") 
    end

    local n = size(loc, 1) 
    local nb = getNeighbors(loc, m)

    local B1, F1, B1order = getNNGPmatsST(nb, loc, time, params.rangeS1, params.rangeT1)

    local w1 = params.sw1 * ( LowerTriangular(B1) \ (sqrt.(F1) .* randn(n)) )

    #######

    local locUnq = unique(loc, dims = 1)
    local nUnq = size(locUnq, 1)

    # For each location in the full stack, what's it's position among the set of unique locations?
    map2unq = indexin(loc2str(loc), loc2str(locUnq))

    if nUnq < n

        #local timeStep = (maximum(time) - minimum(time)) / (nKnots - 1)
        local timeKnots = reshape( collect( range(minimum(time), maximum(time), length = nKnots) ), :, 1 )

        local Q2 = expCor(timeKnots, timeKnots, params.rangeT2)
        local w2 = vec( cholesky(Q2).U \ randn(nKnots, nUnq))
        local B2Rows, B2Cols, B2Order = getBtNZ(n, map2unq, nKnots)
        local B2Compact = expCor(time, timeKnots, params.rangeT2)
        local B2 = sparse(B2Rows, B2Cols, view(vec(B2Compact'), B2Order))

        local eta2 = params.sw2*(B2*w2)

    else
        eta2 = params.sw2*randn(n)
        @warn "No unique locations... Are you sure this is what you want?"
    end

    local y = params.beta .+ w1 + eta2 + sqrt(params.tSq)*randn(n)

    return y
 
end