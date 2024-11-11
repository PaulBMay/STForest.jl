function getBtNZ(n::Integer, map2unq::Vector, nKnots::Int64)

    local nnz = n*nKnots
    local BtRows = zeros(Int64, nnz)
    local BtCols = zeros(Int64, nnz)

    for i in 1:n

        ind = ((i-1)*nKnots+1):(i*nKnots)
        BtRows[ind] .= i
        BtCols[ind] = ((map2unq[i]-1)*nKnots+1):(map2unq[i]*nKnots)

    end

    local BtOrder = sortperm(BtCols + (BtRows ./ (n+1)))

    return BtRows[BtOrder], BtCols[BtOrder], BtOrder

end