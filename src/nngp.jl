# Get nearest neighbor indices of a coordinate set. Based soley on spatial Euclidean, which is of dubious merit in some space-time scenarios.
function getNeighbors(loc::Matrix{Float64}, m::Int64)

    local n = size(loc,1)

    local tree = KDTree(loc')

    skip_fun(i) = i >= ind

    local nb = zeros(Int64, n - m - 1, m)

    local ind = m+2

    for i in (m+2):n
        nb[i-m-1,:] = knn(tree, loc[i,:], m, false, skip_fun)[1]
        ind += 1
    end

    return nb

end


##########################
# Get NNGP mats.
# Spatial and Space-time versions, getting the B sparse matrix and F diagonal vector.
# There are in place versions, taking advantage of the consistent sparse structure across all covariance parameters.
# These in place versions require an ordering vector (returned by the not in place) to map the dense n x m covariance to the non-zero entries of the sparse matrix.
##########################

function getNNGPmatsST(nb::Matrix{Int64}, loc::Matrix{Float64}, time::Matrix{Float64}, rho_s::Number, rho_t::Number)

    local n = size(loc,1)
    local m = size(nb, 2)

    local Bnnz = sum(1:(m+1)) + (n - m - 1)*(m+1)
    local Bvals = zeros(Bnnz)
    Bvals[1] = 1

    local Brows = zeros(Int64, Bnnz)
    local Bcols = zeros(Int64, Bnnz)
    Brows[1] = 1 
    Bcols[1] = 1

    local Fvals = zeros(n)
    Fvals[1] = 1.0

    curInd = 1

    @views for i in 2:(m+1)

        indi = 1:(i-1)

        mi = i - 1

        rho = expCor(loc[indi,:], rho_s, time[indi,:], rho_t) 

        k = expCor(loc[indi,:], loc[[i],:], rho_s, time[indi,:], time[[i],:], rho_t)

        cholesky!(rho)

        ldiv!(UpperTriangular(rho)', k)

        Fvals[i] = 1 - dot(k, k)

        ldiv!(UpperTriangular(rho), k)

        Bvals[(curInd+1):(curInd + mi + 1)] = [1; -k]
        Brows[(curInd+1):(curInd + mi + 1)] .= i
        Bcols[(curInd+1):(curInd + mi + 1)] = [i; indi]

        curInd += mi + 1

    end

    local rho = zeros(m, m)
    local T = zeros(m,m)
    local k = zeros(m,1)
    local Tcross = zeros(m,1)

    @views for i in (m+2):n
        
        indi = nb[i - m - 1,:]

        expCor!(rho, T, loc[indi,:], rho_s, time[indi,:], rho_t)

        expCor!(k, Tcross, loc[indi,:], loc[[i],:], rho_s, time[indi,:], time[[i],:], rho_t)

        cholesky!(rho)

        ldiv!(UpperTriangular(rho)', k)

        Fvals[i] = 1 - dot(k, k)

        ldiv!(UpperTriangular(rho), k)

        Bvals[(curInd+1):(curInd + m + 1)] = [1; -k]
        Brows[(curInd+1):(curInd + m + 1)] .= i
        Bcols[(curInd+1):(curInd + m + 1)] = [i; indi]

        curInd += m + 1

    end

    B = sparse(Brows, Bcols, Bvals)

    Border = invperm(sortperm( @.(Bcols + ( Brows ./ (n+1) ))))

    #println(cor(Bvals, B.nzval[Border]))

    return B, Fvals, Border

end

function getNNGPmatsST!(B::SparseMatrixCSC, Fvals::Vector{Float64}, Border::Vector{Int64}, nb::Matrix{Int64}, loc::Matrix{Float64}, time::Matrix{Float64}, rho_s::Number, rho_t::Number)

    dist = Distances.Euclidean()
    n = size(loc,1)
    m = size(nb, 2)

    curInd = 1

    @views for i in 2:(m+1)

        indi = 1:(i-1)

        mi = i - 1

        rho = expCor(loc[indi,:], rho_s, time[indi,:], rho_t) 

        k = expCor(loc[indi,:], loc[[i],:], rho_s, time[indi,:], time[[i],:], rho_t)

        cholesky!(rho)

        ldiv!(UpperTriangular(rho)', k)

        Fvals[i] = 1 - dot(k, k)

        ldiv!(UpperTriangular(rho), k)

        valIndex = Border[(curInd+1):(curInd + mi + 1)]

        B.nzval[valIndex] .= [1; -k]

        curInd += mi + 1

    end

    rho = zeros(m, m)
    T = zeros(m,m)
    k = zeros(m,1)
    Tcross = zeros(m,1)

    @views for i in (m+2):n
        
        indi = nb[i - m - 1,:]

        expCor!(rho, T, loc[indi,:], rho_s, time[indi,:], rho_t)

        expCor!(k, Tcross, loc[indi,:], loc[[i],:], rho_s, time[indi,:], time[[i],:], rho_t)

        cholesky!(rho)

        ldiv!(UpperTriangular(rho)', k)

        Fvals[i] = 1 - dot(k, k)

        ldiv!(UpperTriangular(rho), k)

        valIndex = Border[(curInd+1):(curInd + m + 1)]

        B.nzval[valIndex] .= [1; -k]

        curInd += m + 1

    end

    return nothing

end

function getNNGPmatsS(nb::Matrix{Int64}, loc::Matrix{Float64}, rho_s::Float64)

    dist = Distances.Euclidean()
    n = size(loc,1)
    m = size(nb, 2)

    Bnnz = sum(1:(m+1)) + (n - m - 1)*(m+1)
    Bvals = zeros(Bnnz)
    Bvals[1] = 1

    Brows = zeros(Int32, Bnnz)
    Bcols = zeros(Int32, Bnnz)
    Brows[1] = 1 
    Bcols[1] = 1

    Fvals = zeros(n)
    Fvals[1] = 1

    curInd = 1

    @views for i in 2:(m+1)

        indi = 1:(i-1)

        mi = i - 1

        rho = expCor(loc[indi,:], rho_s) 

        k = expCor(loc[indi,:], loc[[i],:], rho_s)

        cholesky!(rho)

        ldiv!(UpperTriangular(rho)', k)

        Fvals[i] = 1 - dot(k, k)

        ldiv!(UpperTriangular(rho), k)

        Bvals[(curInd+1):(curInd + mi + 1)] = [1; -k]
        Brows[(curInd+1):(curInd + mi + 1)] .= i
        Bcols[(curInd+1):(curInd + mi + 1)] = [i; indi]

        curInd += mi + 1

    end

    rho = zeros(m, m)
    T = zeros(m,m)
    k = zeros(m,1)
    Tcross = zeros(m,1)

    @views for i in (m+2):n
        
        indi = nb[i - m - 1,:]

        expCor!(rho, loc[indi,:], rho_s)

        expCor!(k, loc[indi,:], loc[[i],:], rho_s)

        cholesky!(rho)

        ldiv!(UpperTriangular(rho)', k)

        Fvals[i] = 1 - dot(k, k)

        ldiv!(UpperTriangular(rho), k)

        Bvals[(curInd+1):(curInd + m + 1)] = [1; -k]
        Brows[(curInd+1):(curInd + m + 1)] .= i
        Bcols[(curInd+1):(curInd + m + 1)] = [i; indi]

        curInd += m + 1


    end

    B = sparse(Brows, Bcols, Bvals)

    Border = invperm(sortperm( @.(Bcols + ( Brows ./ (n+1) ))))

    return B, Fvals, Border

end

function getNNGPmatsS!(B::SparseMatrixCSC, Fvals::Vector{Float64}, Border::Vector{Int64}, nb::Matrix{Int64}, loc::Matrix{Float64}, rho_s::Number)

    dist = Distances.Euclidean()
    n = size(loc,1)
    m = size(nb, 2)

    curInd = 1

    @views for i in 2:(m+1)

        indi = 1:(i-1)

        mi = i - 1

        rho = expCor(loc[indi,:], rho_s) 

        k = expCor(loc[indi,:], loc[[i],:], rho_s)

        cholesky!(rho)

        ldiv!(UpperTriangular(rho)', k)

        Fvals[i] = 1 - dot(k, k)

        ldiv!(UpperTriangular(rho), k)

        valIndex = Border[(curInd+1):(curInd + mi + 1)]

        B.nzval[valIndex] .= [1; -k]

        curInd += mi + 1

    end

    rho = zeros(m, m)
    T = zeros(m,m)
    k = zeros(m,1)
    Tcross = zeros(m,1)

    @views for i in (m+2):n
        
        indi = nb[i - m - 1,:]

        expCor!(rho, loc[indi,:], rho_s)

        expCor!(k, loc[indi,:], loc[[i],:], rho_s)

        cholesky!(rho)

        ldiv!(UpperTriangular(rho)', k)

        Fvals[i] = 1 - dot(k, k)

        ldiv!(UpperTriangular(rho), k)
        
        valIndex = Border[(curInd+1):(curInd + m + 1)]

        B.nzval[valIndex] .= [1; -k]

        curInd += m + 1


    end

    return nothing

end


#######
# Similar, but for prediction
########

function getNeighborsP(loc, locPred, m)

    np = size(locPred)[1]

    tree = KDTree(loc')

    nbp = hcat(
        knn(tree, locPred', m, false)[1]...
    )'

    return Matrix(nbp)

end

function getNNGPmatsSTP(nb::Matrix{Int64}, loc::Matrix{Float64}, time::Matrix{Float64}, locp::Matrix{Float64}, timep::Matrix{Float64}, rho_s::Number, rho_t::Number)

    local n = size(loc,1)
    local np = size(locp, 1)
    local m = size(nb, 2)

    local Bnnz = np*m
    local Bvals = zeros(Bnnz)

    local Brows = repeat(1:np, inner = m)
    local Bcols = vec(nb')

    local Fvals = zeros(np)

    local rho = zeros(m, m)
    local T = zeros(m,m)
    local k = zeros(m,1)
    local Tcross = zeros(m,1)

    @views for i in 1:np
        
        indi = nb[i,:]

        expCor!(rho, T, loc[indi,:], rho_s, time[indi,:], rho_t)

        expCor!(k, Tcross, loc[indi,:], locp[[i],:], rho_s, time[indi,:], timep[[i],:], rho_t)

        cholesky!(rho)

        ldiv!(UpperTriangular(rho)', k)

        Fvals[i] = 1 - dot(k, k)

        ldiv!(UpperTriangular(rho), k)

        Bvals[((i-1)*m + 1):(i*m)] .= k

    end

    B = sparse(Brows, Bcols, Bvals, np, n)

    Border = invperm(sortperm( @.(Bcols + ( Brows ./ (np+1) ))))

    return B, Fvals, Border


end

function getNNGPmatsSTP!(B::SparseMatrixCSC, Fvals::Vector{Float64}, Border::Vector{Int64}, nb::Matrix{Int64}, loc::Matrix{Float64}, time::Matrix{Float64}, locp::Matrix{Float64}, timep::Matrix{Float64}, rho_s::Number, rho_t::Number)

    local n = size(loc,1)
    local np = size(locp, 1)
    local m = size(nb, 2)

    local rho = zeros(m, m)
    local T = zeros(m,m)
    local k = zeros(m,1)
    local Tcross = zeros(m,1)

    @views for i in 1:np
        
        indi = nb[i,:]

        expCor!(rho, T, loc[indi,:], rho_s, time[indi,:], rho_t)

        expCor!(k, Tcross, loc[indi,:], locp[[i],:], rho_s, time[indi,:], timep[[i],:], rho_t)

        cholesky!(rho)

        ldiv!(UpperTriangular(rho)', k)

        Fvals[i] = 1 - dot(k, k)

        ldiv!(UpperTriangular(rho), k)

        B.nzval[Border[((i-1)*m + 1):(i*m)]] .= k

    end

    return nothing

end

function getNNGPmatsSP(nb::Matrix{Int64}, loc::Matrix{Float64}, locp::Matrix{Float64}, rho_s::Number)

    local n = size(loc,1)
    local np = size(locp, 1)
    local m = size(nb, 2)

    local Bnnz = np*m
    local Bvals = zeros(Bnnz)

    local Brows = repeat(1:np, inner = m)
    local Bcols = vec(nb')

    local Fvals = zeros(np)

    local rho = zeros(m, m)
    local k = zeros(m,1)

    @views for i in 1:np
        
        indi = nb[i,:]

        expCor!(rho, loc[indi,:], rho_s)

        expCor!(k, loc[indi,:], locp[[i],:], rho_s)

        cholesky!(rho)

        ldiv!(UpperTriangular(rho)', k)

        Fvals[i] = 1 - dot(k, k)

        ldiv!(UpperTriangular(rho), k)

        Bvals[((i-1)*m + 1):(i*m)] .= k

    end

    B = sparse(Brows, Bcols, Bvals, np, n)

    Border = invperm(sortperm( @.(Bcols + ( Brows ./ (np+1) ))))

    return B, Fvals, Border


end


function getNNGPmatsSP!(B::SparseMatrixCSC, Fvals::Vector{Float64}, Border::Vector{Int64}, nb::Matrix{Int64}, loc::Matrix{Float64}, locp::Matrix{Float64}, rho_s::Number)

    local n = size(loc,1)
    local np = size(locp, 1)
    local m = size(nb, 2)

    local rho = zeros(m, m)
    local k = zeros(m,1)

    @views for i in 1:np
        
        indi = nb[i,:]

        expCor!(rho, loc[indi,:], rho_s)

        expCor!(k, loc[indi,:], locp[[i],:], rho_s)

        cholesky!(rho)

        ldiv!(UpperTriangular(rho)', k)

        Fvals[i] = 1 - dot(k, k)

        ldiv!(UpperTriangular(rho), k)

        B.nzval[Border[((i-1)*m + 1):(i*m)]] .= k

    end

    return nothing

end
