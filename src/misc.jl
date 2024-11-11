# Spatial plot function. No 'using Plots' in here, so will crash if you haven't loaded Plots.jl in the referencing scope.
function quiltplot(loc, z)
    display(scatter(loc[:,1], loc[:,2], zcolor = z, markerstrokewidth = 0, alpha = 0.8))
    return nothing
end

# Create sparse id matrix
function speye(n::Integer)
    return sparse(1:n,1:n, 1.0)
end

# Paste location rows into a single string. Janky tool to identify unique locations and location matches...
function loc2str(loc::AbstractMatrix{Float64})
    return string.(loc[:,1]).*("_".*string.(loc[:,2]))
end

function softmax(x)
    return 1/(1 + exp(-x))
end

function datasplit(data::InputData)

    pos = data.y .> 0
    z = 1*pos

    dataPos = InputData(data.y[pos], data.X[pos,:], data.loc[pos,:], data.time[pos,:])
    dataZ = InputData(z, data.X, data.loc, data.time)

    return dataZ, dataPos


end

function getPropVars(path::String, vars::Vector{String}, nUse::Integer)

    p = CSV.read(path, DataFrame)
    nSamps = size(p,1)

    use = (nSamps - nUse + 1):nSamps

    pvars = Matrix( p[use, Symbol.(vars)] )
    
    thetaVar = (2.4^2 / size(pvars, 2))*cov( log.(pvars) ) 

    return thetaVar

end

function getLastSamp(path::String, vars::Vector{String})

    p = CSV.read(path, DataFrame)

    last = size(p,1)

    lastSamp = NamedTuple(p[last, Symbol.(vars)])

    return lastSamp

end