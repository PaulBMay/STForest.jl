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

function getPropVars(path::String, add::Float64, nUse::Integer)

    p = CSV.read(path, DataFrame)
    nSamps = size(p,1)
    
    use = (nSamps - nUse + 1):nSamps

    thetay = log.(hcat(p.swy1[use], p.rangeSy1[use], p.rangeTy1[use], p.swy2[use], p.rangeTy2[use], p.t2y[use]))
    thetayVar = (2.4^2/6)*cov(thetay) + add*I(6)

    thetaz = log.(hcat(p.swz[use], p.rangeSz[use]))
    thetazVar = (2.4^2/3)*cov(thetaz) + add*I(2)

    return thetayVar, thetazVar

end

function getLastSamp(path::String)

    p = CSV.read(path, DataFrame)

    last = size(p,1)

    lastSamp = (swy1 = p.swy1[last], rangeSy1 = p.rangeSy1[last], rangeTy1 = p.rangeTy1[last], swy2 = p.swy2[last], rangeTy2 = p.rangeTy2[last], t2y = p.t2y[last], swz = p.swz[last], rangeSz = p.rangeSz[last])

    return lastSamp

end