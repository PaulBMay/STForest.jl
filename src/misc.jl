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