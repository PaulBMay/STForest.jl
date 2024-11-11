##############
# Exponential covariance/cross-covariance functions.
# Utilizing multiple dispatch here. There are internal covariances, cross covariance, and in-place memory versions of both of those.
# Also versions of all of the above for separable exponential space-time. So that makes 8 total... Probably sweatier than required.
##############

# Single location set
function expCor(loc::AbstractMatrix{Float64}, rho::Number)
    return exp.( -pairwise(Euclidean(), loc, dims = 1) ./ rho )
end

# Single location set, in place
function expCor!(S::AbstractMatrix{Float64}, loc::AbstractMatrix{Float64}, rho::Number)
    pairwise!(S, Euclidean(), loc, dims = 1)
    @. S .= exp(-S / rho) 
    return nothing
end

# Cross locations
function expCor(loc1::AbstractMatrix{Float64}, loc2::AbstractMatrix{Float64}, rho::Number)
    return exp.( -pairwise(Euclidean(), loc1, loc2, dims = 1) ./ rho )
end

#Cross locations, in place
function expCor!(Scross::AbstractMatrix{Float64}, loc1::AbstractMatrix{Float64}, loc2::AbstractMatrix{Float64}, rho::Number)
    pairwise!(Scross, Euclidean(), loc1, loc2, dims = 1)
    @. Scross .= exp(-Scross / rho) 
    return nothing
end

#ST Single location set
function expCor(loc::AbstractMatrix{Float64}, rho_s::Number, time::AbstractMatrix{Float64}, rho_t::Number)
    return exp.( (-pairwise(Euclidean(), loc, dims = 1) ./ rho_s) +  (-pairwise(Euclidean(), time, dims = 1) ./ rho_t))
end

#ST Single location set, in place
function expCor!(S::AbstractMatrix{Float64}, T::AbstractMatrix{Float64}, loc::AbstractMatrix{Float64}, rho_s::Number, time::AbstractMatrix{Float64}, rho_t::Number)
    pairwise!(S, Euclidean(), loc, dims = 1)
    pairwise!(T, Euclidean(), time, dims = 1) 
    @. S .= exp(-S/rho_s - T/rho_t)
    return nothing
end

# ST Cross locations
function expCor(loc1::AbstractMatrix{Float64}, loc2::AbstractMatrix{Float64}, rho_s::Number, time1::AbstractMatrix{Float64}, time2::AbstractMatrix{Float64}, rho_t::Number)
    return exp.( (-pairwise(Euclidean(), loc1, loc2, dims = 1) ./ rho_s) +  (-pairwise(Euclidean(), time1, time2, dims = 1) ./ rho_t))
end

# ST Cross locations, in place
function expCor!(Scross::AbstractMatrix{Float64}, Tcross::AbstractMatrix{Float64}, loc1::AbstractMatrix{Float64}, loc2::AbstractMatrix{Float64}, rho_s::Number, time1::AbstractMatrix{Float64}, time2::AbstractMatrix{Float64}, rho_t::Number)
    pairwise!(Scross, Euclidean(), loc1, loc2, dims = 1)
    pairwise!(Tcross, Euclidean(), time1, time2, dims = 1) 
    @. Scross .= exp(-Scross/rho_s - Tcross/rho_t)
    return nothing
end




