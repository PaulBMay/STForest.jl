# Approximation of MAPs for bernoulli_mcmc.jl using Laplace approximations


function newtonupdate!(x::Vector, Qp::SparseMatrixCSV, grad::Vector)
    x += Qp \ grad
end

function thetaz_nlp(theta::Vector, data::InputData, priors::NamedTuple, nb::Matrix{Int64}, nrtol::Float64, maxiter::Int64)

    n = length(data.y)

    sw,rangeS,rangeT = exp.(theta)

    B,F,Border = getNNGPmatsST(nb, data.loc, data.time, rangeS, rangeT)

    Q = blockdiag(spdiagm(priors.beta[:,2]), B'*spdiagm(1 ./ F)*B)

    Dsgn = sparse_hcat(data.X, speye(n))

    neffects = size(Dsgn, 2)

    ####################
    # Newton Raphson to find posterior mode of the effects
    ######################

    effects = zeros(neffects)
    probs = softmax.(Dsgn*effects)
    Omega = spdiagm(probs.*(1 .- probs))
    Qp = Q + Dsgn'*Omega*Dsgn
    Qpc = cholesky(Hermitian(Qp))
    grad = Dsgn'*(data.y - probs) + (priors.beta[:,1] .* priors.beta[:,2])
    effects += (Qpc \ grad)

    error = 2.0
    count = 0

    while (error > nrtol) && (count <= maxiter)

        probs .= softmax.(Dsgn*effects)
        Omega .= spdiagm(probs.*(1 .- probs))
        Qp .= Q + Dsgn'*Omega*Dsgn
        cholesky!(Qpc, Qp)
        grad .= Dsgn'*(data.y - probs) + (priors.beta[:,1] .* priors.beta[:,2])
        update .= effects + (Qp \ grad)
        error = norm(effects - update)
        effects .= copy(update)

    end

    ##################
    # Laplace approximation of the log posterior
    ####################

    # p(y | w, θ)

    probs .= softmax.(Dsgn*effects)
    pos = data.y .== 1
    lly = sum(log.(probs[pos])) + sum(log.(1 .- probs[.!pos]))

    # p(w | θ)

    llw = -0.5*(effects'*Q*effects + sum(log.(F)) - sum(log.(priors.beta[:,2])) + neffects*log(2*pi))

    # p(w | y, θ)

    llwc = -0.5*(effects'*Qp*effects + logdet(Qpc) + neffects*log(2*pi))



end