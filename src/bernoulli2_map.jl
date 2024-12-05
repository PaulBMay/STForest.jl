

function thetaz_nlp2(theta::Vector, data::InputData, locUnq::Matrix, Dsgn::SparseMatrixCSC, nb1::Matrix{Int64}, nb2::Matrix{Int64}, priors::NamedTuple, tol::Float64, maxiter::Int64)

    n = length(data.y)
    p = size(data.X, 2)
    neffects = size(Dsgn, 2)
    nUnq = neffects - (n + p)


    sw1,rangeS1,rangeT1,sw2,rangeS2 = exp.(theta)

    B1,F1,B1order = getNNGPmatsST(nb1, data.loc, data.time, rangeS1, rangeT1)
    B2,F2,B2order = getNNGPmatsS(nb2, locUnq, rangeS2)

    Q = blockdiag(
        spdiagm(priors.beta[:,2]), 
        (1 / sw1^2) * (B1'*spdiagm(1 ./ F1)*B1),
        (1 / sw2^2) * (B2'*spdiagm(1 ./ F2)*B2)
    )



    ####################
    # Newton Raphson to find posterior mode of the effects
    ######################

    effects = zeros(neffects)
    update = copy(effects)
    probs = softmax.(Dsgn*effects)
    Omega = spdiagm(probs.*(1 .- probs))
    Qp = Q + Dsgn'*Omega*Dsgn
    Qpc = cholesky(Hermitian(Qp))
    grad = Dsgn'*(data.y - probs) - Q*effects
    grad[1:p] += priors.beta[:,1] .* priors.beta[:,2]
    effects += (Qpc \ grad)

    error = 2.0
    count = 0

    while (error > tol) && (count <= maxiter)

        probs .= softmax.(Dsgn*effects)
        Omega .= spdiagm(probs.*(1 .- probs))
        Qp .= Q + Dsgn'*Omega*Dsgn
        cholesky!(Qpc, Hermitian(Qp))
        grad .= Dsgn'*(data.y - probs) - Q*effects
        grad[1:p] += priors.beta[:,1] .* priors.beta[:,2]
        update .= effects + (Qpc \ grad)
        error = norm(effects - update) / norm(update)
        effects .= copy(update)
        #println(error)

        count += 1

    end

    ##################
    # Laplace approximation of the log posterior
    ####################

    # p(y | w, θ)

    probs .= softmax.(Dsgn*effects)
    pos = data.y .== 1
    lly = sum(log.(probs[pos])) + sum(log.(1 .- probs[.!pos]))

    # p(w | θ)

    effects[1:p] -= priors.beta[:,1]

    llw = -0.5*(effects'*Q*effects + sum(log.(F1)) + sum(log.(F2)) + 2*n*log(sw1) + 2*nUnq*log(sw2) - sum(log.(priors.beta[:,2])) + neffects*log(2*pi))

    # p(w | y, θ)

    #llwc = -0.5*(effects'*Qp*effects - logdet(Qpc) + neffects*log(2*pi))
    llwc = -0.5*( -logdet(Qpc) + neffects*log(2*pi) )

    # p(θ)

    lprior = pcpriorST([sw1, rangeS1, rangeT1], priors.theta10, priors.alpha10) + pcprior([sw2, rangeS2], priors.theta20, priors.alpha20)

    # p(θ | y)

    lpost = lly + llw + lprior - llwc

    return -lpost

    

end


function bernoullimap2(theta::Vector, data::InputData, m::Integer, priors::NamedTuple; sub = 1.0, nr_tol = 1e-4, nr_maxiter = 30, f_tol = 1e-3, g_tol = 1e-3, alpha = 1e-6, show_trace = true, store_trace = false)

    n = length(data.y)
    local nb1 = getNeighbors(data.loc, m)

    local locUnq = unique(data.loc, dims = 1)
    local nb2 = getNeighbors(locUnq, m)

    local map2unq = indexin(loc2str(data.loc), loc2str(locUnq))
    local P = sparse(1:n, map2unq, 1)

    Dsgn = sparse_hcat(data.X, speye(n), P)
    
    thetaMin = optimize(t -> thetaz_nlp2(t, data, locUnq, Dsgn, nb1, nb2, priors, nr_tol, nr_maxiter), 
        theta, BFGS(alphaguess = Optim.LineSearches.InitialStatic(alpha=alpha)), 
        Optim.Options(f_tol = f_tol, g_tol = g_tol, store_trace = store_trace, show_trace = show_trace, extended_trace = (show_trace || store_trace)))

    params = NamedTuple(zip( [:sw1, :rangeS1, :rangeT1, :sw2, :rangeS2], exp.(Optim.minimizer(thetaMin))))


    return params, thetaMin
    
end



