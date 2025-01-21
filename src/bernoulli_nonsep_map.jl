

function thetaz_nonsep_nlp(theta::Vector, data::InputData, timeknots::AbstractArray, nunq::Integer, map2unq::AbstractVector, priors::NamedTuple, nb::Matrix{Int64}, tol::Float64, maxiter::Int64)

    n = length(data.y)
    p = size(data.X, 2)
    nknots = size(timeknots, 1)
    kt = nunq*nknots

    sw1,rangeS1,rangeT1, sw2, rangeT2 = exp.(theta)

    B1,F1,B1order = getNNGPmatsST(nb, data.loc, data.time, rangeS1, rangeT1)

    B2Rows, B2Cols, B2Order = getBtNZ(n, map2unq, nknots)
    B2Compact = expCor(data.time, timeknots, rangeT2)
    B2 = sparse(B2Rows, B2Cols, view(vec(B2Compact'), B2Order))
    Q2 =expCor(timeknots, rangeT2)

    Q = blockdiag(spdiagm(priors.beta[:,2]), 
                    kron(speye(nunq), (sw2^2)*Q2), 
                    (1 / sw1^2) * (B1'*spdiagm(1 ./ F1)*B1)
                    )

    Dsgn = sparse_hcat(data.X, (sw2^2)*B2, speye(n))

    neffects = size(Dsgn, 2)

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

    llw = -0.5*(effects'*Q*effects + sum(log.(F1)) + 2*n*log(sw1) - 2*kt*log(sw2) - nunq*logdet(Q2) - sum(log.(priors.beta[:,2])) + neffects*log(2*pi))

    # p(w | y, θ)

    #llwc = -0.5*(effects'*Qp*effects - logdet(Qpc) + neffects*log(2*pi))
    llwc = -0.5*( -logdet(Qpc) + neffects*log(2*pi) )

    # p(θ)

    lprior1 = pcpriorST([sw1, rangeS1, rangeT1], priors.theta10, priors.alpha10)
    lprior2 = pcprior([sw2, rangeT2], priors.theta20, priors.alpha20)


    # p(θ | y)

    lpost = lly + llw + lprior1 + lprior2 - llwc

    return -lpost

    

end


function bernoulli_nonsep_map(initparams::NamedTuple, data::InputData, timeknots::AbstractArray, m::Integer, priors::NamedTuple; sub = 1.0, nr_tol = 1e-4, nr_maxiter = 30, f_tol = 1e-3, g_tol = 1e-3, alpha = 1e-6, show_trace = true, store_trace = false)

    local nb = getNeighbors(data.loc, m)

    local locunq = unique(data.loc, dims = 1)
    nunq = size(locunq, 1)
    local  map2unq = indexin(loc2str(data.loc), loc2str(locunq))

    local theta = log.(collect(initparams))

    
    thetaMin = optimize(t -> thetaz_nonsep_nlp(t, data, timeknots, nunq, map2unq, priors, nb, nr_tol, nr_maxiter), 
        theta, BFGS(alphaguess = Optim.LineSearches.InitialStatic(alpha=alpha)), 
        Optim.Options(f_tol = f_tol, g_tol = g_tol, store_trace = store_trace, show_trace = show_trace, extended_trace = (show_trace || store_trace)))

    params = NamedTuple(zip( [:sw1, :rangeS1, :rangeT1, :sw2, :rangeT2], exp.(Optim.minimizer(thetaMin))))


    return params, thetaMin
    
end



#= function thetaz_postmode(theta::Vector, data::InputData, priors::NamedTuple, nb::Matrix{Int64}, tol::Float64, maxiter::Int64)

    n = length(data.y)
    p = size(data.X, 2)

    sw,rangeS,rangeT = exp.(theta)

    B,F,Border = getNNGPmatsST(nb, data.loc, data.time, rangeS, rangeT)

    Q = blockdiag(spdiagm(priors.beta[:,2]), (1 / sw^2) * (B'*spdiagm(1 ./ F)*B))

    Dsgn = sparse_hcat(data.X, speye(n))

    neffects = size(Dsgn, 2)

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


    return effects, softmax.(Dsgn*effects)

    

end

function thetaz_nlp_mcmc(theta::Vector, data::InputData, priors::NamedTuple, nb::Matrix{Int64}, nburn::Int64, nsamps::Int64; returnsd = false)

    n = length(data.y)
    p = size(data.X, 2)

    sw,rangeS,rangeT = exp.(theta)

    B,F,Border = getNNGPmatsST(nb, data.loc, data.time, rangeS, rangeT)

    Q = blockdiag(spdiagm(priors.beta[:,2]), (1 / sw^2) * (B'*spdiagm(1 ./ F)*B))

    Dsgn = sparse_hcat(data.X, speye(n))

    neffects = size(Dsgn, 2)

    ####################
    # mcmc to integrate out the effects
    ######################

    effects = zeros(neffects)
    omega = fill(0.3, n)
    Qp = Q + Dsgn'*spdiagm(omega)*Dsgn
    Qpc = cholesky(Hermitian(Qp))

    zProj = Dsgn'*(data.y .- 0.5)
    zProj[1:p] += priors.beta[:,1] .* priors.beta[:,2]
    pos = data.y .== 1
    mu = Dsgn*effects
    probs = softmax.(mu)


    lpost = zeros(nburn + nsamps)

    for i = 1:(nburn + nsamps)

        Qp .= Q + Dsgn'*spdiagm(omega)*Dsgn
        effects .= getGaussSamp!(Qpc, Qp, zProj)

        mu .= Dsgn*effects
        omega .= rpg.(mu)

        probs = softmax.(mu)
        effects
        lpost[i] = sum(log.(probs[pos])) + sum(log.(1 .- probs[.!pos])) - 0.5*(effects'*Q*effects)
        
    end

    lpostmu = mean(lpost[(nburn+1):(nsamps+nburn)])

    if returnsd
        lpostsd = sqrt(var(lpost[(nburn+1):(nsamps+nburn)])) / sqrt(nsamps)
    end

    lpostmu += -0.5*(sum(log.(F)) + 2*n*log(sw) - sum(log.(priors.beta[:,2])) + neffects*log(2*pi)) + pcpriorST([sw, rangeS, rangeT], priors.theta0, priors.alpha0)


    if returnsd
        return -lpostmu, lpostsd
    else
        return -lpostmu
    end


    

end =#