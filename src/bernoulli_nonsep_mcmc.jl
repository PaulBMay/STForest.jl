# adaptive proposal variance for theta
function bernoulli_nonsep_mcmc(data::InputData, m::Int64, timeKnots::AbstractArray, initparams::NamedTuple, priors::NamedTuple, outDir::String, nSamps::Int64; adaptStart = 50, pgwarmup = 10)

    ###################
    # Check to see if the Tuples have the required fields
    ####################

    # params
    gtg = haskey(initparams, :sw1) & haskey(initparams, :rangeS1) & haskey(initparams, :rangeT1) & haskey(initparams, :sw2) & haskey(initparams, :rangeT2)
    if !gtg
        error("bad 'initparams': The expected fields are 'sw1, rangeS1, rangeT1, sw2, rangeT2'.") 
    end
    # priors
    gtg = haskey(priors, :theta10) & haskey(priors, :alpha10) & haskey(priors, :theta20) & haskey(priors, :alpha20) & haskey(priors, :beta)
    if !gtg
        error("bad 'priors': The expected fields are 'theta10, alpha10, theta20, alpha20, beta'.") 
    end



    ############################
    # Data dimensions and prep
    ##########################

    n = size(data.y, 1) # sample size
    p = size(data.X, 2) # num predictors
    nKnots = size(timeKnots, 1)

    pos = data.y .> 0

    # Unique plot locations
    locUnq = unique(data.loc, dims = 1)
    nUnq = size(locUnq, 1)

    # For each location in the full stack, what's it's position among the set of unique locations?
    map2unq = indexin(loc2str(data.loc), loc2str(locUnq))

    kt = nUnq*nKnots

    size(priors.beta,1) == p || error("nrows of priors.beta does not match ncols of X")

    betaMu = priors.beta[:,1]
    betaPrec = priors.beta[:,2]

    # Does the out_dir exist?

    if !isdir(outDir)
        error("Can't find your outDir")
    end

    # Prepare CSV's

    loctimeOut = joinpath(outDir, "locTime.csv")
    CSV.write(loctimeOut, DataFrame(lon = data.loc[:,1], lat = data.loc[:,2], time = data.time[:,1]))

    paramsOut = joinpath(outDir, "zparams.csv")
    paramsDf = DataFrame(zeros(1, p + 5), 
        ["beta_".*string.(0:(p-1)); ["sw1", "rangeS1", "rangeT1. sw2, rangeT2"]]
        )

    wOut = joinpath(outDir, "wz.csv")
    wDf = DataFrame(zeros(1,n), "w_".*string.(1:n))


    # fixed/random effect values
    effects = zeros(p + kt + n)
    w = view(effects, (p+kt+1):(p+kt+n))
    beta = view(effects, 1:p)
    w2 = view(effects, (p+1):(p+kt))

    sw1, rangeS1, rangeT1, sw2, rangeT2 = initparams.sw1, initparams.rangeS1, initparams.rangeT1, initparams.sw2, initparams.rangeT2
    sw1p, rangeS1p, rangeT1p, sw2p, rangeT2p = sw1, rangeS1, rangeT1, sw2, rangeT2

    paramsDf[1,:] = [beta; [sw1, rangeS1, rangeT1, sw2, rangeT2]]
    wDf[1,:] = w

    CSV.write(paramsOut, paramsDf)
    CSV.write(wOut, wDf)


    ####################
    # Lord have mercy that was boring.
    # Now fun stuff. Get the neighbor sets and initial NNGP mats
    #####################

    print("Getting neighbor sets\n")

    nb = getNeighbors(data.loc, m)

    print("Initial NNGP mats\n")

    B1,F1,B1Order = getNNGPmatsST(nb, data.loc, data.time, rangeS, rangeT)
    B1p,F1p = copy(B1), copy(F1)

    B2Rows, B2Cols, B2Order = getBtNZ(n, map2unq, nKnots)
    B2Compact = expCor(data.time, timeKnots, rangeT2)
    B2 = sparse(B2Rows, B2Cols, view(vec(B2Compact'), B2Order))
    Q2 = expCor(timeKnots, rangeT2)
    B2p, Q2p = copy(B2), copy(Q2)

    Dsgn = sparse_hcat(data.X, (sw2^2)*B2, speye(n))
    Dsgnp = copy(Dsgn)

    zProj = Dsgn'*(data.y .- 0.5)

    zProj[1:p] += betaPrec .* betaMu

    pg = rpg.(fill(0.3, n))




    ##############

    Q = blockdiag(
        spdiagm(betaPrec),
        kron(speye(nUnq), (sw2^2)*Q2),
        (1/sw1^2)*B1'*spdiagm(1 ./ F1)*B1
    ) + Dsgn'*spdiagm(pg)*Dsgn

    Qc = cholesky(Hermitian(Q))

    acceptTheta1 = false
    acceptTheta2 = false

    theta1mat = zeros(nSamps+1, 3)
    theta1mat[1,:] = log.([sw1, rangeS1, rangeT1])

    theta2mat = zeros(nSamps+1,2)
    theta2mat[1,:] = log.([sw2, rangeT2])

    theta1Var = 1e-5*Matrix(I,3,3)
    theta2Var = 1e-4*Matrix(I,2,2)

    #####################
    # pg warmup
    #####################

    println("Warming up Polya-Gamma and random effect values")

    Qprior = blockdiag(
        spdiagm(betaPrec),
        kron(speye(nUnq), (sw2^2)*Q2),
        (1/sw1^2)*B1'*spdiagm(1 ./ F1)*B1
    )

    for i = 1:pgwarmup

       Q .= Qprior + Dsgn'*spdiagm(pg)*Dsgn

       effects .= getGaussSamp!(Qc, Q, zProj)

       #######################
       # Sample pg
       #######################

       pg .= rpg.(Dsgn*effects)

    end

    Qprior = []


    #########################
    # Begin Gibbs sampler
    #########################

    for i = ProgressBar(1:nSamps)

       ############################
       # Sample beta, w
       ############################

       Q = blockdiag(
        spdiagm(betaPrec),
        kron(speye(nUnq), (sw2^2)*Q2),
        (1/sw1^2)*B1'*spdiagm(1 ./ F1)*B1
        ) + Dsgn'*spdiagm(pg)*Dsgn

       effects .= getGaussSamp!(Qc, Q, zProj)

       #######################
       # Sample pg
       #######################

       pg .= rpg.(Dsgn*effects)

       ###########################
       # Sample sw1, rangeS1, rangeT1
       ###########################

       if i >= adaptStart
        theta1Var .= (2.4^2/3)*cov(theta1mat[1:i,:])
       end

       currentTheta1 = theta1mat[i,:]
       propTheta1 = currentTheta1 + cholesky(theta1Var).L*randn(3)

       sw1p, rangeS1p, rangeT1p = exp.(propTheta1)

       getNNGPmatsST!(B1p, F1p, B1Order, nb, data.loc, data.time, rangeS1p, rangeT1p)

       llProp = wll(B1p, F1p, sw1p^2, w)
       ll = wll(B1, F1, sw1^2, w)

       priorProp = pcpriorST([sw1p, rangeS1p, rangeT1p], priors.theta10, priors.alpha10)
       prior = pcpriorST([sw1, rangeS1, rangeT1], priors.theta10, priors.alpha10)

       acceptProb = exp.(llProp + priorProp + sum(propTheta1) - ll - prior - sum(currentTheta1))

       acceptTheta1 = rand(1)[1] < acceptProb


       if acceptTheta1
            sw1, rangeS1, rangeT1 = sw1p, rangeS1p, rangeT1p
            theta1mat[i+1,:] = copy(propTheta1)
            B1.nzval .= B1p.nzval
            F1 .= F1p
       else
            theta1mat[i+1,:] = copy(currentTheta1)
       end

       ###########################
       # Sample sw2, rangeT2
       ###########################

       if i >= adaptStart
        theta2Var .= (2.4^2/2)*cov(theta2mat[1:i,:])
       end

       currentTheta2 = theta2mat[i,:]
       propTheta2 = currentTheta2 + cholesky(theta2Var).L*randn(2)

       sw2p, rangeT2p = exp.(propTheta2)

       expCor!(Q2p, timeKnots, rangeT2p)
       expCor!(B2Compact, data.time, timeKnots, rangeT2p)
       B2p.nzval .= view(vec(B2Compact'), B2Order)
       Dsgnp .= sparse_hcat(data.X, sw2p^2*B2p, speye(n))

       w2mat = reshape(w2, nKnots, nUnq)
       w2cross = w2mat*w2mat'

       llProp = -0.5*( (sw2p^2)*tr(w2cross*Q2p) - 2*kt*log(sw2p) - nUnq*logdet(Q2p) )
       ll = -0.5*( (sw2^2)*tr(w2cross*Q2) - 2*kt*log(sw2) - nUnq*logdet(Q2) )

       priorProp = pcprior([sw2p, rangeT2p], priors.theta20, priors.alpha20)
       prior = pcprior([sw2, rangeT2], priors.theta20, priors.alpha20)

       acceptProb = exp.(llProp + priorProp + sum(propTheta2) - ll - prior - sum(currentTheta2))

       acceptTheta2 = rand(1)[1] < acceptProb


       if acceptTheta2
            sw2, rangeT2 = sw2p, rangeT2p
            theta2mat[i+1,:] = copy(propTheta2)
            Q2 .= copy(Q2p)
            B2.nzval .= copy(B2p.nzval)
            Dsgn.nzval = copy(Dsgnp.nzval)

       else
            theta2mat[i+1,:] = copy(currentTheta2)
       end




       #########################
       # Good work, team. Let's write out these results
       #########################

       paramsDf[1,:] = [beta; [sw1, rangeS1, rangeT1, sw2, rangeT2]]
       wDf[1,:] = w

       CSV.write(paramsOut, paramsDf; append = true, header = false)
       CSV.write(wOut, wDf; append = true, header = false)






    end


    return nothing


end