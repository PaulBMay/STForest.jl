function NNGP_Bernoulli2(data::InputData, m::Int64, initparams::NamedTuple, priors::NamedTuple, thetaVar::Matrix, outDir::String, nSamps::Int64; thetalog = false)

    ###################
    # Check to see if the Tuples have the required fields
    ####################

    # params
    gtg = haskey(initparams, :sw1) & haskey(initparams, :rangeS1) & haskey(initparams, :rangeT1) & haskey(initparams, :sw2) & haskey(initparams, :rangeS2)
    if !gtg
        error("bad 'initparams': The expected fields are 'sw1, rangeS1, rangeT, sw2, rangeS2'.") 
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

    size(priors.beta,1) == p || error("nrows of priors.beta does not match ncols of X")

    betaMu = priors.beta[:,1]
    betaPrec = priors.beta[:,2]

    locUnq = unique(data.loc, dims = 1)
    nUnq = size(locUnq, 1)
    map2unq = match(loc2str(data.loc), loc2str(locUnq))

    P = sparse(1:n, map2unq, 1)

    # Does the out_dir exist?

    if !isdir(outDir)
        error("Can't find your outDir")
    end

    # Prepare CSV's

    loctimeOut = joinpath(outDir, "locTime.csv")
    CSV.write(loctimeOut, DataFrame(lon = data.loc[:,1], lat = data.loc[:,2], time = data.time[:,1]))

    locunqOut = joinpath(outDir, "locUnq.csv")
    CSV.write(locunqOut, DataFrame(lon = locUnq[:,1], lat = locUnq[:,2]))


    paramsOut = joinpath(outDir, "zparams.csv")
    paramsDf = DataFrame(zeros(1, p + 5), 
        ["beta_".*string.(0:(p-1)); ["sw1", "rangeS1", "rangeT1", "sw2", "rangeS2"]]
        )

    w1Out = joinpath(outDir, "w1z.csv")
    w1Df = DataFrame(zeros(1,n), "w_".*string.(1:n))

    w2Out = joinpath(outDir, "w2z.csv")
    w2Df = DataFrame(zeros(1,nUnq), "w_".*string.(1:nUnq))


    # fixed/random effect values
    effects = zeros(p + n + nUnq)
    beta = view(effects, 1:p)
    w1 = view(effects, (p+1):(p+n))
    w2 = view(effects, (p+n+1):(p+n+nUnq))

    sw1, rangeS1, rangeT1, sw2, rangeS2 = initparams.sw1, initparams.rangeS1, initparams.rangeT1, initparams.sw2, initparams.rangeS2

    paramsDf[1,:] = [beta; [sw1, rangeS1, rangeT1, sw2, rangeS2]]
    w1Df[1,:] = w1
    w2Df[1,:] = w2

    CSV.write(paramsOut, paramsDf)
    CSV.write(w1Out, w1Df)
    CSV.write(w2Out, w2Df)

    if thetalog

        thetalogdf = DataFrame(zeros(16)', ["sw1", "rangeS1", "rangeT1", "sw2", "rangeS2", "ll", "prior", "sw1p", "rangeS1p", "rangeT1p", "sw2p", "rangeS2p", "llp", "priorp", "acceptprob", "accept"])
        thetalogdf.accept .= false

        thetalogout = joinpath(outDir, "zthetalog.csv")
        CSV.write(thetalogout, thetalogdf)

    end

    ####################
    # Lord have mercy that was boring.
    # Now fun stuff. Get the neighbor sets and initial NNGP mats
    #####################

    print("Getting neighbor sets\n")

    nb1 = getNeighbors(data.loc, m)
    nb2 = getNeighbors(locUnq, m)

    print("Initial NNGP mats\n")

    B1,F1,B1Order = getNNGPmatsST(nb1, data.loc, data.time, rangeS1, rangeT1)
    B1p,F1p = copy(B1), copy(F1)

    B2,F2,B2Order = getNNGPmatsS(nb1, locUnq, rangeS2)
    B2p,F2p = copy(B2), copy(F2)

    Dsgn = sparse_hcat(data.X, speye(n), P)

    zProj = Dsgn'*(data.y .- 0.5)

    zProj[1:p] += betaPrec .* betaMu

    pg = rpg.(fill(0.3, n))




    ##############

    Q = blockdiag(
        spdiagm(betaPrec),
        (1/sw1^2)*B1'*spdiagm(1 ./ F1)*B1,
        (1/sw2^2)*B2'*spdiagm(1 ./ F2)*B2
    ) + Dsgn'*spdiagm(pg)*Dsgn

    Qc = cholesky(Hermitian(Q))

    currentTheta = log.([sw1, rangeS1, rangeT1, sw2, rangeS2])
    propTheta = copy(currentTheta)
    acceptTheta = 0

    prop_chol = cholesky(thetaVar).L

    #########################
    # Begin Gibbs sampler
    #########################

    for i = ProgressBar(1:nSamps)

       ############################
       # Sample beta, w
       ############################

       Q .= blockdiag(
            spdiagm(betaPrec),
            (1/sw1^2)*B1'*spdiagm(1 ./ F1)*B1,
            (1/sw2^2)*B2'*spdiagm(1 ./ F2)*B2
        ) + Dsgn'*spdiagm(pg)*Dsgn

       effects .= getGaussSamp!(Qc, Q, zProj)

       #######################
       # Sample pg
       #######################

       pg .= rpg.(Dsgn*effects)

       ###########################
       # Sample sw, rangeS
       ###########################

       propTheta = currentTheta + prop_chol*randn(5)

       sw1p, rangeS1p, rangeT1p, sw2p, rangeS2p = exp.(propTheta)

       getNNGPmatsST!(B1p, F1p, B1Order, nb1, data.loc, data.time, rangeS1p, rangeT1p)
       getNNGPmatsS!(B2p, F2p, B2Order, nb2, locUnq, rangeS2p)

       llProp = wll(B1p, F1p, sw1p^2, w1) + wll(B2p, F2p, sw2p^2, w2)
       ll = wll(B1, F1, sw1^2, w1) + wll(B2, F2, sw2^2, w2)

       priorProp = pcpriorST([sw1p, rangeS1p, rangeT1p], priors.theta10, priors.alpha10) + pcpriorS([sw2p, rangeS2p], priors.theta20, priors.alpha20)
       prior = pcpriorST([sw1, rangeS1, rangeT1], priors.theta10, priors.alpha10) + pcpriorS([sw2, rangeS2], priors.theta20, priors.alpha20)

       acceptProb = exp.(llProp + priorProp + sum(propTheta) - ll - prior - sum(currentTheta))

       acceptTheta = rand(1)[1] < acceptProb

       if thetalog
        thetalogdf[1,:] = [sw1, rangeS1, rangeT1, sw2, rangeS2, ll, prior, sw1p, rangeS1p, rangeT1p, sw2p, rangeS2p, llProp, priorProp, acceptProb, acceptTheta]
       end

       if acceptTheta
            sw1, rangeS1, rangeT1, sw1, rangeS2 = sw1p, rangeS1p, rangeT1p, sw2p, rangeS2p
            currentTheta .= copy(propTheta)
            B1.nzval .= B1p.nzval
            F1 .= F1p
            B2.nzval .= B2p.nzval
            F2 .= F2p
       end



       #########################
       # Good work, team. Let's write out these results
       #########################

       paramsDf[1,:] = [beta; [sw1, rangeS1, rangeT1, sw2, rangeS2]]
       w1Df[1,:] = w1
       w2Df[1,:] = w2

       CSV.write(paramsOut, paramsDf; append = true, header = false)
       CSV.write(w1Out, w1Df; append = true, header = false)
       CSV.write(w2Out, w2Df; append = true, header = false)
       if thetalog
        CSV.write(thetalogout, thetalogdf; append = true, header = false)
       end





    end


    return nothing


end