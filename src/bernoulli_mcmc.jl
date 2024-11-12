function NNGP_Bernoulli(data::InputData, m::Int64, initparams::NamedTuple, priors::NamedTuple, thetaVar::Matrix, outDir::String, nSamps::Int64)

    ###################
    # Check to see if the Tuples have the required fields
    ####################

    # params
    gtg = haskey(initparams, :sw) & haskey(initparams, :rangeS) & haskey(initparams, :rangeT)
    if !gtg
        error("bad 'initparams': The expected fields are 'sw, rangeS, rangeT'.") 
    end
    # priors
    gtg = haskey(priors, :theta0) & haskey(priors, :alpha0) & haskey(priors, :beta)
    if !gtg
        error("bad 'priors': The expected fields are 'theta0, alpha0'.") 
    end



    ############################
    # Data dimensions and prep
    ##########################

    n = size(data.y, 1) # sample size
    p = size(data.X, 2) # num predictors

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
    paramsDf = DataFrame(zeros(1, p + 3), 
        ["beta_".*string.(0:(p-1)); ["sw", "rangeS", "rangeT"]]
        )

    wOut = joinpath(outDir, "wz.csv")
    wDf = DataFrame(zeros(1,n), "w_".*string.(1:n))


    # fixed/random effect values
    effects = zeros(p + n)
    w = view(effects, (p+1):(p+n))
    beta = view(effects, 1:p)

    sw, rangeS, rangeT = initparams.sw, initparams.rangeS, initparams.rangeT

    paramsDf[1,:] = [beta; [sw, rangeS, rangeT]]
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

    B,F,BOrder = getNNGPmatsST(nb, data.loc, data.time, rangeS, rangeT)
    Bp,Fp = copy(B), copy(F)

    Dsgn = sparse_hcat(data.X, speye(n))

    zProj = Dsgn'*(data.y .- 0.5)

    zProj[1:p] += betaPrec .* betaMu

    pg = rpg.(fill(0.3, n))




    ##############

    Q = blockdiag(
        spdiagm(betaPrec),
        (1/sw^2)*B'*spdiagm(1 ./ F)*B
    ) + Dsgn'*spdiagm(pg)*Dsgn

    Qc = cholesky(Hermitian(Q))

    currentTheta = log.([sw, rangeS, rangeT])
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
            (1/sw^2)*B'*spdiagm(1 ./ F)*B
        ) + Dsgn'*spdiagm(pg)*Dsgn

       effects .= getGaussSamp!(Qc, Q, zProj)

       #######################
       # Sample pg
       #######################

       pg .= rpg.(Dsgn*effects)

       ###########################
       # Sample sw, rangeS
       ###########################

       propTheta = currentTheta + prop_chol*randn(3)

       swp, rangeSp, rangeTp = exp.(propTheta)

       getNNGPmatsST!(Bp, Fp, BOrder, nb, data.loc, data.time, rangeSp, rangeTp)

       llProp = wll(Bp, Fp, swp^2, w)
       ll = wll(B, F, sw^2, w)

       priorProp = pcpriorST([swp, rangeSp, rangeTp], priors.theta0, priors.alpha0)
       prior = pcprior([sw, rangeS, rangeT], priors.theta0, priors.alpha0)

       acceptProb = exp.(llProp + priorProp + sum(propTheta) - ll - prior - sum(currentTheta))

       acceptTheta = rand(1)[1] < acceptProb

       if acceptTheta
            sw, rangeS, rangeT = swp, rangeSp, rangeTp
            currentTheta .= copy(propTheta)
            B.nzval .= Bp.nzval
            F .= Fp
       end



       #########################
       # Good work, team. Let's write out these results
       #########################

       paramsDf[1,:] = [beta; [sw, rangeS, rangeT]]
       wDf[1,:] = w

       CSV.write(paramsOut, paramsDf; append = true, header = false)
       CSV.write(wOut, wDf; append = true, header = false)




    end


    return nothing


end