
function NNGP_Bernoulli(data::InputData, m::Int64, initparams::NamedTuple, priors::NamedTuple, thetaVar::Matrix, outDir::String, nSamps::Int64; thetalog = false)

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

    if thetalog

        thetalogdf = DataFrame(zeros(12)', ["sw", "rangeS", "rangeT", "ll", "prior", "swp", "rangeSp", "rangeTp", "llp", "priorp", "acceptprob", "accept"])
        thetalogdf.accept .= false

        thetalogout = joinpath(outDir, "zthetalog.csv")
        CSV.write(thetalogout, thetalogdf)

    end

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
       prior = pcpriorST([sw, rangeS, rangeT], priors.theta0, priors.alpha0)

       acceptProb = exp.(llProp + priorProp + sum(propTheta) - ll - prior - sum(currentTheta))

       acceptTheta = rand(1)[1] < acceptProb

       if thetalog
        thetalogdf[1,:] = [sw, rangeS, rangeT, ll, prior, swp, rangeSp, rangeTp, llProp, priorProp, acceptProb, acceptTheta]
       end

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
       if thetalog
        CSV.write(thetalogout, thetalogdf; append = true, header = false)
       end





    end


    return nothing


end

# adaptive proposal variance for theta
function NNGP_Bernoulli(data::InputData, m::Int64, initparams::NamedTuple, priors::NamedTuple, outDir::String, nSamps::Int64; adaptStart = 50, pgwarmup = 10, thetalog = false)

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

    if thetalog

        thetalogdf = DataFrame(zeros(12)', ["sw", "rangeS", "rangeT", "ll", "prior", "swp", "rangeSp", "rangeTp", "llp", "priorp", "acceptprob", "accept"])
        thetalogdf.accept .= false

        thetalogout = joinpath(outDir, "zthetalog.csv")
        CSV.write(thetalogout, thetalogdf)

    end

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

    acceptTheta = 0

    thetamat = zeros(nSamps+1, 3)
    thetamat[1,:] = log.([sw, rangeS, rangeT])

    thetaVar = 1e-5*Matrix(I,3,3)

    #####################
    # pg warmup
    #####################

    println("Warming up Polya-Gamma and random effect values")

    Qprior = blockdiag(
        spdiagm(betaPrec),
        (1/sw^2)*B'*spdiagm(1 ./ F)*B
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
       # Sample sw, rangeS, rangeT
       ###########################

       if i >= adaptStart
        thetaVar .= (2.4^2/3)*cov(thetamat[1:i,:])
       end

       currentTheta = thetamat[i,:]
       propTheta = currentTheta + cholesky(thetaVar).L*randn(3)

       swp, rangeSp, rangeTp = exp.(propTheta)

       getNNGPmatsST!(Bp, Fp, BOrder, nb, data.loc, data.time, rangeSp, rangeTp)

       llProp = wll(Bp, Fp, swp^2, w)
       ll = wll(B, F, sw^2, w)

       priorProp = pcpriorST([swp, rangeSp, rangeTp], priors.theta0, priors.alpha0)
       prior = pcpriorST([sw, rangeS, rangeT], priors.theta0, priors.alpha0)

       acceptProb = exp.(llProp + priorProp + sum(propTheta) - ll - prior - sum(currentTheta))

       acceptTheta = rand(1)[1] < acceptProb

       if thetalog
        thetalogdf[1,:] = [sw, rangeS, rangeT, ll, prior, swp, rangeSp, rangeTp, llProp, priorProp, acceptProb, acceptTheta]
       end

       if acceptTheta
            sw, rangeS, rangeT = swp, rangeSp, rangeTp
            thetamat[i+1,:] = copy(propTheta)
            B.nzval .= Bp.nzval
            F .= Fp
       else
            thetamat[i+1,:] = copy(currentTheta)
       end



       #########################
       # Good work, team. Let's write out these results
       #########################

       paramsDf[1,:] = [beta; [sw, rangeS, rangeT]]
       wDf[1,:] = w

       CSV.write(paramsOut, paramsDf; append = true, header = false)
       CSV.write(wOut, wDf; append = true, header = false)
       if thetalog
        CSV.write(thetalogout, thetalogdf; append = true, header = false)
       end





    end


    return nothing


end

# adaptive proposal and iterative solver to sample the latent effects

function getGaussSampITS!(Q, Pl, zProj, B, F, Dsgn, betaPrec, sw, omega, reltol)

    n = length(F)
    k = length(zProj)
    p = k - n

    Q .= blockdiag(
        spdiagm(betaPrec),
        (1/sw^2)*B'*spdiagm(1 ./ F)*B
    ) + Dsgn'*spdiagm(omega)*Dsgn

    UpdatePreconditioner!(Pl, Q)

    ztilde = copy(zProj)
    ztilde[1:p] += sqrt.(betaPrec) .* randn(p)
    ztilde[(p+1):k] += (1 / sw) *  ( B'*(sqrt.(1 ./ F) .* randn(n)) )
    ztilde += Dsgn' * (sqrt.(omega) .* randn(n))
    
    prob = LinearProblem(Q, ztilde)

    sol = solve(prob, KrylovJL_CG(), Pl = Pl, reltol = reltol)

    return sol.u

end


function NNGP_Bernoulli_ITS(data::InputData, m::Int64, initparams::NamedTuple, priors::NamedTuple, outDir::String, nSamps::Int64; adaptStart = 50, its_reltol = 1e-6, thetalog = false)

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

    if thetalog

        thetalogdf = DataFrame(zeros(12)', ["sw", "rangeS", "rangeT", "ll", "prior", "swp", "rangeSp", "rangeTp", "llp", "priorp", "acceptprob", "accept"])
        thetalogdf.accept .= false

        thetalogout = joinpath(outDir, "zthetalog.csv")
        CSV.write(thetalogout, thetalogdf)

    end

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

    Pl = CholeskyPreconditioner(Q, 2)

    #Qc = cholesky(Hermitian(Q))

    acceptTheta = 0

    thetamat = zeros(nSamps+1, 3)
    thetamat[1,:] = log.([sw, rangeS, rangeT])

    thetaVar = 1e-5*Matrix(I,3,3)

    #########################
    # Begin Gibbs sampler
    #########################

    for i = ProgressBar(1:nSamps)

       ############################
       # Sample beta, w
       ############################     

       effects .= getGaussSampITS!(Q, Pl, zProj, B, F, Dsgn, betaPrec, sw, pg, its_reltol)

       #######################
       # Sample pg
       #######################

       pg .= rpg.(Dsgn*effects)

       ###########################
       # Sample sw, rangeS, rangeT
       ###########################

       if i >= adaptStart
        thetaVar .= (2.4^2/3)*cov(thetamat[1:i,:])
       end

       currentTheta = thetamat[i,:]
       propTheta = currentTheta + cholesky(thetaVar).L*randn(3)

       swp, rangeSp, rangeTp = exp.(propTheta)

       getNNGPmatsST!(Bp, Fp, BOrder, nb, data.loc, data.time, rangeSp, rangeTp)

       llProp = wll(Bp, Fp, swp^2, w)
       ll = wll(B, F, sw^2, w)

       priorProp = pcpriorST([swp, rangeSp, rangeTp], priors.theta0, priors.alpha0)
       prior = pcpriorST([sw, rangeS, rangeT], priors.theta0, priors.alpha0)

       acceptProb = exp.(llProp + priorProp + sum(propTheta) - ll - prior - sum(currentTheta))

       acceptTheta = rand(1)[1] < acceptProb

       if thetalog
        thetalogdf[1,:] = [sw, rangeS, rangeT, ll, prior, swp, rangeSp, rangeTp, llProp, priorProp, acceptProb, acceptTheta]
       end

       if acceptTheta
            sw, rangeS, rangeT = swp, rangeSp, rangeTp
            thetamat[i+1,:] = copy(propTheta)
            B.nzval .= Bp.nzval
            F .= Fp
       else
            thetamat[i+1,:] = copy(currentTheta)
       end



       #########################
       # Good work, team. Let's write out these results
       #########################

       paramsDf[1,:] = [beta; [sw, rangeS, rangeT]]
       wDf[1,:] = w

       CSV.write(paramsOut, paramsDf; append = true, header = false)
       CSV.write(wOut, wDf; append = true, header = false)
       if thetalog
        CSV.write(thetalogout, thetalogdf; append = true, header = false)
       end





    end


    return nothing


end
