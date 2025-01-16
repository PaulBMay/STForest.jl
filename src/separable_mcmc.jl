function separable_mcmc(data::InputData, m::Integer, initparams::NamedTuple, spriors::NamedTuple, thetaVar::Matrix, outDir::String, nSamps::Int64)

    # params
    gtg = haskey(initparams, :sw) & haskey(initparams, :rangeS) & haskey(initparams, :rangeT) & haskey(initparams, :tSq)
    if !gtg
        error("bad 'initparams': The expected fields are 'sw, rangeS, rangeT, tSq'.") 
    end
    # spriors
    gtg = haskey(spriors, :theta0) & haskey(spriors, :alpha0)
    if !gtg
        error("bad 'spriors': The expected fields are 'theta0, alpha0'.") 
    end
     
    ############################
    # Data dimensions and prep
    ##########################

    n = size(data.y, 1) # sample size
    p = size(data.X, 2) # num predictors


    ##########################
    # Initial values and CSV creation
    ##########################

    # Does the out_dir exist?

    if !isdir(outDir)
        error("Can't find your outDir")
    end

    # Prepare CSV's

    loctimeOut = joinpath(outDir, "locTimePos.csv")
    CSV.write(loctimeOut, DataFrame(lon = data.loc[:,1], lat = data.loc[:,2], time = data.time[:,1]))

    paramsOut = joinpath(outDir, "yparams.csv")
    paramsDf = DataFrame(zeros(1, p + 4), 
        ["beta_".*string.(0:(p-1)); ["sw", "rangeS", "rangeT", "tSq"]]
        )

    wOut = joinpath(outDir, "wy.csv")
    wDf = DataFrame(zeros(1,n), "w_".*string.(1:n))

    

    # Parameter/effect values

    effects = zeros(p + n)
    beta = view(effects, 1:p)
    w = view(effects, (p+1):(p+n))

    sw, rangeS, rangeT, tSq = initparams.sw, initparams.rangeS, initparams.rangeT, initparams.tSq
    swp, rangeSp, rangeTp, tSqp = sw, rangeS, rangeT, tSq


    paramsDf[1,:] = [beta; [sw, rangeS, rangeT, tSq]]
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

    Dsgn = sparse_hcat(data.X, speye(n)) 

    yProj = Dsgn'*(data.y ./ tSq)

    betaPrec = fill(0.01, p)

    ##############

    Q = blockdiag(
        spdiagm(betaPrec),
        (1/sw^2)*B'*spdiagm(1 ./ F)*B
    ) + (1/tSq)*Dsgn'*Dsgn

    Qc = cholesky(Hermitian(Q))
    Qpc = copy(Qc)

    currentTheta = log.([sw, rangeS, rangeT, tSq])
    propTheta = copy(currentTheta)
    lp = thetayLP(currentTheta, spriors, data.y, betaPrec, F, Dsgn, Qc)
    lpProp = lp
    acceptTheta = false

    prop_chol = cholesky(thetaVar).L

    lpDf = DataFrame(lp = lp, lpProp = lpProp, accept = acceptTheta*1)
    lpOut = joinpath(outDir, "lp.csv")
    CSV.write(lpOut, lpDf)


    #########################
    # Begin Gibbs sampler
    #########################

    for i = ProgressBar(1:nSamps)

       ########################
       # Sample beta, w1, w2
       ########################

       effects .= getGaussSamp(Qc, yProj)

       ###########################
       # Sample all spatial parameters associated with y
       ###########################

       lpDf.lp[1] = lp

       propTheta = currentTheta + prop_chol*randn(4)

       swp, rangeSp, rangeTp, tSqp = exp.(propTheta)

       # Get NNGP and fixed-rank matrices associated with the proposal values
       getNNGPmatsST!(B, F, BOrder, nb, data.loc, data.time, rangeSp, rangeTp)
       # Posterior precision for the proposal values
       Dsgn .= sparse_hcat(data.X, speye(n))
       Q .= blockdiag(
            spdiagm(betaPrec),
            (1/swp^2)*B'*spdiagm(1 ./ F)*B
       ) + (1/tSqp)*Dsgn'*Dsgn

       cholesky!(Qpc, Hermitian(Q))

       lpProp = thetayLP(propTheta, spriors, data.y, betaPrec, F, Dsgn, Qpc)

       acceptProb = exp.(lpProp + sum(propTheta)  - lp - sum(currentTheta))

       acceptTheta = rand(1)[1] < acceptProb

       if acceptTheta
        sw, rangeS, rangeT, tSq = exp.(propTheta)
        Qc = copy(Qpc)
        lp = lpProp
        yProj .= Dsgn'*(data.y ./ tSq)
        currentTheta .= copy(propTheta)
       end

       lpDf.lpProp[1] = lpProp

       lpDf.accept[1] = acceptTheta*1

       #########################
       # Good work, team. Let's write out these results
       #########################

       paramsDf[1,:] = [beta; [sw, rangeS, rangeT, tSq]]
       wDf[1,:] = w

       CSV.write(paramsOut, paramsDf; append = true, header = false)
       CSV.write(wOut, wDf; append = true, header = false)
       CSV.write(lpOut, lpDf; append = true, header = false)

    end



    return nothing


end


function thetayLP(thetay::Vector, spriors::NamedTuple, yp::Vector,  betayprec::Vector, Fy::Vector, Dsgny::SparseMatrixCSC, QyPostChol::SparseArrays.CHOLMOD.Factor)

    local swy, rangeSy, rangeTy, t2y = exp.(thetay)

    local np = length(yp)

    local ypSolve = (yp ./ t2y) - (Dsgny*(QyPostChol \ (Dsgny'*yp) ) ./ t2y^2)

    local sse = dot(yp, ypSolve)

    local priorldet = sum(log.(betayprec)) - 2*np*log(swy) - sum(log.(Fy))

    local ldet = np*log(t2y) + logdet(QyPostChol) - priorldet

    local ll = -0.5*(sse + ldet)

    local prior1 = pcpriorST([swy, rangeSy, rangeTy], spriors.theta0, spriors.alpha0)

    local lp = ll + prior1 

    return lp


end
