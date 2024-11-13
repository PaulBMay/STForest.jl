function bernoulli_predict(readdir, Xpred, locpred, timepred, m)

    npred = size(locpred,1)

    params = CSV.read(joinpath(readdir, "zparams.csv"), DataFrame)

    sw = params.sw
    rangeS = params.rangeS
    rangeT = params.rangeT

    beta = Matrix(select(params, r"beta_\d"))

    p = size(beta,2)
    p == size(Xpred, 2) || error("Read coeffecients of different dims than provided XPred")
    
    w = CSV.read(joinpath(readdir, "wz.csv"), Tables.matrix)

    loctime = CSV.read(joinpath(readdir, "locTime.csv"), Tables.matrix)
    loc = loctime[:,1:2]
    time = loctime[:,[3]]

    nSamps = size(w,1)

    ###

    print("Getting neighbors...\n")
    nb = getNeighborsP(loc, locpred, m)

    print("Getting initial NNGP mats...\n")
    B, F, BOrder = getNNGPmatsSTP(nb, loc, time, locpred, timepred, rangeS[1], rangeT[1])

    predSamps = zeros(Int64, nSamps, npred)

    ###############
    # Compute mean and variance iteratively
    ##############

    print("Start predictions with $nSamps samples\n")

    @views for i in ProgressBar(1:nSamps)

        getNNGPmatsSTP!(B, F, BOrder, nb, loc, time, locpred, timepred, rangeS[i], rangeT[i])

        mu = softmax.(Xpred*beta[i,:] + B*w[i,:] + sw[i]*sqrt.(F).*randn(npred))

        predSamps[i,:] = 1*(rand(npred) .< mu)

    end

    return predSamps

end