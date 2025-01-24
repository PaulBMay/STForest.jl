function bernoulli_nonsep_predict(readdir, Xpred, locpred, timepred, m)

    npred = size(locpred,1)

    params = CSV.read(joinpath(readdir, "zparams.csv"), DataFrame)

    sw1 = params.sw1
    rangeS1 = params.rangeS1
    rangeT1 = params.rangeT1

    sw2 = params.sw2

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
    B, F, BOrder = getNNGPmatsSTP(nb, loc, time, locpred, timepred, rangeS1[1], rangeT1[1])

    predSamps = zeros(nSamps, npred)

    ###############
    # Compute mean and variance iteratively
    ##############

    print("Start predictions with $nSamps samples\n")

    @views for i in ProgressBar(1:nSamps)

        getNNGPmatsSTP!(B, F, BOrder, nb, loc, time, locpred, timepred, rangeS1[i], rangeT1[i])

        mu = softmax.(Xpred*beta[i,:] + B*w[i,:] + sw1[i]*sqrt.(F).*randn(npred) + sw2[i]*randn(npred))

        predSamps[i,:] = mu

    end

    return predSamps

end