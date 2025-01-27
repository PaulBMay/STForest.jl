function bernoulli_nonsep_predict(readdir, Xpred, locpred, timepred, timeknots, m)

    npred = size(locpred,1)
    nknots = size(timeknots,1)

    params = CSV.read(joinpath(readdir, "zparams.csv"), DataFrame)

    sw1 = params.sw1
    rangeS1 = params.rangeS1
    rangeT1 = params.rangeT1

    sw2 = params.sw2
    rangeT2 = params.rangeT2

    beta = Matrix(select(params, r"beta_\d"))

    p = size(beta,2)
    p == size(Xpred, 2) || error("Read coeffecients of different dims than provided XPred")
    
    w = CSV.read(joinpath(readdir, "wz.csv"), Tables.matrix)

    loctime = CSV.read(joinpath(readdir, "locTime.csv"), Tables.matrix)
    loc = loctime[:,1:2]
    time = loctime[:,[3]]

    nsamps = size(w,1)

    ###

    print("Getting neighbors...\n")
    nb = getNeighborsP(loc, locpred, m)

    print("Getting initial NNGP mats...\n")
    B, F, BOrder = getNNGPmatsSTP(nb, loc, time, locpred, timepred, rangeS1[1], rangeT1[1])

    ############
    # Unique locations
    ############

    locpredunq = unique(locpred, dims = 1)
    npredunq = size(locpredunq, 1)

    multTimes = nPredUnq < npred


    if multTimes

        map2unq = indexin(loc2str(locpred), loc2str(locpredunq))
        BtRows, BtCols, BtOrder = getBtNZ(npred, map2unq, nknots)
        BtCompact = expCor(timepred, timeknots, rangeT2[1])
        Bt = sparse(BtRows, BtCols, view(vec(BtCompact'), BtOrder))
        Qt = expCor(timeknots, rangeT2[1])

    end


    predSamps = zeros(nsamps, npred)

    ###############
    # Compute mean and variance iteratively
    ##############

    print("Start predictions with $nsamps samples\n")

    @views for i in ProgressBar(1:nsamps)

        getNNGPmatsSTP!(B, F, BOrder, nb, loc, time, locpred, timepred, rangeS1[i], rangeT1[i])

        gz = Xpred*beta[i,:] + B*w[i,:] + sw1[i]*sqrt.(F).*randn(npred)

        if multTimes

            expCor!(Qt, timeknots, rangeT2[i])
            expCor!(BtCompact, timepred, timeknots, rangeT2[i])
            Bt.nzval .= view(vec(BtCompact'), BtOrder)
            w2 = vec( cholesky(Qt).U \ randn(nknots, npredunq))

            gz += sw2[i]*(Bt*w2)

        else

            gz += sw2[i]*rand(npred)

        end

        predSamps[i,:] = softmax.(gz)
    end

    return predSamps

end