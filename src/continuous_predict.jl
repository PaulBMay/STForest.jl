function continuous_predict(read_dir, XPred, locPred, timePred, m)

    #############
    # Internal functions
    #############

    function softmax(x)
        return 1/(1 + exp(-x))
    end


    ###############
    # Read data
    ###############

    yparams = CSV.read(joinpath(read_dir, "yparams.csv"), DataFrame)

    betay = Matrix(select(yparams, r"beta_\d"))

    p = size(betay,2)

    if p != size(XPred,2)
        error("Read coeffecients of different dims than provided XPred")
    end

    wy1 = CSV.read(joinpath(read_dir, "wy1.csv"), Tables.matrix)


    loctimepos = CSV.read(joinpath(read_dir, "locTimePos.csv"), Tables.matrix)
    locpos = view(loctimepos, :, 1:2)
    timepos = view(loctimepos, :, [3])

    #locposunq = CSV.read(joinpath(read_dir, "locPosUnq.csv"), Tables.matrix)

    timeknots = CSV.read(joinpath(read_dir, "timeKnots.csv"), Tables.matrix)

    nKnots = size(timeknots, 1)

    ###########
    # Dimensions
    #############

    nSamps = size(wy1,1)
    nPred = size(locPred,1)

    ############
    # Unique locations
    ############

    locPredUnq = unique(locPred, dims = 1)
    nPredUnq = size(locPredUnq, 1)

    multTimes = nPredUnq < nPred

    ##########
    # Neighbor sets
    #############

    print("Getting neighbors...\n")

    nby = getNeighborsP(locpos, locPred, m)

    # Initialize NNGP mats

    print("Initial NNGP mats")

    By, Fy, ByOrder = getNNGPmatsSTP(nby, locpos, timepos, locPred, timePred, yparams.rangeS1[1], yparams.rangeT1[1])

    if multTimes
        map2unq = indexin(loc2str(locPred), loc2str(locPredUnq))
        BtRows, BtCols, BtOrder = getBtNZ(nPred, map2unq, nKnots)
        BtCompact = expCor(timePred, timeknots, yparams.rangeT2[1])
        Bt = sparse(BtRows, BtCols, view(vec(BtCompact'), BtOrder))
        Qt = expCor(timeknots, yparams.rangeT2[1])

    end

    #############
    # Prepare results
    #############

    predSamps = zeros(Float64, nSamps, nPred)


    ###############
    # Compute mean and variance iteratively
    ##############

    print("Start predictions with $nSamps samples\n")

    @views for i in ProgressBar(1:nSamps)


        getNNGPmatsSTP!(By, Fy, ByOrder, nby, locpos, timepos, locPred, timePred, yparams.rangeS1[i], yparams.rangeT1[i])


        # Yp stuff

        ypt = XPred*betay[i,:] + By*wy1[i,:] + yparams.sw1[i]*(sqrt.(Fy).*randn(nPred)) + sqrt(yparams.tSq[i])*randn(nPred)

        if multTimes

            expCor!(Qt, timeknots, yparams.rangeT2[i])
            expCor!(BtCompact, timePred, timeknots, yparams.rangeT2[i])
            Bt.nzval .= view(vec(BtCompact'), BtOrder)
            wy2 = vec( cholesky(Qt).U \ randn(nKnots, nPredUnq))

            ypt += yparams.sw2[i]*(Bt*wy2)

        else

            ypt += yparams.sw2[i]*rand(nPred)

        end

        # Combine, project, and update mean and variance

        predSamps[i,:] = ypt


    end

    return predSamps



end