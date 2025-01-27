function agg_predict(read_dir, XPred, locPred, timePred, pwr, m, Projection)

    #############
    # Internal functions
    #############

    function softmax(x)
        return 1/(1 + exp(-x))
    end

    function ratpwr(x, pwr)
        return  (abs(x)^pwr)*sign(x)
    end

    ###############
    # Read data
    ###############

    yparams = CSV.read(joinpath(read_dir, "yparams.csv"), DataFrame)

    zparams = CSV.read(joinpath(read_dir, "zparams.csv"), DataFrame)

    betay = Matrix(select(yparams, r"beta_\d"))
    betaz = Matrix(select(zparams, r"beta_\d"))

    p = size(betay,2)

    if p != size(XPred,2)
        error("Read coeffecients of different dims than provided XPred")
    end

    wy1 = CSV.read(joinpath(read_dir, "wy1.csv"), Tables.matrix)
    wz = CSV.read(joinpath(read_dir, "wz.csv"), Tables.matrix)

    loctime = CSV.read(joinpath(read_dir, "locTime.csv"), Tables.matrix)
    loc = view(loctime, :, 1:2)
    time = view(loctime, :, [3])

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
    nProj = size(Projection,1)

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

    nbz = getNeighborsP(loc, locPred, m)
    nby = getNeighborsP(locpos, locPred, m)

    # Initialize NNGP mats

    println("Initial NNGP mats")

    Bz, Fz, BzOrder = getNNGPmatsSTP(nbz, loc, time, locPred, timePred, zparams.rangeS[1], zparams.rangeT[1])
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

    predSamps = zeros(Float64, nSamps, nProj)


    ###############
    # Compute mean and variance iteratively
    ##############

    print("Start predictions with $nSamps samples\n")

    @views for i in ProgressBar(1:nSamps)


        getNNGPmatsSTP!(Bz, Fz, BzOrder, nbz, loc, time, locPred, timePred, zparams.rangeS[i], zparams.rangeT[i])
        getNNGPmatsSTP!(By, Fy, ByOrder, nby, locpos, timepos, locPred, timePred, yparams.rangeS1[i], yparams.rangeT1[i])

        # Z stuff
        zprob = softmax.(
            XPred*betaz[i,:] + 
            Bz*wz[i,:] +
            zparams.sw[i]*sqrt.(Fz).*randn(nPred)
        )

        z = 1 .* (zprob .> rand(nPred))

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

        yp = ratpwr.(ypt, pwr)

        # Combine, project, and update mean and variance

        predSamps[i,:] = Projection*(yp.*z)


    end

    return predSamps



end


function agg_predict(readdir_z, readdir_y, XPred, locPred, timePred, pwr, m, Projection)

    #############
    # Internal functions
    #############

    function softmax(x)
        return 1/(1 + exp(-x))
    end

    function ratpwr(x, pwr)
        return  (abs(x)^pwr)*sign(x)
    end

    ###############
    # Read data
    ###############

    yparams = CSV.read(joinpath(readdir_y, "yparams.csv"), DataFrame)

    zparams = CSV.read(joinpath(readdir_z, "zparams.csv"), DataFrame)

    betay = Matrix(select(yparams, r"beta_\d"))
    betaz = Matrix(select(zparams, r"beta_\d"))

    p = size(betay,2)

    if p != size(XPred,2)
        error("Read coeffecients of different dims than provided XPred")
    end

    wy1 = CSV.read(joinpath(readdir_y, "wy1.csv"), Tables.matrix)
    wz = CSV.read(joinpath(readdir_z, "wz.csv"), Tables.matrix)

    loctime = CSV.read(joinpath(readdir_z, "locTime.csv"), Tables.matrix)
    loc = view(loctime, :, 1:2)
    time = view(loctime, :, [3])

    loctimepos = CSV.read(joinpath(readdir_y, "locTimePos.csv"), Tables.matrix)
    locpos = view(loctimepos, :, 1:2)
    timepos = view(loctimepos, :, [3])

    #locposunq = CSV.read(joinpath(read_dir, "locPosUnq.csv"), Tables.matrix)

    timeknots = CSV.read(joinpath(readdir_y, "timeKnots.csv"), Tables.matrix)

    nKnots = size(timeknots, 1)

    ###########
    # Dimensions
    #############

    nSamps = size(wy1,1)
    nPred = size(locPred,1)
    nProj = size(Projection,1)

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

    nbz = getNeighborsP(loc, locPred, m)
    nby = getNeighborsP(locpos, locPred, m)

    # Initialize NNGP mats

    print("Initial NNGP mats")

    Bz, Fz, BzOrder = getNNGPmatsSTP(nbz, loc, time, locPred, timePred, zparams.rangeS1[1], zparams.rangeT1[1])
    By, Fy, ByOrder = getNNGPmatsSTP(nby, locpos, timepos, locPred, timePred, yparams.rangeS1[1], yparams.rangeT1[1])

    if multTimes

        map2unq = indexin(loc2str(locPred), loc2str(locPredUnq))
        BtRows, BtCols, BtOrder = getBtNZ(nPred, map2unq, nKnots)

        BtyCompact = expCor(timePred, timeknots, yparams.rangeT2[1])
        Bty = sparse(BtRows, BtCols, view(vec(BtyCompact'), BtOrder))
        Qty = expCor(timeknots, yparams.rangeT2[1])

        BtzCompact = expCor(timePred, timeknots, zparams.rangeT2[1])
        Btz = sparse(BtRows, BtCols, view(vec(BtzCompact'), BtOrder))
        Qtz = expCor(timeknots, zparams.rangeT2[1])

    end

    #############
    # Prepare results
    #############

    predSamps = zeros(Float64, nSamps, nProj)


    ###############
    # Compute mean and variance iteratively
    ##############

    print("Start predictions with $nSamps samples\n")

    @views for i in ProgressBar(1:nSamps)


        getNNGPmatsSTP!(Bz, Fz, BzOrder, nbz, loc, time, locPred, timePred, zparams.rangeS1[i], zparams.rangeT1[i])
        getNNGPmatsSTP!(By, Fy, ByOrder, nby, locpos, timepos, locPred, timePred, yparams.rangeS1[i], yparams.rangeT1[i])

        gz = XPred*betaz[i,:] + Bz*wz[i,:] + zparams.sw1[i]*sqrt.(Fz).*randn(nPred)
        gy = XPred*betay[i,:] + By*wy1[i,:] + yparams.sw1[i]*(sqrt.(Fy).*randn(nPred)) + sqrt(yparams.tSq[i])*randn(nPred)

        if multTimes

            expCor!(Qty, timeknots, yparams.rangeT2[i])
            expCor!(BtyCompact, timePred, timeknots, yparams.rangeT2[i])
            Bty.nzval .= view(vec(BtyCompact'), BtOrder)
            wy2 = vec( cholesky(Qty).U \ randn(nKnots, nPredUnq))

            gy += yparams.sw2[i]*(Bty*wy2)

            expCor!(Qtz, timeknots, zparams.rangeT2[i])
            expCor!(BtzCompact, timePred, timeknots, zparams.rangeT2[i])
            Btz.nzval .= view(vec(BtzCompact'), BtOrder)
            wz2 = vec( cholesky(Qtz).U \ randn(nKnots, nPredUnq))

            gz += zparams.sw2[i]*(Btz*wz2)

        else

            gy += yparams.sw2[i]*rand(nPred)
            gz += zparams.sw2[i]*rand(nPred)

        end

        yp = ratpwr.(gy, pwr)
        z = 1 .* (softmax.(gz) .> rand(nPred))

        # Combine, project, and update mean and variance

        predSamps[i,:] = Projection*(yp.*z)


    end

    return predSamps



end

