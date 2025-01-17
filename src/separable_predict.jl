function separable_predict(read_dir, XPred, locPred, timePred, m)

    ###############
    # Read data
    ###############

    yparams = CSV.read(joinpath(read_dir, "yparams.csv"), DataFrame)

    betay = Matrix(select(yparams, r"beta_\d"))

    p = size(betay,2)

    if p != size(XPred,2)
        error("Read coeffecients of different dims than provided XPred")
    end

    wy = CSV.read(joinpath(read_dir, "wy.csv"), Tables.matrix)


    loctimepos = CSV.read(joinpath(read_dir, "locTimePos.csv"), Tables.matrix)
    locpos = view(loctimepos, :, 1:2)
    timepos = view(loctimepos, :, [3])


    ###########
    # Dimensions
    #############

    nSamps = size(wy,1)
    nPred = size(locPred,1)


    ##########
    # Neighbor sets
    #############

    print("Getting neighbors...\n")

    nby = getNeighborsP(locpos, locPred, m)

    # Initialize NNGP mats

    print("Initial NNGP mats")

    By, Fy, ByOrder = getNNGPmatsSTP(nby, locpos, timepos, locPred, timePred, yparams.rangeS[1], yparams.rangeT[1])


    #############
    # Prepare results
    #############

    predSamps = zeros(Float64, nSamps, nPred)


    ###############
    # Compute mean and variance iteratively
    ##############

    print("Start predictions with $nSamps samples\n")

    @views for i in ProgressBar(1:nSamps)


        getNNGPmatsSTP!(By, Fy, ByOrder, nby, locpos, timepos, locPred, timePred, yparams.rangeS[i], yparams.rangeT[i])


        # Yp stuff

        ypt = XPred*betay[i,:] + By*wy[i,:] + yparams.sw[i]*(sqrt.(Fy).*randn(nPred)) + sqrt(yparams.tSq[i])*randn(nPred)

        # Combine, project, and update mean and variance

        predSamps[i,:] = ypt


    end

    return predSamps



end