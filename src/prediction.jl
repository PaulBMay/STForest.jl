function NNGP_ZIST_PRED(read_dir, XPred, locPred, timePred, pwr, m, Projection)

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

    params = CSV.read(joinpath(read_dir, "params.csv"), DataFrame)

    swy1 = params.swy1
    rangeSy1 = params.rangeSy1
    rangeTy1 = params.rangeTy1

    swy2 = params.swy2
    rangeTy2 = params.rangeTy2

    t2y = params.t2y

    swz = params.swz
    rangeSz = params.rangeSz

    betay = Matrix(select(params, r"betay_\d"))
    betaz = Matrix(select(params, r"betaz_\d"))

    p = size(betay,2)

    if p != size(XPred,2)
        error("Read coeffecients of different dims than provided XPred")
    end

    wy1 = CSV.read(joinpath(read_dir, "wy1.csv"), Tables.matrix)
    wz = CSV.read(joinpath(read_dir, "wz.csv"), Tables.matrix)

    locUnq = CSV.read(joinpath(read_dir, "loc_unq.csv"), Tables.matrix)
    loctimep = CSV.read(joinpath(read_dir, "loctime_pos.csv"), Tables.matrix)
    locp = loctimep[:,1:2]
    timep = loctimep[:,[3]]
    timeKnots = CSV.read(joinpath(read_dir, "time_knots.csv"), Tables.matrix)
    nKnots = size(timeKnots, 1)

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

    if multTimes
        # For each location in the full stack, what's it's position among the set of unique locations?
        map2unq = indexin(loc2str(locPred), loc2str(locPredUnq))

        # Organize this is a sparse matrix. Entry (i,j) is one if row i is associated with unique location j
        PPred = sparse(1:nPred, map2unq, true)

        # Covariates for the unique locations
        XPredz = (transpose(PPred)*XPred) ./ sum(PPred, dims = 1)[1,:]
    else
        XPredz = XPred
    end

    ##########
    # Neighbor sets
    #############

    print("Getting neighbors...\n")

    nbz = getNeighborsP(locUnq, locPredUnq, m)
    nby = getNeighborsP(locp, locPred, m)

    # Initialize NNGP mats

    print("Initial NNGP mats")

    Bz, Fz, BzOrder = getNNGPmatsSP(nbz, locUnq, locPredUnq, rangeSz[1])
    By, Fy, ByOrder = getNNGPmatsSTP(nby, locp, timep, locPred, timePred, rangeSy1[1], rangeTy1[1])

    if multTimes

        BtRows, BtCols, BtOrder = getBtNZ(nPred, map2unq, nKnots)
        BtCompact = expCor(timePred, timeKnots, rangeTy2[1])
        Bt = sparse(BtRows, BtCols, view(vec(BtCompact'), BtOrder))
        Qt =expCor(timeKnots, rangeTy2[1])

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


        getNNGPmatsSP!(Bz, Fz, BzOrder, nbz, locUnq, locPredUnq, rangeSz[i])
        getNNGPmatsSTP!(By, Fy, ByOrder, nby, locp, timep, locPred, timePred, rangeSy1[i], rangeTy1[i])

        # Z stuff
        zprob = softmax.(
            XPredz*betaz[i,:] + 
            Bz*wz[i,:] +
            swz[i]*sqrt.(Fz).*randn(nPredUnq)
        )

        z = PPred*(zprob .> rand(nPredUnq))

        # Yp stuff

        ypt = XPred*betay[i,:] + By*wy1[i,:] + swy1[i]*(sqrt.(Fy).*randn(nPred)) + sqrt(t2y[i])*randn(nPred)

        if multTimes

            expCor!(Qt, timeKnots, rangeTy2[i])
            expCor!(BtCompact, timePred, timeKnots, rangeTy2[i])
            Bt.nzval .= view(vec(BtCompact'), BtOrder)
            wy2 = vec( cholesky(Qt).U \ randn(nKnots, nPredUnq))

            ypt += swy2[i]*(Bt*wy2)

        else

            ypt += swy2[i]*rand(nPred)

        end

        yp = ratpwr.(ypt, pwr)

        # Combine, project, and update mean and variance

        predSamps[i,:] = Projection*(yp.*z)


    end

    return predSamps



end
