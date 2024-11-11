


function dataSimulation(n::Integer, nUnq::Integer, nKnots::Integer, Lspace::Number, Ltime::Number, params::NamedTuple; m = 25)

    local locCand = Lspace*rand(nUnq, 2)
    local loc = locCand[sample(1:nCand, n),:]
    local locUnq = unique(loc, dims = 1)
    nUnq = size(locUnq,1)

    local time = Ltime*rand(n,1)

    local map2unq = indexin(loc[:,1],locUnq[:,1])
    local P = sparse(1:n, map2unq, 1)

    local htime = Ltime/(nKnots - 1)
    local timeKnots = reshape(collect(0:htime:Ltime),:,1)

    # Simulate z

    local nbz = getNeighbors(locUnq, m)
    local Bz, Fz, BzOrder = getNNGPmatsS(nbz, locUnq, rangeSz)
    local gzmu = betaz .+ swz * ( LowerTriangular(Bz) \ (sqrt.(Fz) .* randn(nUnq)) )
    local zmu = 1 ./ (1 .+ exp.(-gzmu))
    local z = 1*(zmu .> rand(nUnq))
    local zf = P*z

    # Simulate y

    local nby = getNeighbors(loc, m)
    local By, Fy, ByOrder = getNNGPmatsST(nby, loc, time, rangeSy1, rangeTy1)
    local wy1f = (LowerTriangular(By) \ (sqrt.(Fy) .* randn(n) ))

    local Qt = expCor(timeKnots, timeKnots, rangeTy2)
    local wy2f = vec( cholesky(Qt).U \ randn(nKnots, nUnq))
    local BtRows, BtCols, BtOrder = getBtNZ(time, map2unq, nKnots)
    local BtCompact = expCor(time, timeKnots, rangeTy2)
    local Bt = sparse(BtRows, BtCols, view(vec(BtCompact'), BtOrder))

    local yp = betay .+ wy1f + swy2*Bt*w2f + sqrt(t2y)*randn(n)

    local y = yp.*zf

    local X = ones(n,1)

    return InputData(y, X, loc, time)

end

function NNGP_ZIST(data::InputData, initparams::NamedTuple, spriors::SpatialPriors, sp::SpatialParams, mcmc::McmcParams, outDir::String, nSamps::Int64; writew2 = false)

     
    ############################
    # Data dimensions and prep
    ##########################

    n = size(data.y, 1) # sample size
    p = size(data.X, 2) # num predictors
    nKnots = size(sp.timeKnots, 1)

    ##########################################################
    # I am making the forest/non-forest discrims time static.
    # I also have differing random errors across space (expected to be larger) than time (expected to be smaller)
    ##########################################################

    # Unique plot locations
    locUnq = unique(data.loc, dims = 1)
    nUnq = size(locUnq, 1)

    # For each location in the full stack, what's it's position among the set of unique locations?
    map2unq = indexin(loc2str(data.loc), loc2str(locUnq))

    # Organize this is a sparse matrix. Entry (i,j) is one if row i is associated with unique location j
    P = sparse(1:n, map2unq, 1)

    # Forest binary for the unique locations. Considered "forest" if any measurement at any time had positive AGBD
    z = (transpose(P)*(data.y .> 0)) .> 0

    Xz = (transpose(P)*data.X) ./ sum(P, dims = 1)[1,:]

    # Back-trace this. Now a zero AGBD plot could be considered "forest" if a measurement at another time had positive AGBD
    zf = (P*z) .> 0

    np = sum(zf)
    npUnq = sum(z)
    kt = npUnq*nKnots


    # Isolate "forest" values. 
    yp = copy(data.y[zf]) 
    Xp = copy(data.X[zf,:])
    locp = copy(data.loc[zf,:])
    timep = copy(data.time[zf,:])

    # We'll need a projection to unique locations for the positive AGBD plots, too

    map2unqp = @views indexin(loc2str(locp), loc2str(locUnq[z,:]))


    ##########################
    # Initial values and CSV creation
    ##########################

    # Does the out_dir exist?

    if !isdir(outDir)
        error("Can't find your outDir")
    end

    # Prepare CSV's

    locOut = joinpath(outDir, "loc_unq.csv")
    CSV.write(locOut, DataFrame(lon = locUnq[:,1], lat = locUnq[:,2]), forest = z)

    loctimepOut = joinpath(outDir, "loctime_pos.csv")
    CSV.write(loctimepOut, DataFrame(lon = locp[:,1], lat = loc[zf,2], time = time[zf,1]))

    knotsOut = joinpath(outDir, "time_knots.csv")
    CSV.write(knotsOut, DataFrame(time = sp.timeKnots[:,1]))

    paramsOut = joinpath(outDir, "params.csv")
    paramsDf = DataFrame(zeros(1, 2*p + 8), 
        ["betay_".*string.(0:(p-1)); "betaz_".*string.(0:(p-1)); ["swy1", "rangeSy1", "rangeTy1", "swy2", "rangeTy2", "t2y", "swz", "rangeSz"]]
        )

    wy1Out = joinpath(outDir, "wy1.csv")
    wy1Df = DataFrame(zeros(1,np), "wy1_".*string.(1:np))

    if writew2
        wy2Out = joinpath(outDir, "wy2.csv")
        wy2Df = DataFrame(zeros(1, kt), "wy2_l" .* string.(repeat(1:npUnq, inner = nKnots)) .* "_k" .* string.(repeat(1:nKnots, npUnq)) )
    end

    wzOut = joinpath(outDir, "wz.csv")
    wzDf = DataFrame(zeros(1,nUnq), "wz_".*string.(1:nUnq))

    

    # Parameter/effect values

    sampy = zeros(p + kt + np)
    betay = sampy[1:p]
    wy1 = sampy[(p+kt+1):(p+kt+np)]
    wy2 = sampy[(p+1):(p+kt)]

    sampz = zeros(p + nUnq)
    wz = view(sampz, (p+1):(p+nUnq))
    betaz = view(sampz, 1:p)

    swy1, rangeSy1, rangeTy1, swy2, rangeTy2, t2y, swz, rangeSz = initparams.swy1, initparams.rangeSy1, initparams.rangeTy1, initparams.swy2, initparams.rangeTy2, initparams.t2y, initparams.swz, initparams.rangeSz
    swy1p, rangeSy1p, rangeTy1p, swy2p, rangeTy2p, t2yp = swy1, rangeSy1, rangeTy1, swy2, rangeTy2, t2y


    paramsDf[1,:] = [betay; betaz; [swy1, rangeSy1, rangeTy1, swy2, rangeTy2, t2y, swz, rangeSz]]
    wy1Df[1,:] = wy1
    if writew2; wy2Df[1,:] = wy2 end
    wzDf[1,:] = wz

    CSV.write(paramsOut, paramsDf)
    CSV.write(wy1Out, wy1Df)
    if writew2; CSV.write(wy2Out, wy2Df) end
    CSV.write(wzOut, wzDf)



    ####################
    # Lord have mercy that was boring.
    # Now fun stuff. Get the neighbor sets and initial NNGP mats
    #####################


    print("Getting neighbor sets\n")

    nby = getNeighbors(locp, sp.m)
    nbz = getNeighbors(locUnq, sp.m)

    print("Initial NNGP mats\n")

    By,Fy,ByOrder = getNNGPmatsST(nby, locp, timep, rangeSy1, rangeTy1)
    #Byp,Fyp = copy(By), copy(Fy)

    Bz,Fz,BzOrder = getNNGPmatsS(nbz, locUnq, rangeSz)
    Bzp,Fzp = copy(Bz), copy(Fz)

    BtRows, BtCols, BtOrder = getBtNZ(np, map2unqp, nKnots)
    BtCompact = expCor(timep, sp.timeKnots, rangeTy2)
    Bt = sparse(BtRows, BtCols, view(vec(BtCompact'), BtOrder))
    Qt =expCor(sp.timeKnots, rangeTy2)
    #BtCompactp, Btp, Qtp = copy(BtCompact), copy(Bt), copy(Qt)

    Dsgny = sparse_hcat(Xp, swy2^2*Bt, speye(np)) 
    Dsgnz = sparse_hcat(Xz, speye(nUnq))

    yProj = Dsgny'*(yp ./ t2y)
    zProj = Dsgnz'*(z .- 0.5)

    pg = rpg.(fill(0.3, nUnq))

    byPrec = fill(0.01, p)
    bzPrec = fill(0.5, p)


    ##############

    Qy = blockdiag(
        spdiagm(byPrec),
        kron(speye(npUnq), swy2^2*Qt),
        (1/swy1^2)*By'*spdiagm(1 ./ Fy)*By
    ) + (1/t2y)*Dsgny'*Dsgny

    Qyc = cholesky(Hermitian(Qy))
    Qypc = copy(Qyc)

    currentThetay = log.([swy1, rangeSy1, rangeTy1, swy2, rangeTy2, t2y])
    propThetay = copy(currentThetay)
    lp = thetayLP(currentThetay, spriors, yp, byPrec, Fy, Qt, Dsgny, Qyc)
    lpProp = lp
    acceptThetay = 0

    propy_chol = cholesky(mcmc.thetayVar).L

    lpDf = DataFrame(lp = lp, lpProp = lpProp, accept = acceptThetay*1)
    lpOut = joinpath(outDir, "lp.csv")
    CSV.write(lpOut, lpDf)

    ##################

    Qz = blockdiag(
        spdiagm(bzPrec),
        (1/swz^2)*Bz'*spdiagm(1 ./ Fz)*Bz
    ) + Dsgnz'*spdiagm(pg)*Dsgnz

    Qzc = cholesky(Hermitian(Qz))

    currentThetaz = log.([swz, rangeSz])
    propThetaz = copy(currentThetaz)
    acceptThetaz = 0

    propz_chol = cholesky(mcmc.thetazVar).L

    #########################
    # Begin Gibbs sampler
    #########################

    for i = ProgressBar(1:nSamps)

       ########################
       # Sample betay, wy
       ########################

       sampy = getGaussSamp(Qyc, yProj)


       ############################
       # Sample betaz, wz
       ############################

       Qz .= blockdiag(
            spdiagm(bzPrec),
            (1/swz^2)*Bz'*spdiagm(1 ./ Fz)*Bz
        ) + Dsgnz'*spdiagm(pg)*Dsgnz

       sampz .= getGaussSamp!(Qzc, Qz, zProj)

       wz .= sampz[(p+1):(p+nUnq)]
       betaz .= sampz[1:p]

       #######################
       # Sample pg
       #######################

       pg .= rpg.(Dsgnz*sampz)

       ###########################
       # Sample all spatial parameters associated with y
       ###########################

       lpDf.lp[1] = lp

       #= println("###################### \n")
       println("Sample "*string(i)*"\n")
       println("The current lp is "*string(lp)*"\n")
       println("The current thetay is "*string(currentThetay)*"\n") =#

       # Get proposal values from a log-normal distribution
       propThetay = currentThetay + propy_chol*randn(6)

       swy1p, rangeSy1p, rangeTy1p, swy2p, rangeTy2p, t2yp = exp.(propThetay)

       # Get NNGP and fixed-rank matrices associated with the proposal values
       getNNGPmatsST!(By, Fy, ByOrder, nby, locp, timep, rangeSy1p, rangeTy1p)
       expCor!(Qt, sp.timeKnots, rangeTy2p)
       expCor!(BtCompact, timep, sp.timeKnots, rangeTy2p)
       Bt.nzval .= view(vec(BtCompact'), BtOrder)
       # Posterior precision for the proposal values
       Dsgny .= sparse_hcat(Xp, swy2p^2*Bt, speye(np))
       Qy .= blockdiag(
            spdiagm(byPrec),
            kron(speye(npUnq), swy2p^2*Qt),
            (1/swy1p^2)*By'*spdiagm(1 ./ Fy)*By
       ) + (1/t2yp)*Dsgny'*Dsgny

       cholesky!(Qypc, Hermitian(Qy))

       lpProp = thetayLP(propThetay, spriors, yp, byPrec, Fy, Qt, Dsgny, Qypc)

       #println("The proposed lp is "*string(lpProp)*"\n")

       #println("Did my og lp change?? "*string(lp)*"\n")


       acceptProby = exp.(lpProp + sum(propThetay)  - lp - sum(currentThetay))

       #println("accepting with probability "*string(acceptProby)*"\n")

       acceptThetay = rand(1)[1] < acceptProby

       if acceptThetay
        swy1, rangeSy1, rangeTy1, swy2, rangeTy2, t2y = exp.(propThetay)
        Qyc = copy(Qypc)
        lp = lpProp
        yProj .= Dsgny'*(yp ./ t2y)
        currentThetay .= copy(propThetay)
       end

       lpDf.lpProp[1] = lpProp

       lpDf.accept[1] = acceptThetay*1


       ###########################
       # Sample swz, rangeSz
       ###########################

       propThetaz = currentThetaz + propz_chol*randn(2)

       swzp, rangeSzp = exp.(propThetaz)

       getNNGPmatsS!(Bzp, Fzp, BzOrder, nbz, locUnq, rangeSzp)

       llProp = wll(Bzp, Fzp, swzp^2, wz)
       ll = wll(Bz, Fz, swz^2, wz)

       priorProp = pcprior([swzp, rangeSzp], spriors.thetaz0, spriors.alphaz0)
       prior = pcprior([swz, rangeSz], spriors.thetaz0, spriors.alphaz0)

       acceptProbz = exp.(llProp + priorProp + sum(propThetaz) - ll - prior - sum(currentThetaz))

       acceptThetaz = rand(1)[1] < acceptProbz

       if acceptThetaz
            swz, rangeSz = swzp, rangeSzp
            Bz.nzval .= Bzp.nzval
            Fz .= Fzp
       end



       #########################
       # Good work, team. Let's write out these results
       #########################

       paramsDf[1,:] = [sampy[1:p]; betaz; [swy1, rangeSy1, rangeTy1, swy2, rangeTy2, t2y, swz, rangeSz]]
       wy1Df[1,:] = sampy[(1+p+kt):end]
       wzDf[1,:] = wz

       CSV.write(paramsOut, paramsDf; append = true, header = false)
       CSV.write(wy1Out, wy1Df; append = true, header = false)
       CSV.write(wzOut, wzDf; append = true, header = false)
       CSV.write(lpOut, lpDf; append = true, header = false)

       if writew2
        wy2Df[1,:] = wy2
        CSV.write(wy2Out, wy2Df; append = true, header = false)
       end



    end



    return nothing


end


