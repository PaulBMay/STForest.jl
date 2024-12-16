
function STvariogram(y::Vector, loc::Matrix, time::Matrix, spaceIntervals::Matrix, timeIntervals::Matrix)

    n = size(loc, 1)

    nsi = size(spaceIntervals, 1)
    nti = size(timeIntervals, 1)

    vbin = zeros(nsi, nti)
    nbin = zeros(Int64, nsi, nti)

    maxSpace = maximum(spaceIntervals)
    maxTime = maximum(timeIntervals)

    for i in 1:(n-1)

        v = (y[(i+1):n] .- y[i]).^2 ./ 2

        ds = sqrt.( sum( (loc[(i+1):n, :] .- loc[[i],:]).^2, dims = 2 ) )
        dt = sqrt.( sum( (time[(i+1):n, :] .- time[[i],:]).^2, dims = 2 ) )

        inBounds = (ds .< maxSpace) .&& (dt .< maxTime)

        for j in 1:(n-i)

            if inBounds[j]

                spacebin = findfirst((ds[j] .> spaceIntervals[:,1]) .&& (ds[j] .< spaceIntervals[:,2])) 
                timebin = findfirst((dt[j] .> timeIntervals[:,1]) .&& (dt[j] .<= timeIntervals[:,2]))

                if !isnothing(spacebin) && !isnothing(timebin)

                    vbin[spacebin, timebin] += v[j]
                    nbin[spacebin, timebin] += 1  
                    
                end

            end

        end

    end


    return vbin ./ max.(nbin, 1), nbin


end

function STvariogramTheory(spaceIntervals::Matrix, timeIntervals::Matrix, covFunc::Function)

    spaceSeq = mean(spaceIntervals, dims = 2)
    timeSeq = mean(timeIntervals, dims = 2)

    nsi, nti = length(spaceSeq), length(timeSeq)

    vbin = zeros(nsi, nti)

    s2 = covFunc(0, 0)

    for i in 1:nsi
        for j in 1:nti
            vbin[i,j] = s2 - covFunc(spaceSeq[i], timeSeq[j])
        end
    end

    return vbin

end

function STcovariogramTheory(spaceIntervals::Matrix, timeIntervals::Matrix, covFunc::Function)

    spaceSeq = mean(spaceIntervals, dims = 2)
    timeSeq = mean(timeIntervals, dims = 2)

    nsi, nti = length(spaceSeq), length(timeSeq)

    vbin = zeros(nsi, nti)

    for i in 1:nsi
        for j in 1:nti
            vbin[i,j] = covFunc(spaceSeq[i], timeSeq[j])
        end
    end

    return vbin

end


