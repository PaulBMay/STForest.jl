# Polya Gamma Sampler, convenience wrapper from PolyaGammaSamplers.jl
function rpg(z)
    return rand(PolyaGammaPSWSampler(1, z), 1)[1]
end

# Conditional samples of the Gaussian latent effects. Updates the factorization 'Qpc' with the cholesky of 'Qp', assuming the same sparsity structure.
function getGaussSamp!(Qpc::SparseArrays.CHOLMOD.Factor, Qp::SparseMatrixCSC, yProj::Vector{Float64})

    cholesky!(Qpc, Hermitian(Qp))

    samp = Qpc \ yProj

    error = (Qpc.U \ randn(length(yProj)))

    samp += @view error[invperm(Qpc.p)]

    return samp


end

# Same as above, but assumes the appropriate 'Qpc' was computed in the outer scope.
function getGaussSamp(Qpc::SparseArrays.CHOLMOD.Factor, yProj::Vector{Float64})

    samp = Qpc \ yProj

    error = (Qpc.U \ randn(length(yProj)))

    samp += @view error[invperm(Qpc.p)]

    return samp

end