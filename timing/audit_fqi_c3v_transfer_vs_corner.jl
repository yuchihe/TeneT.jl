using LinearAlgebra
using Printf

include("fqi_honeycomb_c3v_qrctmrg_chis.jl")

function dense_edge_transfer_matrix(R; conjugate_boundary::Bool=false)
    chi = size(R, 1)
    T = Matrix{eltype(R)}(undef, chi^2, chi^2)
    E = zeros(eltype(R), chi, chi)
    col = 1
    for j in 1:chi, i in 1:chi
        fill!(E, zero(eltype(R)))
        E[i, j] = one(eltype(R))
        T[:, col] .= vec(edge_transfer(E, R; conjugate_boundary))
        col += 1
    end
    return T
end

function run_audit()
    chi = parse(Int, get(ENV, "FQI_C3V_AUDIT_CHI", "16"))
    BLAS.set_num_threads(parse(Int, get(ENV, "FQI_BLAS_THREADS", "2")))

    Tu_single, Tv_single = fqi_c3v_sublattice_tensors(0.9, 0.1; etype=Float64)
    Tu_single ./= sqrt(norm(Tu_single))
    Tv_single ./= sqrt(norm(Tv_single))
    Tu = double_layer3(Tu_single)
    Tv = double_layer3(Tv_single)

    env = single_site_seed_env(Tu_single, chi)
    env, err, iter, xi, xi_err, ratio_err = leading_boundary_dl(
        env, Tu, Tv;
        tol=1e-10, maxiter=2600, miniter=1000,
        show_every=100, verbosity=0,
        xi_check_every=100, stable_checks=3,
    )

    spec = transfer_spectrum(env)
    Tdense = dense_edge_transfer_matrix(env.R)
    dense_vals = eigvals(Tdense)
    dense_vals = dense_vals[sortperm(abs.(dense_vals); rev=true)]
    dense_mu = abs.(dense_vals ./ dense_vals[1])

    cspec = corner_spectrum(env.C)
    corner_mu_like = cspec ./ cspec[1]

    println(@sprintf("AUDIT chi=%d iter=%d err=%.12e xi=%.12f", chi, iter, err, spec.xi))
    println("corner_svd_norm2 = ", join(map(x -> @sprintf("%.12e", x), cspec[1:8]), ", "))
    println("corner_svd_over_s1 = ", join(map(x -> @sprintf("%.12e", x), corner_mu_like[1:8]), ", "))
    println("krylov_transfer_mu = ", join(map(x -> @sprintf("%.12e", x), spec.mu[1:8]), ", "))
    println("dense_transfer_mu = ", join(map(x -> @sprintf("%.12e", x), dense_mu[1:8]), ", "))
    println(@sprintf("dense_vs_krylov_head_err=%.12e", norm(dense_mu[1:8] .- spec.mu[1:8])))
    println(@sprintf("corner_vs_transfer_head_err=%.12e", norm(corner_mu_like[1:8] .- spec.mu[1:8])))
end

run_audit()
