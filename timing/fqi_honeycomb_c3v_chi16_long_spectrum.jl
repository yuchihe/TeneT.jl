using LinearAlgebra
using Printf

include("fqi_honeycomb_c3v_qrctmrg_chis.jl")

function parse_checkpoints(maxiter)
    s = get(ENV, "FQI_C3V_LONG_CHECKPOINTS", "600,1000,1600,2500,5000,7500,10000")
    cps = parse.(Int, split(s, ","))
    return sort(unique([c for c in cps if 0 < c <= maxiter]))
end

function run_long_chi16()
    chi = parse(Int, get(ENV, "FQI_C3V_LONG_CHI", "16"))
    maxiter = parse(Int, get(ENV, "FQI_C3V_LONG_MAXITER", "10000"))
    checkpoints = parse_checkpoints(maxiter)
    BLAS.set_num_threads(parse(Int, get(ENV, "FQI_BLAS_THREADS", "2")))

    Tu_single, Tv_single = fqi_c3v_sublattice_tensors(0.9, 0.1; etype=Float64)
    Tu_single ./= sqrt(norm(Tu_single))
    Tv_single ./= sqrt(norm(Tv_single))
    Tu = double_layer3(Tu_single)
    Tv = double_layer3(Tv_single)

    env = single_site_seed_env(Tu_single, chi)
    println("=== FQI C3v long transfer-spectrum run ===")
    println("chi = $chi, maxiter = $maxiter")
    println("checkpoints = $checkpoints")
    println("julia_threads = $(Threads.nthreads()), blas_threads = $(BLAS.get_num_threads())")
    flush(stdout)

    t0 = time()
    prev_mu = nothing
    corner_delta = Inf
    for iter in 1:maxiter
        env, corner_delta = c3v_dl_step(env, Tu, Tv)
        if iter in checkpoints
            spec = transfer_spectrum(env; nev=8)
            cspec = corner_spectrum(env.C)
            rank12 = count(>(1e-12), cspec)
            rank14 = count(>(1e-14), cspec)
            mu_head = spec.mu[1:8]
            drift = prev_mu === nothing ? Inf : norm(mu_head .- prev_mu)
            prev_mu = copy(mu_head)
            println(@sprintf("CHECK iter=%d xi=%.15f gap=%.15e mu_drift=%.3e corner_delta=%.3e rank12=%d rank14=%d elapsed=%.2f",
                             iter, spec.xi, spec.gap[spec.lambda2_index], drift,
                             real(corner_delta), rank12, rank14, time() - t0))
            println("  mu = ", join(map(x -> @sprintf("%.15e", x), mu_head), ", "))
            println("  gap = ", join(map(x -> @sprintf("%.15e", x), spec.gap[1:8]), ", "))
            println("  xi_modes = ", join(map(x -> isfinite(x) ? @sprintf("%.15e", x) : "Inf", spec.xi_modes[1:8]), ", "))
            println("  corner_svd_over_s1 = ", join(map(x -> @sprintf("%.15e", x / cspec[1]), cspec[1:8]), ", "))
            flush(stdout)
        end
    end
end

run_long_chi16()
