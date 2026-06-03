using LinearAlgebra
using Printf
using Random

include("fqi_honeycomb_c3v_qrctmrg_chis.jl")

function parse_seed_list()
    s = get(ENV, "FQI_C3V_SEEDS", "tensor,1,2,3,4")
    return strip.(split(s, ","))
end

function random_seed_env(Tu_single, chi::Int, seed::Int, mode::Symbol)
    rng = MersenneTwister(seed)
    Q = size(Tu_single, 1)^2
    C = zeros(eltype(Tu_single), chi, chi)
    R = zeros(eltype(Tu_single), chi, Q, chi)
    if mode == :full
        C .= randn(rng, eltype(Tu_single), chi, chi)
        R .= randn(rng, eltype(Tu_single), chi, Q, chi)
    elseif mode == :support
        n = min(chi, Q)
        C[1:n, 1:n] .= randn(rng, eltype(Tu_single), n, n)
        R[1:n, :, 1:n] .= randn(rng, eltype(Tu_single), n, Q, n)
    else
        throw(ArgumentError("unknown FQI_C3V_SEED_MODE=$mode (use full or support)"))
    end
    C ./= norm(C)
    R ./= norm(R)
    return C3vDLEnv(C, R)
end

function make_seed_env(Tu_single, chi::Int, seed_label, mode::Symbol)
    seed_label == "tensor" && return single_site_seed_env(Tu_single, chi)
    return random_seed_env(Tu_single, chi, parse(Int, seed_label), mode)
end

function run_seed_robustness()
    chi = parse(Int, get(ENV, "FQI_C3V_SEED_CHI", "32"))
    maxiter = parse(Int, get(ENV, "FQI_C3V_SEED_MAXITER", "10000"))
    miniter = parse(Int, get(ENV, "FQI_C3V_SEED_MINITER", "1000"))
    tol = parse(Float64, get(ENV, "FQI_C3V_SEED_TOL", "1e-10"))
    show_every = parse(Int, get(ENV, "FQI_C3V_SEED_SHOW_EVERY", "500"))
    xi_check_every = parse(Int, get(ENV, "FQI_C3V_SEED_XI_CHECK_EVERY", "100"))
    stable_checks = parse(Int, get(ENV, "FQI_C3V_SEED_STABLE_CHECKS", "3"))
    mode = Symbol(get(ENV, "FQI_C3V_SEED_MODE", "full"))
    seeds = parse_seed_list()

    BLAS.set_num_threads(parse(Int, get(ENV, "FQI_BLAS_THREADS", "2")))

    Tu_single, Tv_single = fqi_c3v_sublattice_tensors(0.9, 0.1; etype=Float64)
    Tu_single ./= sqrt(norm(Tu_single))
    Tv_single ./= sqrt(norm(Tv_single))
    Tu = double_layer3(Tu_single)
    Tv = double_layer3(Tv_single)

    println("=== FQI C3v seed robustness ===")
    println("chi = $chi, seeds = $seeds, random_mode = $mode")
    println("maxiter = $maxiter, miniter = $miniter, tol = $tol")
    println("xi_check_every = $xi_check_every, stable_checks = $stable_checks")
    println("julia_threads = $(Threads.nthreads()), blas_threads = $(BLAS.get_num_threads())")
    flush(stdout)

    xis = Float64[]
    mus = Vector{Vector{Float64}}()
    labels = String[]

    for seed_label in seeds
        println("\n--- seed = $seed_label ---")
        flush(stdout)
        env0 = make_seed_env(Tu_single, chi, seed_label, mode)
        env, err, iter, xi, xi_err, ratio_err = leading_boundary_dl(
            env0, Tu, Tv;
            tol, maxiter, miniter,
            show_every, verbosity=2,
            xi_check_every, stable_checks,
        )
        spec = transfer_spectrum(env; nev=8)
        cspec = corner_spectrum(env.C)
        rank12 = count(>(1e-12), cspec)
        rank14 = count(>(1e-14), cspec)
        mu_head = collect(spec.mu[1:8])
        push!(xis, spec.xi)
        push!(mus, mu_head)
        push!(labels, seed_label)

        println(@sprintf("SEED_RESULT seed=%s iter=%d err=%.12e xi=%.15f gap=%.15e xi_err=%.12e ratio_err=%.12e rank12=%d rank14=%d",
                         seed_label, iter, err, spec.xi, spec.gap[spec.lambda2_index],
                         xi_err, ratio_err, rank12, rank14))
        println("  mu = ", join(map(x -> @sprintf("%.15e", x), mu_head), ", "))
        println("  xi_modes = ", join(map(x -> isfinite(x) ? @sprintf("%.15e", x) : "Inf", spec.xi_modes[1:8]), ", "))
        println("  corner_svd_over_s1 = ", join(map(x -> @sprintf("%.15e", x / cspec[1]), cspec[1:8]), ", "))
        flush(stdout)
    end

    ref_mu = mus[1]
    ref_xi = xis[1]
    println("\n=== seed comparison to $(labels[1]) ===")
    for i in eachindex(labels)
        println(@sprintf("SEED_COMPARE seed=%s xi_delta=%.12e mu_delta=%.12e",
                         labels[i], abs(xis[i] - ref_xi), norm(mus[i] .- ref_mu)))
    end
    println(@sprintf("SEED_SPREAD xi_min=%.15f xi_max=%.15f xi_spread=%.12e",
                     minimum(xis), maximum(xis), maximum(xis) - minimum(xis)))
end

run_seed_robustness()
