using TeneT
using Random
using LinearAlgebra
using OptimKit
using Printf

seed = 42
Random.seed!(seed)

atype = Array
etype = Float64
D = parse(Int, get(ENV, "TENET_BENCHMARK_D", "4"))
chi = parse(Int, get(ENV, "TENET_BENCHMARK_CHI", "32"))
chishift = 0
pattern = [1;;]
ctm_maxiter = parse(Int, get(ENV, "TENET_BENCHMARK_CTM_MAXITER", "50"))
ctm_miniter = parse(Int, get(ENV, "TENET_BENCHMARK_CTM_MINITER", "3"))
ctm_maxiter_ad = parse(Int, get(ENV, "TENET_BENCHMARK_CTM_MAXITER_AD", "4"))
ctm_miniter_ad = parse(Int, get(ENV, "TENET_BENCHMARK_CTM_MINITER_AD", "1"))
ctm_show_every = parse(Int, get(ENV, "TENET_BENCHMARK_CTM_SHOW_EVERY", "5"))
ctm_tol = parse(Float64, get(ENV, "TENET_BENCHMARK_CTM_TOL", "1e-8"))
opt_maxiter = parse(Int, get(ENV, "TENET_BENCHMARK_OPT_MAXITER", "1"))
opt_gradtol = parse(Float64, get(ENV, "TENET_BENCHMARK_OPT_GRADTOL", "1e-7"))

model = Heisenberg(lattice=Honeycomb(:c3v),
                   S=0.5, Jx=1.0, Jy=1.0, Jz=1.0,
                   ifrotate=true,
                   couplingtype=:uniform,
                   bondratio=1.0)

boundary_alg = C3vQRCTMRG(; maxiter=ctm_maxiter,
                            miniter=ctm_miniter,
                            maxiter_ad=ctm_maxiter_ad,
                            miniter_ad=ctm_miniter_ad,
                            show_every=ctm_show_every,
                            tol=ctm_tol,
                            verbosity=3)

params = GradientOptimize(model=model,
                          pattern=pattern,
                          boundary_alg=boundary_alg,
                          optimizer=LBFGS(200; maxiter=opt_maxiter, verbosity=3, gradtol=opt_gradtol),
                          maxiter_restart=1,
                          verbosity=3,
                          folder=joinpath(pkgdir(TeneT), "data/benchmark_heisenberg_c3v_D4_chi32"),
                          ifSU=false,
                          SUτ=0,
                          ifprecondition=false,
                          reuse_env=true,
                          ifsave_env=false,
                          ifload_env=false,
                          ifsave_lbfgs=false,
                          ifload_lbfgs=false,
                          ifplot=false)

function restriction_ipeps(A)
    return C3vHeisenberg_restriction(A)
end

println("=== honeycomb Heisenberg C3v QRCTMRG benchmark ===")
println("seed=$seed atype=$atype etype=$etype D=$D chi=$chi")
println("julia_threads=$(Threads.nthreads()) blas_threads=$(BLAS.get_num_threads())")
println("boundary maxiter=$(boundary_alg.maxiter) tol=$(boundary_alg.tol)")

t0 = time()
A = init_ipeps(; atype, etype, No=0, D, χ=chi, params)
t_init = time() - t0
println(@sprintf("init_ipeps_time = %.3f s", t_init))

A = restriction_ipeps(A)
t1 = time()
A_struct = TeneT.build_A(A, params)
rt = TeneT.initialize_env(A, D, chi, params)
rt, err = TeneT.leading_boundary(rt, A_struct, boundary_alg)
env = TeneT.ObsEnv(rt, A_struct, boundary_alg)
energy, e_dict = TeneT.energy_value(model, A_struct, env, params)
t_boundary = time() - t1

println(@sprintf("single_boundary_energy_time = %.3f s", t_boundary))
println(@sprintf("boundary_err = %.6e", real(err)))
println(@sprintf("energy = %.12f", real(energy)))
println("bond_energy = ", e_dict["bond_Heisenberg_C3v_energy"]["1,1"])

run_optimize = get(ENV, "TENET_BENCHMARK_OPTIMIZE", "false") == "true"
if run_optimize
    t2 = time()
    A_opt, e_opt, g_opt, fgnum, history = optimise_ipeps(A, chi, chishift, params; restriction_ipeps)
    t_opt = time() - t2

    println(@sprintf("lbfgs_time = %.3f s", t_opt))
    println(@sprintf("optimized_energy = %.12f", real(e_opt)))
    println(@sprintf("optimized_grad_norm = %.6e", norm(g_opt)))
    println("optimizer_fg_evals = ", fgnum)
    println("optimizer_history = ", history)
else
    println("one_lbfgs_iteration_time = skipped (set TENET_BENCHMARK_OPTIMIZE=true to run)")
end
println(@sprintf("total_time = %.3f s", time() - t0))
