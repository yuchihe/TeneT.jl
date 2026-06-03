using TeneT
using Random
using OptimKit
using LinearAlgebra
using Printf

seed = parse(Int, get(ENV, "TENET_C3V_SEED", "42"))
Random.seed!(seed)

atype = Array
etype = Float64
D = parse(Int, get(ENV, "TENET_C3V_D", "2"))
chi = parse(Int, get(ENV, "TENET_C3V_CHI", "8"))
chishift = 0
maxiter_restart = 1
lbfgs_maxiter = parse(Int, get(ENV, "TENET_C3V_LBFGS_MAXITER", "5"))
boundary_maxiter = parse(Int, get(ENV, "TENET_C3V_BOUNDARY_MAXITER", "20"))
boundary_maxiter_ad = parse(Int, get(ENV, "TENET_C3V_BOUNDARY_MAXITER_AD", "4"))
ifrotate = parse(Bool, get(ENV, "TENET_C3V_IFROTATE", "true"))

pattern = [1 2]

model = J1J2p(lattice=Honeycomb(:c3v),
              S=0.5, J1=1.0, J2p=0.0,
              ifrotate=ifrotate,
              couplingtype=:uniform, bondratio=1.0)

folder = joinpath(pkgdir(TeneT), "data/$model/$pattern/C3vQRCTMRG/$etype/seed$seed/")

boundary_alg = C3vQRCTMRG(; maxiter=boundary_maxiter,
                            miniter=0,
                            maxiter_ad=boundary_maxiter_ad,
                            miniter_ad=boundary_maxiter_ad,
                            show_every=5,
                            tol=1e-10,
                            verbosity=3)

params = GradientOptimize(model=model,
                          pattern=pattern,
                          boundary_alg=boundary_alg,
                          optimizer=LBFGS(200; maxiter=lbfgs_maxiter,
                                           verbosity=4,
                                           gradtol=1e-7,
                                           linesearch=HagerZhangLineSearch(maxfg=5)),
                          maxiter_restart=maxiter_restart,
                          verbosity=4,
                          folder=folder,
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

println("=== J1J2p Honeycomb C3v QRCTMRG two-site optimization ===")
println("model = $model")
println("pattern = $pattern")
println("D = $D, chi = $chi, seed = $seed")
println("boundary_maxiter = $boundary_maxiter, boundary_maxiter_ad = $boundary_maxiter_ad")
println("lbfgs_maxiter = $lbfgs_maxiter")
flush(stdout)

A0 = init_ipeps(; atype, etype, No=0, D, χ=chi, params)
t0 = time()
Aopt, e, eg, fgnum, history = optimise_ipeps(A0, chi, chishift, params; restriction_ipeps)
elapsed = time() - t0

println(@sprintf("RESULT energy=%.15f gnorm=%.6e fgnum=%d elapsed=%.3f", real(e), norm(eg), fgnum, elapsed))
println("history = ", history)
