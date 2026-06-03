using TeneT
using Random
using OptimKit
using LinearAlgebra

seed = 42
Random.seed!(seed)
atype = Array
etype = Float64
D, chi, chishift, maxiter_restart = 2, 8, 0, 1

# Two-site C3v honeycomb unit cell: site 1 is the active J2p sublattice.
pattern = [1 2]

model = J1J2p(lattice=Honeycomb(:c3v),
              S=0.5, J1=1.0, J2p=0.5,
              ifrotate=false,
              couplingtype=:uniform, bondratio=1.0)
No = 0
folder = joinpath(pkgdir(TeneT), "data/$model/$pattern/C3vQRCTMRG/$etype/seed$seed/")

boundary_alg = C3vQRCTMRG(; maxiter=20,
                            miniter=0,
                            maxiter_ad=4,
                            miniter_ad=4,
                            show_every=5,
                            tol=1e-10,
                            verbosity=3)

params = GradientOptimize(model=model,
                          pattern=pattern,
                          boundary_alg=boundary_alg,
                          optimizer=LBFGS(200; maxiter=5, verbosity=4, gradtol=1e-7,
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

A = init_ipeps(; atype, etype, No, D, χ=chi, params)

function restriction_ipeps(A)
    return C3vHeisenberg_restriction(A)
end

optimise_ipeps(A, chi, chishift, params; restriction_ipeps)
