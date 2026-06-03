function energy_value(model::Kitaev{Honeycomb{:c3v}}, A, env::Union{C3vCTMEnv,C3vTwoSiteCTMEnv}, params::iPEPSOptimize)
    (model.Jx == model.Jy == model.Jz) ||
        throw(ArgumentError("Kitaev{Honeycomb{:c3v}} with C3vQRCTMRG assumes isotropic couplings Jx=Jy=Jz."))

    A1, A2 = _c3v_site_pair(A)
    atype = _arraytype(A1)
    Sz = atype(const_Sz(model.S))

    terms = ((one(model.Jx), Sz, Sz),)
    if env isa C3vTwoSiteCTMEnv && length(A.data) == 2
        Ez_AB, _ = _c3v_two_site_bond(env, A1, A2, terms; site=1)
        Ez_BA, _ = _c3v_two_site_bond(env, A2, A1, terms; site=2)
        Ez = (Ez_AB + Ez_BA) / 2
        e_dict = Dict{String, Dict{String, Any}}(
            "bond_Kitaev_C3v_energy" => Dict{String, Any}(
                "1,2" => Ez_AB,
                "2,1" => Ez_BA,
            ),
        )
    else
        Ez, _ = _c3v_two_site_bond(env, A1, A2, terms)
        e_dict = Dict{String, Dict{String, Any}}(
            "bond_Kitaev_C3v_energy" => Dict{String, Any}("1,1" => Ez),
        )
    end
    e = 0.5 * (model.Jx + model.Jy + model.Jz) * real(Ez)

    params.verbosity >= 3 && println("energy = $(e)")
    return e, e_dict
end
