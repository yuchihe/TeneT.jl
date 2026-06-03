function energy_value(model::Heisenberg{Honeycomb{:c3v}}, A, env::Union{C3vCTMEnv,C3vTwoSiteCTMEnv}, params::iPEPSOptimize)
    A1, A2 = _c3v_site_pair(A)
    atype = _arraytype(A1)
    terms = _heisenberg_bond_terms(model, atype)

    if env isa C3vTwoSiteCTMEnv && length(A.data) == 2
        terms_BA = [(c, OR, OL) for (c, OL, OR) in terms]
        e_AB, _ = _c3v_two_site_bond(env, A1, A2, terms; site=1)
        e_BA, _ = _c3v_two_site_bond(env, A2, A1, terms_BA; site=2)
        e_bond = (e_AB + e_BA) / 2
        e_dict = Dict{String, Dict{String, Any}}(
            "bond_Heisenberg_C3v_energy" => Dict{String, Any}(
                "1,2" => e_AB,
                "2,1" => e_BA,
            ),
        )
    else
        e_bond, _ = _c3v_two_site_bond(env, A1, A2, terms)
        e_dict = Dict{String, Dict{String, Any}}(
            "bond_Heisenberg_C3v_energy" => Dict{String, Any}("1,1" => e_bond),
        )
    end
    e = 3 * real(e_bond) / 2

    params.verbosity >= 3 && println("energy = $(e)")
    return e, e_dict
end
