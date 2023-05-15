using PyBaMM, ProgressMeter
pybamm = PyBaMM.pybamm
np = pyimport("numpy")

parameter_values = pybamm.ParameterValues("Chen2020")


function calculate_theoretical_energy(parameter_values, initial_soc, final_soc, points = 100)
    n_i, p_i = pybamm.lithium_ion.electrode_soh.get_initial_stoichiometries(initial_soc, parameter_values)
    n_f, p_f = pybamm.lithium_ion.electrode_soh.get_initial_stoichiometries(final_soc, parameter_values)
    n_range = collect(range(pyconvert(Float64,n_i), stop=pyconvert(Float64,n_f), length=points))
    p_range = collect(range(pyconvert(Float64,p_i), stop=pyconvert(Float64,p_f), length=points))
    V = zeros(points)
    for i in 1:points
        U⁺ = pyconvert(Float64,parameter_values["Positive electrode OCP [V]"](p_range[i]).value)
        U⁻ = pyconvert(Float64,parameter_values["Negative electrode OCP [V]"](n_range[i]).value) 
        V[i] = U⁺ - U⁻
    end

    #ONLY NEED 1 Q
    Q_max_pos = parameter_values["Maximum concentration in positive electrode [mol.m-3]"]
    #Q_max_neg = parameter_values["Maximum concentration in negative electrode [mol.m-3]"]
    W = parameter_values["Electrode width [m]"]
    H = parameter_values["Electrode height [m]"]
    T_pos = parameter_values["Positive electrode thickness [m]"]
    #T_neg = parameter_values["Negative electrode thickness [m]"]
    #εₛ⁻ = parameter_values["Negative electrode active material volume fraction"]
    εₛ⁺ = parameter_values["Positive electrode active material volume fraction"]
    vol_pos = W*H*T_pos*εₛ⁺
    #vol_neg = W*H*T_neg*εₛ⁻
    #Q_n = vol_neg*Q_max_neg*(n_f - n_i)
    Q_p = vol_pos*Q_max_pos*(p_f - p_i)
    dQ = pyconvert(Float64,Q_p/points)
    E = sum(V.*dQ) * 26.8
    return E
end

E = calculate_theoretical_energy(parameter_values, 1.0, 0.0)

energies = zeros(10, 10)
temps = collect(range(270, stop=340, length=10))
Cs = collect(range(0.1, stop=2.0, length=10))

@showprogress for (i, C) in enumerate(Cs)
    for (j, t) in enumerate(temps)
        parameter_values["Current function [A]"] = C*5.0
        parameter_values["Ambient temperature [K]"] = t
        model = pybamm.lithium_ion.DFN(options = pydict(Dict("calculate discharge energy"=>"true")))
        sim = pybamm.Simulation(model, parameter_values = parameter_values)
        sol = sim.solve(np.array(0:3600/C), initial_soc = 1.0)
        batt_efficiency = sol["Discharge energy [W.h]"].entries[-1]/E
        energies[i,j] = pyconvert(Float64,batt_efficiency)
    end
end

