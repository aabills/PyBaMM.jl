liquid_glycol_dict = Dict(
    "Density [kg.m3]" => 1115.,
    "Specific heat capacity [J.kg.K]" => 0.895,
    "Dynamic viscosity [Pa-s]" => 1.61e-2,
    "Thermal conductivity [W.m-K]" => 0.254,
)
novec_7200_dict = Dict(
    "Density [kg.m3]" => 1430.,
    "Specific heat capacity" => 1214.172,
    "Dynamic viscosity [Pa-s]" => 0.00061,
    "Thermal conductivity [W.m-K]" => 0.0616
)
novec_7500_dict = Dict(
    "Density [kg.m3]" => 1614.,
    "Specific heat capacity [J.kg.K]" => 1135.,
    "Dynamic viscosity [Pa-s]" => .00124,
    "Thermal conductivity [W.m-K]" => 0.065
)
water_dict = Dict(
    "Density [kg.m3]" => 997,
    "Specific heat capacity [J.kg.K]" => 4182,
    "Dynamic viscosity [Pa-s]" => 8.9e-4,
    "Thermal conductivity [W.m-K]" => 0.598
)
coolant_properties = Dict(
    "Liquid glycol" => liquid_glycol_dict,
    "Novec 7500" => novec_7500_dict,
    "Novec 7200" => novec_7200_dict,
    "Water" => water_dict
)