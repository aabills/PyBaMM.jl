
import pybamm


def lico2_volume_change_Ai2020(sto, c_s_max):
    omega = pybamm.Parameter("Positive electrode partial molar volume [m3.mol-1]")
    t_change = omega * c_s_max * sto
    return t_change


def graphite_volume_change_Ai2020(sto, c_s_max):
    p1 = 145.907
    p2 = -681.229
    p3 = 1334.442
    p4 = -1415.710
    p5 = 873.906
    p6 = -312.528
    p7 = 60.641
    p8 = -5.706
    p9 = 0.386
    p10 = -4.966e-05
    t_change = (
        p1 * sto ** 9
        + p2 * sto ** 8
        + p3 * sto ** 7
        + p4 * sto ** 6
        + p5 * sto ** 5
        + p6 * sto ** 4
        + p7 * sto ** 3
        + p8 * sto ** 2
        + p9 * sto
        + p10
    )
    return t_change


def get_parameter_values():
    parameter_values = pybamm.ParameterValues(
        "Mohtat2020"
    )
    parameter_values.update(
        {
            # mechanical properties
            "Positive electrode Poisson's ratio": 0.3,
            "Positive electrode Young's modulus [Pa]": 375e9,
            "Positive electrode reference concentration for free of deformation [mol.m-3]": 0,
            "Positive electrode partial molar volume [m3.mol-1]": -7.28e-7,
            "Positive electrode volume change": lico2_volume_change_Ai2020,
            # Loss of active materials (LAM) model
            "Positive electrode LAM constant exponential term": 2,
            "Positive electrode critical stress [Pa]": 375e6,
            # mechanical properties
            "Negative electrode Poisson's ratio": 0.2,
            "Negative electrode Young's modulus [Pa]": 15e9,
            "Negative electrode reference concentration for free of deformation [mol.m-3]": 0,
            "Negative electrode partial molar volume [m3.mol-1]": 3.1e-6,
            "Negative electrode volume change": graphite_volume_change_Ai2020,
            # Loss of active materials (LAM) model
            "Negative electrode LAM constant exponential term": 2,
            "Negative electrode critical stress [Pa]": 60e6,
            # Other
            "Cell thermal expansion coefficient [m.K-1]": 1.48e-6,
        },
        check_already_exists=False,
    )
    parameter_values.update(
        {
            # SEI: all outer
            "Initial inner SEI thickness [m]": 0,
            "Initial outer SEI thickness [m]": 5e-9,
            "Lower voltage cut-off [V]": 2.5,
            "Upper voltage cut-off [V]": 4.2,
        }
    )
    return parameter_values