function zak(A, p, m, n, Re, Pr_f, Pr_w)
    return A*(Re^p)*(Pr_f^m)*((Pr_f/Pr_w)^n)
end

function nusselt_mixed(staggered::Bool, row, Re, Pr_f, Pr_w)
    if Re < 1
        error("Re < 1 should use the natural convection model.")
    end
    if row == 1
        if staggered
            if Re < 100.0
                Nu = zak(1, 0.4, 0.36, 0.25, Re, Pr_f, Pr_w)
            elseif Re < 2000.0
                Nu = zak(0.8, 0.45, 0.36, 0.25, Re, Pr_f, Pr_w)
            else
                error("Re > 2000 should use nusselt_turbulent")
            end
        else
            if Re < 100
                Nu = zak(1.02, 0.36, 0.36, 0.25, Re, Pr_f, Pr_w)
            elseif Re < 1500
                Nu = zak(0.7, 0.45, 0.36, 0.25, Re, Pr_f, Pr_w)
            end
        end
    else
        if staggered
            if Re < 40
                Nu = zak(1.04, 0.4, 0.36, 0.25, Re, Pr_f, Pr_w)
            elseif Re < 1500
                Nu = zak(0.71, 0.5, 0.36, 0.25, Re, Pr_f, Pr_w)
            else
                error("Re > 1500 should use nusselt_turbulent")
            end
        else
            if Re < 100
                Nu = zak(0.9, 0.4, 0.36, 0.25, Re, Pr_f, Pr_w)
            elseif Re < 1000
                Nu = zak(0.52, 0.5, 0.36, 0.25, Re, Pr_f, Pr_w)
            else
                error("Re > 1000 should use nusselt_turbulent")
            end
        end
    end
    return Nu
end

function nusselt_turbulent(a, b, Re, Pr_f, Pr_w)
    if Re < 1000
        error("for Re < 1000, use nusselt_mixed")
    end
    r = a/b
    if r < 2
        Nu = 0.35*(r^0.2)*(Re^0.6)*(Pr_f^0.36)*((Pr_f/Pr_w)^0.25)
    else
        Nu = 0.4*(Re^0.6)*(Pr_f^0.36)*((Pr_f/Pr_w)^0.25)
    end
    return Nu
end

function euler_series(Re, cs)
    Eu = 0
    for (i, c) in enumerate(cs)
        Eu = Eu + (c/(Re^(i-1)))
    end
    return Eu
end

function euler_inline(Re, a)
    if a < 1
        error("can't do a bank this tightly spaced")
    end
    if Re < 1
        error("euler number not defined for Re<1")
    end
    if a < 1.375
        if Re < 1e3
            cs = (0.272, 2.07e2, 1.02e2, -2.86e2)
        elseif Re < 2e6
            cs = (0.267, 2.49e3, -9.27e6, 1e10)
        end
    elseif a < 1.75
        if Re < 2e3
            cs = (0.263, 8.67e1, -2.02e-1)
        elseif Re < 2e6
            cs = (0.235, 1.97e3, -1.24e7, 3.12e10, -2.74e13)
        end
    elseif a < 2.125
        if Re < 8e2
            cs = (0.188, 5.66e1, -6.46e2, 6.01e3, -1.83e4)
        elseif Re < 2e6
            cs = (0.247, -5.95e-1, 1.5e-1, -1.37e-1, 3.96e-1)
        end
    end
    Eu = euler_series(Re, cs)
    return Eu
end

function euler_stagger(Re, b)
    if b < 1 || b > 2.5
        error("can't do a bank this tightly spaced")
    end
    if Re < 1
        error("euler number not defined for Re<1")
    end
    if b < 1.375
        if Re < 1e3
            cs = (0.795, 2.47e2, 3.35e2, -1.55e3, 2.41e3)
        elseif Re < 2e6
            cs = (0.245, 3.39e3, -9.84e6, 1.32e10, -5.99e12)
        end
    elseif b < 1.75
        if Re < 1e2
            cs = (0.683, 1.11e2, -9.73e1, -4.26e2, 5.74e2)
        elseif Re < 2e6
            cs = (0.203, 2.48e3, -7.58e6, 1.04e10, -4.82e12)
        end
    elseif b < 2.25
        if Re < 1e2
            cs = (0.713, 4.48e1, -1.26e2, -5.82e2)
        elseif Re < 1e4
            cs = (.343, 3.03e2, -7.17e4, 8.8e6, -3.8e8)
        elseif Re < 1e6
            cs = (0.162, 1.81e3, 7.92e7, -1.65e12, 8.72e15)
        end
    end
    Eu = euler_series(Re, cs)
    return Eu
end


