using NonlinearSolve

function mwe(out, u, p)
    out[1] = u[1] - p[1]
end

p = [5.0]

prob = NonlinearProblem(mwe, [5.2], p=p)
u = solve(prob, NewtonRaphson())

prob2 = remake(prob, p=[6.0])
u2 = solve(prob2, NewtonRaphson())