# Rational polynomial interpolation

"""
Linear Rational LeastSquares

The following curvit is done:

`y = p(x) / q(x)`

where `p(x)` and `q(x)` are polynomials.

The linear case is solved by doing a least square fit on

`y * q(x) = p(x)`

where the zero order term o `q(x)` is assumed to be 1.
"""
function linear_rational_fit(
        x::AbstractVector{T}, y::AbstractVector{T}, p, q
) where {T <: Number}
    n = size(x, 1)
    A = zeros(T, n, q + p + 1)
    @inbounds for i in axes(x, 1)
        A[i, 1] = one(T)
        @simd ivdep for k in 1:p
            A[i, k + 1] = x[i]^k
        end
        @simd ivdep for k in 1:q
            A[i, p + 1 + k] = -y[i] * x[i]^k
        end
    end
    return qr!(A, ColumnNorm()) \ y
end

"""
# Type defining a rational polynomial

A rational polynomial is the ratio of two polynomials
and it is often useful in approximating functions.
"""
struct RationalPoly{T <: Number} <: AbstractLeastSquares
    num::Vector{T}
    den::Vector{T}
end
function RationalPoly(p::Integer, q::Integer, ::Type{T} = Float64) where {T <: Number}
    RationalPoly(zeros(T, p + 1), zeros(T, q + 1))
end

function RationalPoly(coefs::AbstractVector{T}, p, q) where {T <: Number}
    RationalPoly(collect(coefs[1:(p + 1)]), [one(T); collect(coefs[(p + 2):end])])
end

"""
Evaluate a rational polynomial
"""
ratval(r::RationalPoly, x) = evalpoly(x, r.num) / evalpoly(x, r.den)

"""
`call` overload for calling directly `ratval`
"""
(r::RationalPoly)(x) = ratval(r, x)

"""
Auxiliary function used in nonlinear least squares
"""
function make_rat_fun(p, q)
    return let p = p, q = q
        (y, x, a) -> begin
            num = view(a, 1:(p + 1))
            den = vcat(one(eltype(x)), view(a, (p + 2):(p + q + 1)))

            @inbounds @simd ivdep for i in eachindex(y)
                y[i] = evalpoly(x[1, i], num) / evalpoly(x[1, i], den) - x[2, i]
            end

            return y
        end
    end
end

"""
# Carry out a nonlinear least squares of rational polynomials

Find the polynomial coefficients that best approximate
the points given by `x` and `y`.
"""
function rational_fit(x, y, p, q, args...; kwargs...)
    coefs0 = linear_rational_fit(x, y, p, q)
    sol = nonlinear_fit(
        make_rat_fun(p, q), stack((x, y); dims=1), coefs0, args...;
        resid_prototype = Vector{eltype(coefs0)}(undef, length(x)), iip=Val(true), kwargs...
    )
    return sol.u
end

function curve_fit(::Type{RationalPoly}, x, y, p, q, args...; kwargs...)
    RationalPoly(rational_fit(x, y, p, q, args...; kwargs...), p, q)
end
