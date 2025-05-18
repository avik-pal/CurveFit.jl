function __linear_fit_internal(
        fnx::F1, x::AbstractArray{T1}, fny::F2, y::AbstractArray{T2}
) where {F1, F2, T1, T2}
    T = promote_type(T1, T2)
    m = length(x)

    sx2, sy2, sxy, sx, sy = zero(T), zero(T), zero(T), zero(T), zero(T)
    @simd ivdep for i in eachindex(x, y)
        fn_xi = fnx(x[i])
        fn_yi = fny(y[i])
        sx += fn_xi
        sy += fn_yi
        sx2 = muladd(fn_xi, fn_xi, sx2)
        sy2 = muladd(fn_yi, fn_yi, sy2)
        sxy = muladd(fn_xi, fn_yi, sxy)
    end

    a0 = (sx2 * sy - sxy * sx) / (m * sx2 - sx * sx)
    a1 = (m * sxy - sx * sy) / (m * sx2 - sx * sx)

    return (a0, a1)
end

"""
Fits a straight line through a set of points, `y = a₁ + a₂ * x`.
"""
function linear_fit(x::AbstractArray{T1}, y::AbstractArray{T2}) where {T1, T2}
    return __linear_fit_internal(identity, x, identity, y)
end

"""
Fits a log function through a set of points: `y = a₁ + a₂ * log(x)`.
"""
function log_fit(x, y)
    return __linear_fit_internal(log, x, identity, y)
end

"""
Fits a power law through a set of points: `y = a₁ * x^a₂`.
"""
function power_fit(x, y)
    a, b = __linear_fit_internal(log, x, log, y)
    return exp(a), b
end

"""
Fits an `exp` through a set of points: `y = a₁ * exp(a₂ * x)`
"""
function exp_fit(x, y)
    a, b = __linear_fit_internal(identity, x, log, y)
    return exp(a), b
end

"""
Create Vandermonde matrix for simple polynomial fit
"""
function vandermondepoly(x, n)
    A = similar(x, length(x), n + 1)
    A[:, 1] .= 1

    @inbounds for i in 1:n
        @simd ivdep for k in axes(A, 1)
            A[k, i + 1] = A[k, i] * x[k]
        end
    end
    return A
end

"""
Fits a polynomial of degree `n` through a set of points.

Simple algorithm, doesn't use orthogonal polynomials or any such thing and therefore
unconditioned matrices are possible. Use it only for low degree polynomials.

This function returns a the coefficients of the polynomial.
"""
poly_fit(x, y, n) = qr(vandermondepoly(x, n)) \ y

"""
High Level interface for fitting straight lines
"""
struct LinearFit{T <: Number} <: AbstractLeastSquares
    coefs::NTuple{2, T}
    LinearFit{T}(coefs) where {T <: Number} = new((coefs[1], coefs[2]))
    LinearFit{T}(c1, c2) where {T <: Number} = new((c1, c2))
end
function LinearFit(x::AbstractVector{T}, y::AbstractVector{T}) where {T <: Number}
    LinearFit{T}(linear_fit(x, y))
end

"""
High Level interface for fitting log laws
"""
struct LogFit{T <: Number} <: AbstractLeastSquares
    coefs::NTuple{2, T}

    LogFit{T}(coefs) where {T <: Number} = new((coefs[1], coefs[2]))
    LogFit{T}(c1, c2) where {T <: Number} = new((c1, c2))
end

function LogFit(x::AbstractVector{T}, y::AbstractVector{T}) where {T <: Number}
    return LogFit{T}(log_fit(x, y))
end

"""
High Level interface for fitting power laws
"""
struct PowerFit{T <: Number} <: AbstractLeastSquares
    coefs::NTuple{2, T}

    PowerFit{T}(coefs) where {T <: Number} = new((coefs[1], coefs[2]))
    PowerFit{T}(c1, c2) where {T <: Number} = new((c1, c2))
end

function PowerFit(x::AbstractVector{T}, y::AbstractVector{T}) where {T <: Number}
    return PowerFit{T}(power_fit(x, y))
end

"""
High Level interface for fitting exp laws
"""
struct ExpFit{T <: Number} <: AbstractLeastSquares
    coefs::NTuple{2, T}

    ExpFit{T}(coefs) where {T <: Number} = new((coefs[1], coefs[2]))
    ExpFit{T}(c1, c2) where {T <: Number} = new((c1, c2))
end

function ExpFit(x::AbstractVector{T}, y::AbstractVector{T}) where {T <: Number}
    return ExpFit{T}(exp_fit(x, y))
end

"""
# Generic interface for curve fitting.

The same function `curve_fit` can be used to fit the data depending on fit type,
which is specified in the first parameter. This function returns an object that
can be used to estimate the value of the fitting model using function `apply_fit`.

## A few examples:

 * `f = curve_fit(LinearFit, x, y)`
 * `f = curve_fit(Polynomial, x, y, n)`

Other types of fit include: LogFit, PowerFit, ExpFit, LinearKingFit, KingFit, RationalPoly.
See the documentation for details.
"""
curve_fit(::Type{T}, x, y) where {T <: AbstractLeastSquares} = T(x, y)
curve_fit(::Type{T}, x, y, args...) where {T <: AbstractLeastSquares} = T(x, y, args...)
curve_fit(::Type{Polynomial}, x, y, n = 1) = Polynomial(poly_fit(x, y, n))

(f::LinearFit)(x) = f.coefs[1] + f.coefs[2] * x
(f::PowerFit)(x) = f.coefs[1] * x^f.coefs[2]
(f::LogFit)(x) = f.coefs[1] + f.coefs[2] * log(x)
(f::ExpFit)(x) = f.coefs[1] * exp(f.coefs[2] * x)
