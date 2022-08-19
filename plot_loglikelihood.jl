using Pkg 
cd(@__DIR__)
Pkg.activate("")
# Household stuff
using DifferentialEquations
using CSV, DataFrames
using StatsPlots
using Distributions
import Random
using LabelledArrays
using UnPack
using DelimitedFiles
using Serialization
using BenchmarkTools
using ModelingToolkit

include("model_functions.jl")

Random.seed!(36541)


# Input

# demographics
pop = 1_000_000
t_y = [10.0, 10.0, 20.0, 30.0] # years spent in each age group
lifespan = sum(t_y)
props = t_y ./ lifespan
n = pop .* props

# Parameters
ψ = 0.05
β = 0.15
η = 0.05
φ = 180
σ = 1.0 / 5.0
ω = 365.0
μ = 180.0
b = 40.0 # daily births
nₘ = n[3]/2.0 # total women of child-bearing age
n_age = 4
n_level = 3

# ageing vector
ϵ = 1.0 ./ (365.0 .* props .* lifespan)

# calculate theoretical daily deaths in each age group from births and transitions in/out to achieve net 0 change
n_out = n .* ϵ
d = vcat(b, n_out[1:end-1]) - n_out

# contacts
c =  [1.4e-6  5.6e-6    2.275e-6   5.44444e-7
        5.6e-6  1.05e-5   2.8875e-6  1.08889e-6
        9.1e-6  1.155e-5  5.25e-6    2.13889e-6
        4.9e-6  9.8e-6    4.8125e-6  1.86667e-6]

# parameter vector
p = vcat(ψ, β, η, φ, σ, ω, μ, b, nₘ, d, ϵ, n, vec(c))

# Inits
u0 = [hcat(zeros(4), zeros(4), zeros(4)) ;;; # M
        hcat(pop .* props .- [0.0, 0.0, 10.0, 0.0], zeros(4), zeros(4)) ;;; # S
        hcat([0.0, 0.0, 10.0, 0.0], zeros(4), zeros(4)) ;;; # I
        hcat(zeros(4), zeros(4), zeros(4)) ;;; # R
        hcat(zeros(4), zeros(4), zeros(4))] 

# Solver settings
tmin = 0.0
tmax = 365.0 #5.0*365.0
tspan = (tmin, tmax)
solvsettings = (abstol = 1.0e-8, 
                reltol = 1.0e-8, 
                saveat = 1.0,
                solver = Tsit5())

# Initiate ODE problem
problem = ODEProblem(MSIR_als!, u0, tspan, p)
problem_mtk = ODEProblem(modelingtoolkitize(problem), [], tspan, jac=true)

sol = solve(problem, 
            solvsettings.solver, 
            abstol=solvsettings.abstol, 
            reltol=solvsettings.reltol, 
            isoutofdomain = (u,p,t)->any(<(0),u),
            saveat=solvsettings.saveat)

sol_array = Array(sol);
# sum over levels (data cannot distinguish between levels)
inc = dropdims(sum(sol_array, dims=2), dims=2)
inc = diff(inc[:,end,:], dims=2)
#plot(inc')

# Fake some data from model
data = rand.(NegativeBinomial2.(ψ, inc))
#scatter(data',legend = false);
#plot!(inc', legend = false) 
parms = p 
ψs = range(.005, .5, length=50)
LLs = map(
        x -> begin 
        parms[1] = x
        loglike(data, parms, problem_mtk, n_age, n_level, solvsettings)
        end,
        ψs) 


plot(ψs, LLs, xlabel="ψ", ylabel="LL", label=false, grid=false)


parms = p 
βs = range(.005, .5, length=50)
LLs = map(
        x -> begin 
        parms[2] = x
        loglike(data, parms, problem_mtk, n_age, n_level, solvsettings)
        end,
        βs) 


plot(βs, LLs, xlabel="β", ylabel="LL", label=false, grid=false)

parms = p 
ηs = range(.005, .5, length=50)
LLs = map(
        x -> begin 
                parms[3] = x
                loglike(data, parms, problem_mtk, n_age, n_level, solvsettings)
                end,
        ηs) 


plot(ηs, LLs, xlabel="η", ylabel="LL", label=false, grid=false)

parms = p 
φs = range(0, 365, length=50)
LLs = map(
        x -> begin 
                parms[4] = x
                loglike(data, parms, problem_mtk, n_age, n_level, solvsettings)
                end,
        φs) 


plot(φs, LLs, xlabel="φ", ylabel="LL", label=false, grid=false)
