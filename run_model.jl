using Pkg 
cd(@__DIR__)
Pkg.activate("")
# Household stuff
using DifferentialEquations
using CSV, DataFrames
using StatsPlots
using Turing
using Distributions
import Random
using LabelledArrays
using UnPack
using DelimitedFiles
using Serialization
using BenchmarkTools
using ModelingToolkit

Random.seed!(36541)

# ODE model: simple SIR model with seasonally forced contact rate
# Model structure: (agegroups=4 * level=3 * state=4)
function MSIR_als!(du, u, p, t)

    ## params
    β = p[2]
    η = p[3]
    φ = p[4]
    σ = p[5]
    ω = 1.0 / p[6]
    μ = 1.0 / p[7]
    b = p[8]
    nₘ = p[9] # number of women in child-bearing age group
    d = p[10:13]
    ϵ = p[14:17]
    n = p[18:21]
    c = reshape(p[22:end], (4,4))

    ## Create views

    # current states
    N = @view u[:,:,1:end-1]
    M = @view u[:,:,1]
    S = @view u[:,:,2]
    I = @view u[:,:,3]
    R = @view u[:,:,4]

    ## differentials
    dN = @view du[:,:,1:end-1]
    dM = @view du[:,:,1] 
    dS = @view du[:,:,2]
    dI = @view du[:,:,3]
    dR = @view du[:,:,4]
    dC = @view du[:,:,5]

    # Transitions

    ## Ageing (includes births)
    ageing_out = zeros(promote_type(eltype(u), eltype(p)), size(N))
    @. ageing_out = ϵ * N
    ageing_in = zeros(promote_type(eltype(u), eltype(p)), size(N))
    ageing_in[2:end,:,:] += ageing_out[1:end-1,:,:] # ageing in from age age groups below

    ## Births
    pₘ = sum(R[3,:]) / nₘ # calculate the proportion of women of child-bearing age who are in the R compartments
    ageing_in[1,1,1] = b * pₘ # births into M compartment
    ageing_in[1,1,2] = b * (1.0-pₘ) # births into S compartment

    ## Deaths (calculated from births and transitions in and out, to maintain a stable N per age group)
    props_a = zeros(promote_type(eltype(u), eltype(p)), size(N))
    @. props_a = N / n # calculate proportions of n in each state
    deaths = zeros(promote_type(eltype(u), eltype(p)), size(N)) # distribute the age-specific deaths among the states
    @. deaths = d * props_a
    
    ## FOI

    # effective per contact transmission probability
    βeff = β * (1.0 + η * cos(2.0 * π * (t-φ) / 365.0))

    # total number of infectious agents by age group
    I_tot = sum(I, dims=2)

    λ = βeff .* vec(sum(c .* I_tot, dims=1))

    ## infections
    infection = .*(λ, S)

    ## clearance
    clearance = .*(σ, I)

    ## waning immunity
    waning_out_M = .*(μ, M) # waning out of M into S
    waning_out_R = zeros(promote_type(eltype(u), eltype(p)), size(R)) # waning out of R
    @. waning_out_R = *(ω, R) 

    waning_from_R = zeros(promote_type(eltype(u), eltype(p)), size(S)) # waning from R into S
    waning_from_R[:,2:end] += waning_out_R[:,1:end-1] 
    waning_from_R[:,end] += waning_out_R[:,end]
    
    # Equations
    ## births
    ## transitions between age groups 
    @. dN = ageing_in - ageing_out - deaths
    ## transitions between states
    @. dM = - waning_out_M
    @. dS = waning_out_M + waning_from_R - infection
    @. dI = infection - clearance
    @. dR = clearance - waning_out_R

    ## cumulative incidence
    @. dC = infection 

    return nothing
end


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

# observation model
function NegativeBinomial2(ψ, incidence)
    p = 1.0/(1.0 + ψ*incidence)
    r = 1.0/ψ
    return NegativeBinomial(r, p)
end


# Fake some data from model
data = rand.(NegativeBinomial2.(ψ, inc))
#scatter(data',legend = false);
#plot!(inc', legend = false) 

# Fit model to fake data

# Set up as Turing model
Turing.setadbackend(:forwarddiff)

@model function prior()
    ψ ~ Uniform(1e-6, 0.1) #Beta(1.1, 50.0) 
    β ~ Uniform(1e-6, 1.0) #Beta(1.5, 5.0) # 
    η ~ Beta(1.5, 10.0) #Uniform(0.0,1.0) 
    φ ~ Uniform(0.0,364.0)
    return [ψ, β, η, φ]
end

# Define prior and fixed theta
theta_fix = p[5:end]


@model function turingmodel_mtk(prior, theta_fix, problem, n_age, n_level, solvsettings) 
    
    issuccess = true

    # Sample prior parameters.
    theta_est = @submodel prior()
    # Update `p`.
    #theta_fix=convert.(eltype(theta_est), theta_fix)
    #promote_type(eltype(theta_fix), eltype(theta_est))

    p = vcat(theta_est, theta_fix) 
    # Update problem and solve ODEs
    problem_new = remake(problem; p=p) 

    sol = solve(problem_new, 
                    solvsettings.solver, 
                    abstol=solvsettings.abstol, 
                    reltol=solvsettings.reltol, 
                    isoutofdomain = (u,p,t)->any(<(0),u),
                    saveat=solvsettings.saveat);

    # Return early if integration failed
    issuccess &= (sol.retcode === :Success)
    if !issuccess
        Turing.@addlogprob! -Inf
        return nothing
    end
    
    sol_array = Array(sol);
    sol_array = sol_array[end-(n_age*n_level-1):end,:] # the last n_age *n_level rows correspond to the C compartment
    sol_array = diff(sol_array, dims=2)
    # sum over levels (data cannot distinguish between levels)
    incidence = reshape(sol_array, (n_age,n_level,size(sol_array,2)))
    incidence = dropdims(sum(incidence, dims=2), dims=2)
    # avoid numerical instability issue
    incidence = max.(eltype(incidence)(1e-9), incidence) 

    # likelihood
    obs_ts ~ arraydist(@. NegativeBinomial2(theta_est[1], incidence))

    return(; sol, incidence, p=p, obs_ts) 

end

# Setup and condition model
model = turingmodel_mtk(prior, 
                    theta_fix,
                    problem_mtk,
                    n_age,
                    n_level, 
                    solvsettings) | (obs_ts = data,);
                     
retval, issuccess = model();

# Fit 
rng = Random.MersenneTwister(66)
chain = sample(model, NUTS(3000, 0.65), MCMCThreads(), 1000, 6, progress=true) 

plot(chain)