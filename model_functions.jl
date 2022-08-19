
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


function loglike(obs_ts, parms, problem, n_age, n_level, solvsettings) 
    issuccess = true

    # Update problem and solve ODEs
    problem_new = remake(problem; p=parms) 

    sol = solve(problem_new, 
                    solvsettings.solver, 
                    abstol=solvsettings.abstol, 
                    reltol=solvsettings.reltol, 
                    isoutofdomain = (u,p,t)->any(<(0),u),
                    saveat=solvsettings.saveat);

    # Return early if integration failed
    issuccess &= (sol.retcode === :Success)
    if !issuccess
        return  -Inf
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
    LL = sum(logpdf.(NegativeBinomial2.(parms[1], incidence), obs_ts))
    return LL
end

# observation model
function NegativeBinomial2(ψ, incidence)
    p = 1.0/(1.0 + ψ*incidence)
    r = 1.0/ψ
    return NegativeBinomial(r, p)
end
