###############################################
###############################################
###PART 1: FILES CALLING STRUCTURE
# modelling_WRAPPER
#   subfunctions - nlm          (for generating synthetic data)
#   subfunctions - non nlm      (for calculating moments)
#   subfunctions - non nlm      (for identifying dividing phase)
#   function - optimization     (the real optimization part)
#     modelling_WRAPPER         (the rho functions and the models to be used)
#       subfunctions - nlm      (for running the simulation)
#       subfunctions - non nlm  (for calculating moments)


###############################################
###############################################
###PART 2: GENERAL INPUTS
#user inputs for finding scrips and files, and inputting simulation and correction parameters
genRoot   = "general path to models, data and results"
scriptRoot= "path to models"
fileRoot  = "path to data folder"
inputLeaf = "path to specific file"

dyeCycle  = [0.07,0.5,1.5,2.5]  #from OD measurements during timecourse    #dyeCycle[1] must be greater than 0! (ideally, >0.01)
extTP     = [3,6,10,13]         #timepoints to be employed (if starting from real data)
startFrom = 0                   #-1=synth data; 0=exp data, no steps done; 1=optimization performed
runPars   = [30,0.01]           #number of cell cycles, time step in cell cyles
techPars  = [0.045,0.01,2.2]    #unstained at t0; %error in classification; corrective factor for Δdye calculation

#initializing all possible cores for parallelising runs
cd(genRoot*scriptRoot)
if !isdir("Results"); mkdir("Results"); end
workN     = maximum(workers())
maxWorkN  = Sys.CPU_CORES
if workN<maxWorkN; addprocs(maxWorkN-workN); end

#loading external libraries and ancillary files on all cores
using StatsBase
using Distributions
using KernelDensityEstimate
#using HypothesisTests
@everywhere import Distributions.rand
@everywhere import Distributions.pdf
@everywhere import Distributions.sample #aggiunta mia
@everywhere import Distributions.Normal
@everywhere import Distributions.TruncatedNormal
@everywhere import Distributions.Uniform
@everywhere import Distributions.weights
include("types.jl")
include("subfunctions - nlm.jl")
include("subfunctions - non nlm.jl")
include("function - optimization.jl")



###############################################
###############################################
###PART 3: FILES LOADING OR POPULATION GENERATION

if startFrom<0  #create synthetic data using noisy linear map
  #NOTES: model: θ+, subpops+, nonEl_post (i.e. it is assumed that elongation stops at the appearance of the septum)
  genPars   = [1,5,0.10,0.05,0.01, 0.1,0,0.5,0.5, 1,2,0.75, 0.5,0.5]   #a,b,μ,ϴ,μgamma, aNoEl_mit,bNonEl_mit,aNonEl_post,bNonEl_post, netoR, # %smaller pop,% rate smaller pop
  DNApars   = [10000.0,0.5,0.7 ,0.7, 0.2,0.3,0.5]                      #intN,intG2m,accrG1b, G1perc, G2m_mu,G1b_mu,Sx_mu
  DNAphases = [1.40,0.825,0.10]                                        #αS,βS,μS
  simPars   = Vector{Real}(3);  simPars[1] = 10000; simPars[2:3] = runPars  #add number of cells
  fullMat   = Array{Array{Float64, 2}}(4)
  d0        = nlm_unst_WRAP(genPars,simPars[1],simPars[2],simPars[3],true,false,true,true,true)
  for aa in 1:4
    tMat  = nlm_dye_WRAP(genPars,simPars[1],dyeCycle[aa],simPars[3],d0,true,true,true,true,true,techPars[1],techPars[3])
    tMat  = tMat[map(isfinite,tMat[:,3]),:];
    d1DNA = nlm_DNA_SG2([DNApars;DNAphases],simPars[1],tMat[:,[6,7,2,4]])

    #simulating noise in cell classification
    mIL     = ceil(Int,simPars[1]*techPars[2])
    messInd = sample(1:simPars[1],2*mIL,replace=false)
    d1DNA[messInd] = d1DNA[[messInd[(mIL+1):(2*mIL)];messInd[1:mIL]]]

    #adding data to the general matrices
    fullMat[aa] = hcat(tMat[:,1:5],d1DNA)   #dyeD,raw,dyeR,ccPhase,unstInd,DNAint
  end
  resFoldRoot = "Results\\";  resFileRoot = "run_"*lpad(sample(1:999999),6,0)*"_"
  outRoot     = resFoldRoot*resFileRoot
  writedlm(outRoot*"parameters.txt", [genPars;dyeCycle;DNApars;DNAphases])

else            #files loading
  inputRoot   = genRoot*fileRoot
  indexVec    = [8,2,9,18,21,13]  #dyeD,rawL,dyeR,ccPhase,unstInd,DNAint
  fullMat0    = readdlm(inputRoot*inputLeaf,',',header=true)
  importFeatName = fullMat0[2]
  fullMat0    = fullMat0[1]
  fullMat0    = fullMat0[fullMat0[:,18].>0,:];                        #rimuovo eventuali errori nell'individuazione fasi
  fullMat0[fullMat0[:,18].>2,13]  = 2*fullMat0[fullMat0[:,18].>2,13]  #se cellule binucleate, semplicemente raddoppio l'intensità nucleare
  fullMat0[:,21]  = 0; fullMat0[fullMat0[:,9].==0,21] = 1;            #uniformo unstained check esterni a quelli richiesti per il modello
  fullMat0[:,end] = parse.(Int64,replace.(fullMat0[:,end],"t",""));   #necessario per dividere in timepoints
  fullMat     = Array{Array{Float64, 2}}(4)
  fullMat[1]  = fullMat0[fullMat0[:,end].==extTP[1],indexVec];  dyeCycle[1] = unique(fullMat0[fullMat0[:,end].==extTP[1],20])[1];
  fullMat[2]  = fullMat0[fullMat0[:,end].==extTP[2],indexVec];  dyeCycle[2] = unique(fullMat0[fullMat0[:,end].==extTP[2],20])[1];
  fullMat[3]  = fullMat0[fullMat0[:,end].==extTP[3],indexVec];  dyeCycle[3] = unique(fullMat0[fullMat0[:,end].==extTP[3],20])[1];
  fullMat[4]  = fullMat0[fullMat0[:,end].==extTP[4],indexVec];  dyeCycle[4] = unique(fullMat0[fullMat0[:,end].==extTP[4],20])[1];

  outRoot     = "Results\\"*inputLeaf*"_run_"*lpad(sample(1:999),3,0)
  simPars     = Vector{Real}(3);  simPars[1] = size(fullMat[3])[1]; simPars[2:3] = runPars
end
extPars     = [simPars;techPars]
extPars[5]  = ceil(Int,extPars[1]*extPars[5])



###############################################
###############################################
###PART 4: CALCULATE ALL MOMENTS
expd_TAB    = moments_global_bootstrap(fullMat)
expd_END    = Array{Any}(4)
expd_END[1] = expd_TAB
expd_END[2] = extPars
expd_END[4] = sample(1:extPars[1],2*extPars[5],replace=false)


###############################################
###############################################
###PART 5:  FINDING OUT THE PHASE
#AT THE MOMENT, THE SYNTHETIC GENERATION IS SET TO GENERATE POPULATIONS DIVIDING IN S PHASE

expd_DNA        = Array{Any}(3)
expd_DNA[1]     = Array{Array{Float64,1}}(2)
expd_DNA[1][1]  = vec(expd_TAB[1][1:3,16:20]')
expd_DNA[1][2]  = vec(expd_TAB[2][1:3,16:20]')
expd_DNA[2]     = extPars
expd_DNA[3]     = round.(Int,extPars[1]*expd_TAB[1][1:3,25])
if startFrom<1
  phase_sel     = phase_detect(expd_DNA)
  writedlm(outRoot*"divPhase.txt",phase_sel)
else
  phase_sel      = readdlm(outRoot*"divPhase.txt")
end

if      phase_sel == "G1"
    partMod_DNA = Uniform.([0.0,1.01,2.01, 0.0,0.0,0.0],[110000,2.0,4.0, 1.0,1.0,1.0])
    partMod_phS = Uniform.(0.0,[2.0,20.0,1.0,2.0,20.0,1.0])
    # 19:24   IntN,intG2m,intSm, μGxm,μG1b,μSm, aG1,bG1,μG1, aS,bS,μS
    @everywhere nlm_DNA_ch = nlm_DNA_G1SG2
    expd_END[3] = [6,7,2,4]
    prioriMode  = 3
elseif  phase_sel == "S"
    partMod_DNA = Uniform.([0.0,0.0,0.0, 0.0, 0.0,0.0,0.0],[110000.0,1.0,1.0, 1.0, 1.0,1.0,1.0])
    partMod_phS = Uniform.(0.0,[2.0,20.0,1.0])
    # 19:25   intN,intG2m,accrG1b, G1%, μG2m,μG1b,μSx, aS,bS,muS
    @everywhere nlm_DNA_ch = nlm_DNA_SG2
    expd_END[3] = [6,7,2,4]
    prioriMode  = 2
elseif  phase_sel == "G2"
    partMod_DNA = Uniform.([0.0,1.01,2.01, 0.0,0.0, 0.0,0.0,0.0],[110000.0,2.0,4.0, 1.0,1.0, 1.0,1.0,1.0])
    # 19:26   intN,intG1b,intSb, G1%,S%(nonG1), μG2m,μGxb,μSb
    partMod_phS = []
    @everywhere nlm_DNA_ch = nlm_DNA_G2
    expd_END[3] = 4
    prioriMode  = 1
end



###############################################
###############################################
###PART 6: DESCRIBE MODELS

partMod_size  = Uniform.(0.00001,[1.99999,20.0,1.0,1.0,0.02])   # 1:5               αT,βT,μT, ϴ,γ
partMod_modA  = Uniform.(0.0,[5.0,5.0,5.0,5.0])                 # 6:9               αNonEl_mit,βNonEl_mit, αNonEl_post,βNonEl_post  (nonEl_post=false)
partMod_modB  = Uniform.(0.0,[1.0,5.0,5.0])                     # 6:8               αNonEl_mit,αNonEl_post,βNonEl_post  (nonEl_post=true)
partMod_dye   = Uniform.(0.00001,[2.0,20.0,1.0])                # 10:12 (or 9:11)   netoTa, netoTb,netoR
partMod_subp  = Uniform.([0.0,0.1],[0.5,1.0])                   # 13:14 (or 12:3)   %smaller pop,% rate smaller pop
partMod_OD_t0 = TruncatedNormal(dyeCycle[1],0.015,0.01,Inf)
partMod_ODs   = [partMod_OD_t0;Normal.(dyeCycle[2:4],0.05)]     # 15:18 (max)       t1,t07,t10,t12



#models name code: model_xyz;  x=relevance of division noise; y=subpopulations yes/no;  z=modelA or modelB of nonElongation;
#checking relevance of division noise doesn't require additional models in this phase
model_x0A     = vcat(partMod_size, partMod_modA, partMod_dye,                partMod_ODs, partMod_DNA, partMod_phS)
model_x0B     = vcat(partMod_size, partMod_modB, partMod_dye,                partMod_ODs, partMod_DNA, partMod_phS)
model_x1A     = vcat(partMod_size, partMod_modA, partMod_dye, partMod_subp,  partMod_ODs, partMod_DNA, partMod_phS)
model_x1B     = vcat(partMod_size, partMod_modB, partMod_dye, partMod_subp,  partMod_ODs, partMod_DNA, partMod_phS)
model_END     = Vector[model_x0A,model_x0B, model_x1A,model_x1B,  model_x0A,model_x0B, model_x1A,model_x1B]



###############################################
###############################################
###PART 7: DESCRIBE ERROR FUNCTIONS

@everywhere function rho_lens_00A(expd,parami::Vector{Float64})
  parametri = parami[1:12];
  dyeCycle  = parami[13:16];  parDNA    = parami[17:end];
  Ntot = expd[2][1];    Nmess = expd[2][5];   messInd = expd[4]

  d0        = nlm_unst_WRAP(parametri,Ntot,expd[2][2],expd[2][3],true,false,false,false,false)
  if d0[1,1] ==0; return(Inf);  end
  simMat    = Array{Array{Float64, 2}}(4)
  for aa in 1:4
    tMat  = nlm_dye_WRAP(parametri,Ntot,dyeCycle[aa],expd[2][3],d0,true,true,false,false,false,expd[2][4],expd[2][6]);
    tMat  = tMat[map(isfinite,tMat[:,3]),:];
    tDNA  = nlm_DNA_ch(parDNA,Ntot,tMat[:,expd[3]])

    tDNA[messInd] = tDNA[[messInd[(Nmess+1):(2*Nmess)];messInd[1:Nmess]]]
    tMat[:,6]   = tDNA;
    simMat[aa]  = tMat[:,1:6]
  end
  sim_TAB   = moments_global(simMat)

  distance = ((sim_TAB - expd[1][1])./expd[1][2]).^2;
  return(sqrt(sumNArm(distance)))
end
#rho_lens_00A(expd_END,[genPars[1:12];dyeCycle;DNApars;DNAphases])

@everywhere function rho_lens_00B(expd,parami::Vector{Float64})
  parametri = parami[1:11]; splice!(parametri,6,[parametri[6],0.0])
  dyeCycle  = parami[12:15];  parDNA    = parami[16:end];
  Ntot = expd[2][1];    Nmess = expd[2][5];   messInd = expd[4]

  d0        = nlm_unst_WRAP(parametri,Ntot,expd[2][2],expd[2][3],true,false,false,false,true)
  if d0[1,1] ==0; return(Inf);  end
  simMat    = Array{Array{Float64, 2}}(4)
  for aa in 1:4
    tMat  = nlm_dye_WRAP(parametri,Ntot,dyeCycle[aa],expd[2][3],d0,true,true,false,false,true,expd[2][4],expd[2][6]);
    tMat  = tMat[map(isfinite,tMat[:,3]),:];
    tDNA  = nlm_DNA_ch(parDNA,Ntot,tMat[:,expd[3]])

    tDNA[messInd] = tDNA[[messInd[(Nmess+1):(2*Nmess)];messInd[1:Nmess]]]
    tMat[:,6]   = tDNA;
    simMat[aa]  = tMat[:,1:6]
  end
  sim_TAB   = moments_global(simMat)

  distance = ((sim_TAB - expd[1][1])./expd[1][2]).^2;
  return(sqrt(sumNArm(distance)))
end
#rho_lens_00B(expd_END,[genPars[[1:6;8:12]];dyeCycle;DNApars;DNAphases])

@everywhere function rho_lens_01A(expd,parami::Vector{Float64})
  parametri = parami[1:14];
  dyeCycle  = parami[15:18];  parDNA    = parami[19:end];
  Ntot = expd[2][1];    Nmess = expd[2][5];   messInd = expd[4]

  d0        = nlm_unst_WRAP(parametri,Ntot,expd[2][2],expd[2][3],true,false,false,true,false)
  if d0[1,1] ==0; return(Inf);  end
  simMat    = Array{Array{Float64, 2}}(4)
  for aa in 1:4
    tMat  = nlm_dye_WRAP(parametri,Ntot,dyeCycle[aa],expd[2][3],d0,true,true,false,true,false,expd[2][4],expd[2][6]);
    tMat  = tMat[map(isfinite,tMat[:,3]),:];
    tDNA  = nlm_DNA_ch(parDNA,Ntot,tMat[:,expd[3]])

    tDNA[messInd] = tDNA[[messInd[(Nmess+1):(2*Nmess)];messInd[1:Nmess]]]
    tMat[:,6]   = tDNA;
    simMat[aa]  = tMat[:,1:6]
  end
  sim_TAB   = moments_global(simMat)

  distance = ((sim_TAB - expd[1][1])./expd[1][2]).^2;
  return(sqrt(sumNArm(distance)))
end
#rho_lens_01A(expd_END,[genPars[1:14];dyeCycle;DNApars;DNAphases])

@everywhere function rho_lens_01B(expd,parami::Vector{Float64})
  parametri = parami[1:13]; splice!(parametri,6,[parametri[6],0.0])
  dyeCycle  = parami[14:17];  parDNA    = parami[18:end];
  Ntot = expd[2][1];    Nmess = expd[2][5];   messInd = expd[4]

  d0        = nlm_unst_WRAP(parametri,Ntot,expd[2][2],expd[2][3],true,false,false,true,true)
  if d0[1,1] ==0; return(Inf);  end
  simMat    = Array{Array{Float64, 2}}(4)
  for aa in 1:4
    tMat  = nlm_dye_WRAP(parametri,Ntot,dyeCycle[aa],expd[2][3],d0,true,true,false,true,true,expd[2][4],expd[2][6]);
    tMat  = tMat[map(isfinite,tMat[:,3]),:];
    tDNA  = nlm_DNA_ch(parDNA,Ntot,tMat[:,expd[3]])

    tDNA[messInd] = tDNA[[messInd[(Nmess+1):(2*Nmess)];messInd[1:Nmess]]]
    tMat[:,6]   = tDNA;
    simMat[aa]  = tMat[:,1:6]
  end
  sim_TAB   = moments_global(simMat)

  distance = ((sim_TAB - expd[1][1])./expd[1][2]).^2;
  return(sqrt(sumNArm(distance)))
end
#rho_lens_01B(expd_END,[genPars[[1:6;8:14]];dyeCycle;DNApars;DNAphases])

@everywhere function rho_lens_10A(expd,parami::Vector{Float64})
  parametri = parami[1:12];
  dyeCycle  = parami[13:16];  parDNA    = parami[17:end];
  Ntot = expd[2][1];    Nmess = expd[2][5];   messInd = expd[4]

  d0        = nlm_unst_WRAP(parametri,Ntot,expd[2][2],expd[2][3],true,false,true,false,false)
  if d0[1,1] ==0; return(Inf);  end
  simMat    = Array{Array{Float64, 2}}(4)
  for aa in 1:4
    tMat  = nlm_dye_WRAP(parametri,Ntot,dyeCycle[aa],expd[2][3],d0,true,true,true,false,false,expd[2][4],expd[2][6]);
    tMat  = tMat[map(isfinite,tMat[:,3]),:];
    tDNA  = nlm_DNA_ch(parDNA,Ntot,tMat[:,expd[3]])

    tDNA[messInd] = tDNA[[messInd[(Nmess+1):(2*Nmess)];messInd[1:Nmess]]]
    tMat[:,6]   = tDNA;
    simMat[aa]  = tMat[:,1:6]
  end
  sim_TAB   = moments_global(simMat)

  distance = ((sim_TAB - expd[1][1])./expd[1][2]).^2;
  return(sqrt(sumNArm(distance)))
end
#rho_lens_10A(expd_END,[genPars[1:12];dyeCycle;DNApars;DNAphases])

@everywhere function rho_lens_10B(expd,parami::Vector{Float64})
  parametri = parami[1:11]; splice!(parametri,6,[parametri[6],0.0])
  dyeCycle  = parami[12:15];  parDNA    = parami[16:end];
  Ntot = expd[2][1];    Nmess = expd[2][5];   messInd = expd[4]

  d0        = nlm_unst_WRAP(parametri,Ntot,expd[2][2],expd[2][3],true,false,true,false,true)
  if d0[1,1] ==0; return(Inf);  end
  simMat    = Array{Array{Float64, 2}}(4)
  for aa in 1:4
    tMat  = nlm_dye_WRAP(parametri,Ntot,dyeCycle[aa],expd[2][3],d0,true,true,true,false,true,expd[2][4],expd[2][6]);
    tMat  = tMat[map(isfinite,tMat[:,3]),:];
    tDNA  = nlm_DNA_ch(parDNA,Ntot,tMat[:,expd[3]])

    tDNA[messInd] = tDNA[[messInd[(Nmess+1):(2*Nmess)];messInd[1:Nmess]]]
    tMat[:,6]   = tDNA;
    simMat[aa]  = tMat[:,1:6]
  end
  sim_TAB   = moments_global(simMat)

  distance = ((sim_TAB - expd[1][1])./expd[1][2]).^2;
  return(sqrt(sumNArm(distance)))
end
#rho_lens_10B(expd_END,[genPars[[1:6;8:12]];dyeCycle;DNApars;DNAphases])

@everywhere function rho_lens_11A(expd,parami::Vector{Float64})
  parametri = parami[1:14];
  dyeCycle  = parami[15:18];  parDNA    = parami[19:end];
  Ntot = expd[2][1];    Nmess = expd[2][5];   messInd = expd[4]

  d0        = nlm_unst_WRAP(parametri,Ntot,expd[2][2],expd[2][3],true,false,true,true,false)
  if d0[1,1] ==0; return(Inf);  end
  simMat    = Array{Array{Float64, 2}}(4)
  for aa in 1:4
    tMat  = nlm_dye_WRAP(parametri,Ntot,dyeCycle[aa],expd[2][3],d0,true,true,true,true,false,expd[2][4],expd[2][6]);
    tMat  = tMat[map(isfinite,tMat[:,3]),:];
    tDNA  = nlm_DNA_ch(parDNA,Ntot,tMat[:,expd[3]])

    tDNA[messInd] = tDNA[[messInd[(Nmess+1):(2*Nmess)];messInd[1:Nmess]]]
    tMat[:,6]   = tDNA;
    simMat[aa]  = tMat[:,1:6]
  end
  sim_TAB   = moments_global(simMat)

  distance = ((sim_TAB - expd[1][1])./expd[1][2]).^2;
  return(sqrt(sumNArm(distance)))
end
#rho_lens_11A(expd_END,[genPars[1:14];dyeCycle;DNApars;DNAphases])

@everywhere function rho_lens_11B(expd,parami::Vector{Float64})
  parametri = parami[1:13]; splice!(parametri,6,[parametri[6],0.0])
  dyeCycle  = parami[14:17];  parDNA    = parami[18:end];
  Ntot = expd[2][1];    Nmess = expd[2][5];   messInd = expd[4]
  d0        = nlm_unst_WRAP(parametri,Ntot,expd[2][2],expd[2][3],true,false,true,true,true)
  if d0[1,1] ==0; return(Inf);  end
  simMat    = Array{Array{Float64, 2}}(4)
  for aa in 1:4
    tMat  = nlm_dye_WRAP(parametri,Ntot,dyeCycle[aa],expd[2][3],d0,true,true,true,true,true,expd[2][4],expd[2][6]);
    tMat  = tMat[map(isfinite,tMat[:,3]),:];
    tDNA  = nlm_DNA_ch(parDNA,Ntot,tMat[:,expd[3]])

    tDNA[messInd] = tDNA[[messInd[(Nmess+1):(2*Nmess)];messInd[1:Nmess]]]
    tMat[:,6]   = tDNA;
    simMat[aa]  = tMat[:,1:6]
  end
  sim_TAB   = moments_global(simMat)

  distance = ((sim_TAB - expd[1][1])./expd[1][2]).^2;
  #return(distance)
  return(sqrt(sumNArm(distance)))
end
#rho_lens_11B(expd_END,[genPars[[1:6;8:14]];dyeCycle;DNApars;DNAphases])

rho_END    = [rho_lens_00A,rho_lens_00B, rho_lens_01A,rho_lens_01B, rho_lens_10A,rho_lens_10B, rho_lens_11A,rho_lens_11B]



###############################################
###############################################
###PART 8: OPTIMIZATION

mod_ind   = 1:8;    mod_acc=8;
#reminder: sythetic data created with model 8 ("11B")
mod_names = ["00A","00B","01A","01B","10A","10B","11A","11B",]
#reminder: x= {θ-,θ+};  y= {subpops-,subpops+};  z= {nonEl_mit, nonEl_post}

#= INFERENCE
parameters: particles number, data to be fit to, models vector, fitting parameter (keep = 0.5),
error functions vector, termination parameter (the smaller you choose the longer fitting runs),
fitting parameter (keep at 2), division phase, results from previous run =#
if startFrom<1
  pn    = 0;
  Nrun  = 0;
  test_lens = 0.0; #it must be a Float64 number!
  while any(pn.<=0)
    Nrun += 1
    test_lens=APMC(800,expd_END,model_END,0.5,rho_END,0.001,2, prioriMode, test_lens)
    backup = deepcopy(test_lens)  #only for testing purposes
    pn = map(y->size(y)[2],test_lens.pts[:,end]);
    mod_acc = sum(pn.>0); mod_ind = mod_ind[pn.>0]; mod_names = mod_names[pn.>0]
    writedlm(outRoot*"test5_winner1_run"*string(Nrun)*".txt",[mod_names])

    keepInd = find(pn.>0);    # diffVal = diff(keepInd); attempted correction, to be tested
    diffVal=diff([0;keepInd]);  diffInd = find(diffVal.>1);   diffVal = diffVal[diffInd]-1
    model_END       = model_END[keepInd]
    rho_END         = rho_END[keepInd]
    test_lens.pts   = test_lens.pts[keepInd,end-1:end]
    test_lens.sig   = test_lens.sig[keepInd,end-1:end]
    test_lens.wts   = test_lens.wts[keepInd,end-1:end]
    test_lens.pacc  = test_lens.pacc[keepInd,end-1:end]
    test_lens.p     = test_lens.p[keepInd,end-1:end]
    test_lens.epsilon = test_lens.epsilon[end-1:end]
    #test_lens.its  = test_lens.its
    #test_lens.dists= fill(Inf,1,sum(size.(test_lens.pts,2)));
    #test_lens.temp = test_lens.temp[:,findin(test_lens.temp[1,:],keepInd)]
    modVec = test_lens.temp[1,:]
    for ii in 1:length(diffInd)
        tobemod = keepInd[diffInd[ii]:end]
        modVec[findin(test_lens.temp[1,:],tobemod)] = modVec[findin(test_lens.temp[1,:],tobemod)]-diffVal[ii]
    end
    test_lens.temp[1,:] = modVec
    while all(iszero.(test_lens.temp[end-2,:]))
      test_lens.temp = test_lens.temp[[1:end-3;end-1:end],:]
    end
  end
  writedlm(outRoot*"test5_winner1_end.txt",hcat(mod_ind,test_lens.p[:,2],pn))

  foundPars_STR = Vector{Array{Float64,2}}(mod_acc)
  for i in 1:mod_acc
    foundPars_STR[i] = test_lens.pts[i,end]
    writedlm(outRoot*"test5_pars_"*mod_names[i]*".txt",foundPars_STR[i])
  end
else
  fileIndex = readdir(resFoldRoot)
  fileIndex = fileIndex[contains.(fileIndex,resFileRoot*"test5_pars_")]
  mod_acc   = length(fileIndex)
  foundPars_STR = Vector{Array{Float64,2}}(mod_acc)
  for i in 1:mod_acc; foundPars_STR[i] = readdlm(resFoldRoot*fileIndex[i]);  end
  pn = map(y->size(y)[2],foundPars_STR[:,end]);
  mod_ind = Int64.(readdlm(outRoot*"test5_winner1_end.txt"))
end

foundPars_TAB = similar(foundPars_STR)
for i in 1:mod_acc
  tMat = foundPars_STR[i]; tMatN = size(tMat,2)
  if tMatN == 1
      foundPars_TAB[i] = [tMat zeros(size(tMat,1))];
  else
      if tMatN > 1000;  tMat = tMat[:,sample(1:tMatN,1000,replace=false)];  end
      tKde = kde!(tMat) #the only reason for limiting it to 1000 particles is the time needed to compute!
      foundPars_TAB[i] = [getKDEMax(tKde) std(foundPars_STR[i],2)];
  end
end



###############################################
###############################################
###PART 9: CREATING THE FINAL SET OF PARAMETERS

#pn = pn[pn.>1];
#foundPars_VEC = foundPars_TAB[indmax(pn)]; mod_ind=mod_ind[indmax(pn)]
foundPars_VEC = foundPars_TAB[indmax(test_lens.p[:,end])]; mod_ind=mod_ind[indmax(test_lens.p[:,end])]
winMod = repmat([mod_ind],1,size(foundPars_VEC)[2]);
winPhs = repmat([phase_sel],1,size(foundPars_VEC)[2]);

if      phase_sel == "G1"
  G1S_a = foundPars_VEC[end-2].*foundPars_VEC[end-5,:]
  G1S_b = foundPars_VEC[end-2,:].*foundPars_VEC[end-4,:] + foundPars_VEC[end-1,:]
  G1S_m = foundPars_VEC[end-2,:].*foundPars_VEC[end-3,:] + foundPars_VEC[end,:]
  G2_a  = foundPars_VEC[1,:]./G1S_a;
  G2_b  = foundPars_VEC[2,:] - G2_a.*G1S_b;
  G2_m  = foundPars_VEC[3,:] - G2_a.*G1S_m;
  finalSet  = vcat(foundPars_VEC, G2_a',G2_b',G2_m', winMod,winPhs)
elseif  phase_sel == "S"
  G2_a = foundPars_VEC[1,:]./foundPars_VEC[end-2,:];
  G2_b = foundPars_VEC[2,:] - G2_a.*foundPars_VEC[end-1,:];
  G2_m = foundPars_VEC[3,:] - G2_a.*foundPars_VEC[end,:];
  finalSet  = vcat(foundPars_VEC, G2_a',G2_b',G2_m', winMod,winPhs)
elseif  phase_sel == "G2"
  finalSet  = vcat(foundPars_VEC,winMod,winPhs)
end

if iszero(mod(mod_ind,2));  finalSet = [finalSet[1:6,:];[0 0];finalSet[7:end,:]]; end
writedlm(outRoot*"finalPars.txt",finalSet)
# 1:5         αT,βT,μT, ϴ,γ
# 6:9         αNonEl_mit,βNonEl_mit, αNonEl_post,βNonEl_post
# 10:12       netoTa, netoTb,netoR
# (13:14      %smaller pop,% rate smaller pop)
# 15:18       t1,t07,t10,t12
#if      phase_sel == "G1"
  # 19:24   IntN,intG2m,intSm, Gxm_mu,G1b_mu,Sm_mu
  # 25:33   αG1,βG1,μG1, αS,βS,μS, αG2,βG2,μG2
#elseif  phase_sel == "S"
  # 19:25   intN,intG2m,accrG1b, G1perc, G2m_mu,G1b_mu,Sx_mu
  # 26:31   αS,βS,μS, αG2,βG2,μG2
#elseif  phase_sel == "G2"
  # 19:26   intN,intG1b,intSb, G1perc,Sperc(onNONG1), G2m_mu,Gxb_mu,Sb_mu
#end
# end-1:end   winning_model, winning_phase

#to compare:
#setST = [genPars;dyeCycle;DNApars;DNAphases];
#setEN = finalSet[1:length(setST),:];
#setAL = hcat(setST,setEN);
