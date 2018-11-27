#it's unwise to add methods to built-in functions, but it's the easiest way to handle exceptions on data input
@everywhere function rand(x::Vector)
  y=zeros(length(x))
  for i in 1:length(x)
    y[i]=rand(x[i])
  end
  return(y)
end

#it's unwise to add methods to built-in functions, but it's the easiest way to handle exceptions on data input
@everywhere function rand(x::Float64)
  return(x)
end

#it's unwise to add methods to built-in functions, but it's the easiest way to handle exceptions on data input
@everywhere function rand(x::Float64,y::Int64)
  return(fill(x,y))
end

#inner function called in the optimization process
@everywhere function pdf(x::Vector,z::Vector)
  y=Vector(length(x))
  for i in 1:length(x)
    y[i]=pdf(x[i],z[i])
  end
  return(prod(y))
end

#similar to hist() in MatLab, it counts occurrences of elements in vector
@everywhere function binning(zd::Vector)
    ret = zeros(Int64, 4)
    for k in zd;  ret[k] += 1;  end
    return ret
end

@everywhere function sumNArm(zd::Array{Float64,2})
  sum(x for x in zd if (x<10.0^50))
end


#############
#############
#moments calculation, innermost function (splitting needed for speed reasons)
@everywhere function allMoments!(v::Array{Float64,1},w::Array{Float64,1},rr::UnitRange{Int64})
    n   = length(v)
    cm1 = mean(v)
    cm2 = 0.0   #variance
    cm3 = 0.0   #skewness
    cm4 = 0.0   #kurtosis
    cm5 = 0.0   #fifth moment
    for i = 1:n
        @inbounds z = v[i] - cm1
        z2 = z*z;   cm2 += z2
        z2 *= z;    cm3 += z2
        z2 *= z;    cm4 += z2
        z2 *= z;    cm5 += z2
    end
    w[rr[1]] = cm1
    w[rr[2]] = cm2/n;   w[rr[3]] = cm3/n
    w[rr[4]] = cm4/n;   w[rr[5]] = cm5/n
end

#moments calculation, intermediate function (splitting needed for speed reasons)
@everywhere function moments_global_internal(inMAT::Array{Float64,2})
  tLEN    = inMAT[:,2]
  tINT    = inMAT[:,6]
  tY      = inMAT[inMAT[:,5].==0,[2,3,1]]            #tY columns: raw,dyeR,dyeD
  tY[:,3] = abs.(tY[:,3])
  tRATIO  = tY[:,2]
  tDELTA  = tY[:,3];

  tY      .-= mean(tY,1)
  tCovY   = tY'tY/(length(tDELTA)-1)
  #tX 	  = inMat[:,[2,6]] # tX columns: raw, DNA
  #tX     .-= mean(tX,1)
  #tCovX  = tX'tX/(length(tINT)-1)

  #outVEC = zeros(24)
  outVEC = zeros(23)
  allMoments!(tLEN,  outVEC,1:5);
  allMoments!(tRATIO,outVEC,6:10);
  allMoments!(tDELTA,outVEC,11:15);
  allMoments!(tINT,  outVEC,16:20);
  outVEC[21:23] = tCovY[[2,3,6]];
  #outVEC[24] 	= tCovX[2];
  return(outVEC)
end

#moments calculation, external function (without bootstrapping)
@everywhere function moments_global(funcMat::Vector{Array{Float64,2}})
  phasesInd = [1,3,4] #1=fase 1,2;  3=fase 3;   4=fase 4;
  expd_VAL  = zeros(12,25)
  #rows: 3 phases x 4 timepoints (0,4,7,10);
  #columns: 25 moments (5 length, 5 dye_ratio, 5 dye_delta, 5 DNA_int, 3 covariances, 1 ab. of unstained, 1 ab. of the phase)

  for aa in 1:4                 #rows, tpoints 1,2,3,  4,5,6,    7,8,9,  10,11,12
    tMatAA = funcMat[aa]        #tMatAA columns: dyeD,raw,dyeR,nuclei, unstained, DNAint
    sizeAA = size(tMatAA,1)
    for bb in 1:3               #rows, phases 1,4,7,10,   2,5,8,11,   3,6,9,12
      tMatBB  = tMatAA[tMatAA[:,4].==phasesInd[bb],:]     #tMatBB columns = tMatAAcolumns
      sizeBB  = size(tMatBB,1)
      cc = 3*(aa-1) + bb
      expd_VAL[cc,1:23] =   moments_global_internal(tMatBB)
      expd_VAL[cc,24]   =   sum(tMatBB[:,5])/sizeBB
      expd_VAL[cc,25]   =   sizeBB/sizeAA
    end
  end
  return(expd_VAL)
end

#moments calculation, external function (with bootstrapping)
@everywhere function moments_global_bootstrap(funcMat::Vector{Array{Float64,2}})
  phasesInd = [1,3,4] #1=fase 1,2;  3=fase 3;   4=fase 4;
  expd_VAL  = zeros(12,25)
  expd_BOOT = zeros(12,25)
  #rows: 3 phases x 4 timepoints (0,4,7,10);
  #columns: 25 moments (5 length, 5 dye_ratio, 5 dye_delta, 5 DNA_int, 3 covariances, 1 ab. of unstained, 1 ab. of the phase)

  for aa in 1:4               #rows, tpoints 1,2,3,  4,5,6,    7,8,9,  10,11,12
    tMatAA = funcMat[aa]      #tMatAA columns: dyeD,raw,dyeR,nuclei, unstained, DNAint
    sizeAA = size(tMatAA,1)
    for bb in 1:3             #rows, phases 1,4,7,10,   2,5,8,11,   3,6,9,12
      tMatBB  = tMatAA[tMatAA[:,4].==phasesInd[bb],:]   #tMatBB columns = tMatAAcolumns
      sizeBB  = size(tMatBB,1)
      cc = 3*(aa-1) + bb
      expd_VAL[cc,1:23] =   moments_global_internal(tMatBB)
      expd_VAL[cc,24]   =   sum(tMatBB[:,5])/sizeBB
      expd_VAL[cc,25]   =   sizeBB/sizeAA

      tempBoot  = zeros(1000,24)
      for j in 1:1000
        tBBboot = tMatBB[sample(1:sizeBB,sizeBB,replace=true),:]
        tempBoot[j,1:23]  = moments_global_internal(tBBboot)
        tempBoot[j,24]    = sum(tBBboot[:,5])/sizeBB
      end
      expd_BOOT[cc,1:24]  = sqrt.(var(tempBoot,1))
    end

    tempBoot = zeros(1000,3)
    for j in 1:1000
      tAAboot = tMatAA[sample(1:sizeAA,sizeAA,replace=true),4]
      for k in 1:3; tempBoot[j,k] = sum(tAAboot.==phasesInd[k])/sizeAA;  end
    end
    expd_BOOT[3*aa-2:3*aa,25] = sqrt.(var(tempBoot,1))
  end
  return([[expd_VAL];[expd_BOOT]])
end



#############
#############
#needed for phase identification; it returns a DNA intensity distribution
@everywhere function DNA_1_peak(peakM::Float64,peakN::Float64,peakInt::Float64,N::Int64)
  if      N==0;       t1= [];
  elseif  peakN==0;   t1= fill(peakM,N);
  else;               t1= rand(Normal(peakM,peakN),N);  end
  return(peakInt*t1)
end

#wrapping script for finding the phase in which cells divide; it is a small optimization script on its own
@everywhere function phase_detect(expd_DNA)
  #ERROR FUNCTIONS
  #G1: 3 peaks in mononucleate, 1 peak in binucleate
  @everywhere function rho_G1(expd,parami::Vector{Float64})
    #PARAMI: G1perc,Sperc(onNONG1), intN,inG2m,intSm, Gxm_mu,G1b_mu,Sm_mu,
    Nphs = expd[3];
    Nph1 = round.(Int,Nphs[1]*[parami[1],(1-parami[1])*parami[2],0]);	Nph1[3] = Nphs[1]-sum(Nph1);
    fullInd     = [ones(Nphs[1]);ones(Nphs[2])*3;ones(Nphs[3])*4][randperm(expd[2][1])]
    fullCollect = zeros(expd[2][1])
  	temp1 = DNA_1_peak(1.0,		 parami[6],parami[3],Nph1[1])[:,1]
  	temp2 = DNA_1_peak(parami[5],parami[8],parami[3],Nph1[2])[:,1]
  	temp3 = DNA_1_peak(parami[4],parami[6],parami[3],Nph1[3])[:,1]
	fullCollect[fullInd.==1] = [temp1;temp2;temp3]
    fullCollect[fullInd.==3] = DNA_1_peak(2.0,parami[7],parami[3],Nphs[2])[:,1]
    fullCollect[fullInd.==4] = DNA_1_peak(2.0,parami[7],parami[3],Nphs[3])[:,1]

    messInd     = sample(1:(expd[2][1]-1),expd[2][5],replace=false)
    temp12      = fullCollect[messInd]; temp21 = fullCollect[messInd+1]
    fullCollect[messInd] = temp21;  fullCollect[messInd+1] = temp12

    distanza = zeros(15)
    allMoments!(fullCollect[fullInd.==1],distanza,1:5)
    allMoments!(fullCollect[fullInd.==3],distanza,6:10)
    allMoments!(fullCollect[fullInd.==4],distanza,11:15)
    distanza = ((distanza - expd[1][1])./expd[1][2]).^2
    return(sqrt(mean(distanza)))
  end

  #G2: 1 peak in mononucleate, 3 peaks in binucleate
  @everywhere function rho_G2(expd,parami::Vector{Float64})
    #PARAMI: G1perc,Sperc(onNONG1), intN,intG1b,intSb, G2m_mu,Gxb_mu,Sb_mu
    Nphs = expd[3]
    Nph3 = round.(Int,Nphs[3]*[parami[1],(1-parami[1])*parami[2],0]);	Nph3[3] = Nphs[3]-sum(Nph3);
    fullInd     = [ones(Nphs[1]);ones(Nphs[2])*3;ones(Nphs[3])*4][randperm(expd[2][1])]
    fullCollect = zeros(expd[2][1])
    fullCollect[fullInd.==1] = DNA_1_peak(1.0,parami[6],parami[3],Nphs[1])[:,1]
    fullCollect[fullInd.==3] = DNA_1_peak(parami[4],parami[7],parami[3],Nphs[2])[:,1]
  	temp1 = DNA_1_peak(parami[4],parami[7],parami[3],Nph3[1])[:,1]
  	temp2 = DNA_1_peak(parami[5],parami[8],parami[3],Nph3[2])[:,1]
  	temp3 = DNA_1_peak(2.0,parami[7],parami[3],Nph3[3])[:,1]
    fullCollect[fullInd.==4] = [temp1;temp2;temp3]

    messInd     = sample(1:(expd[2][1]-1),expd[2][5],replace=false)
    temp12      = fullCollect[messInd]; temp21 = fullCollect[messInd+1]
    fullCollect[messInd] = temp21;  fullCollect[messInd+1] = temp12

    distanza = zeros(15)
    allMoments!(fullCollect[fullInd.==1],distanza,1:5)
    allMoments!(fullCollect[fullInd.==3],distanza,6:10)
    allMoments!(fullCollect[fullInd.==4],distanza,11:15)
    distanza = ((distanza - expd[1][1])./expd[1][2]).^2
    return(sqrt(mean(distanza)))
  end

  #S:  2 peaks in mononucleate, 2 peaks in binucleate
  @everywhere function rho_S(expd,parami::Vector{Float64})
    #PARAMI G1perc,G2perc, intN,intG2m,accrG1b, G2m_mu,G1b_mu,Sx_mu
  	tPar = parami[4]+parami[5];
  	Nphs = expd[3]
  	Nph1 = round.(Int,Nphs[1]*[0,parami[1]]);	Nph1[1] = Nphs[1] - Nph1[2];
    Nph3 = round.(Int,Nphs[3]*[parami[2],0]);	Nph3[2] = Nphs[3] - Nph3[1];
    fullInd     = [ones(Nphs[1]);ones(Nphs[2])*3;ones(Nphs[3])*4][randperm(expd[2][1])]

    fullCollect = zeros(expd[2][1])
  	temp1 = DNA_1_peak(1.0,		 parami[8],parami[3],Nph1[1])[:,1];
  	temp2 = DNA_1_peak(parami[4],parami[6],parami[3],Nph1[2])[:,1];
  	fullCollect[fullInd.==1] = [temp1;temp2]
    fullCollect[fullInd.==3] = DNA_1_peak(tPar,parami[7],parami[3],Nphs[2])[:,1]
  	temp1 = DNA_1_peak(tPar, parami[7],parami[3],Nph3[1])[:,1];
  	temp2 = DNA_1_peak(2.0,		  parami[8],parami[3],Nph3[2])[:,1];
    fullCollect[fullInd.==4] = [temp1;temp2]

    messInd     = sample(1:(expd[2][1]-1),expd[2][5],replace=false)
    temp12      = fullCollect[messInd]; temp21 = fullCollect[messInd+1]
    fullCollect[messInd] = temp21;  fullCollect[messInd+1] = temp12

    distanza = zeros(15)
    allMoments!(fullCollect[fullInd.==1],distanza,1:5)
    allMoments!(fullCollect[fullInd.==3],distanza,6:10)
    allMoments!(fullCollect[fullInd.==4],distanza,11:15)
    distanza = ((distanza - expd[1][1])./expd[1][2]).^2
    return(sqrt(mean(distanza)))
  end



  #MODELS
  model_G1=vcat(Uniform(0,1),Uniform(0,1), Uniform(0,100000),Uniform(1,2),Uniform(2,4), Uniform(0,1),Uniform(0,1),Uniform(0,1))
  #parami: G1perc,Sperc(onNONG1), intN,inG2m,intSm,     Gxm_mu,G1b_mu,Sm_mu,
  model_G2=vcat(Uniform(0,1),Uniform(0,1), Uniform(0,100000),Uniform(1,2),Uniform(2,4), Uniform(0,1),Uniform(0,1),Uniform(0,1))
  #parami: G1perc,Sperc(onNONG1), intN,intG1b,intSb,    G2m_mu,Gxb_mu,Sb_mu
  model_S =vcat(Uniform(0,1),Uniform(0,1), Uniform(0,100000),Uniform(0,1),Uniform(0,1), Uniform(0,1),Uniform(0,1),Uniform(0,1))
  #parami: G1perc,G2perc,         intN,intG2m,accrG1b,  G2m_mu,G1b_mu,Sx_mu



  #OPTIMIZATION
  models1 = Vector[model_G1,model_G2,model_S]
  rhocal1 = [rho_G1,rho_G2,rho_S]
  test_lens_DNA1=APMC(1000,expd_DNA,models1,0.5,rhocal1,0.001,2,1,0.0)
  a1 = deleteat!(collect(1:3),indmin(test_lens_DNA1.p[:,end]))

  models2 = models1[a1]
  rhocal2 = rhocal1[a1]
  test_lens_DNA2=APMC(1000,expd_DNA,models2,0.5,rhocal2,0.001,2,1,0.0)
  a2 = deepcopy(a1)
  a2 = deleteat!(a2,indmin(test_lens_DNA2.p[:,end]))

  fase      = ["G1","G2","S"][a2][1]
  return(fase)
end
