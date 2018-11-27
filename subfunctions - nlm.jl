using Distributions
using KernelDensity


##############
#STANDARD NLMs
#nlm function - innermost layer  (splitting needed for speed reasons)
#this is the very core of the whole optimization process, the one that is called about 3*10^12 times during the script
@everywhere function nlm_unst_INloop!(N::Int, fins::Vector{Float64}, starts::Vector{Float64}, grs::Vector{Float64}, div_waits::Vector{Float64}, a::Float64,b::Float64, aNonEl::Float64,bNonEl::Float64,d1,d2,d3, Nwei, unbiased::Bool, div_noise::Bool, subpops::Bool)
  for i in 1:N
    @fastmath @inbounds(
    if (div_waits[i]<0)
      tDiv=rand(d2)
      if unbiased;  dtr=sample(1:(N+1))
      else;         dtr=i
      end
      if dtr<(N+1)
        starts[dtr] = fins[i]*(1-tDiv)
        if subpops    grs[dtr]  = rand(d3[sample(1:2,Nwei)]);
        else;         grs[dtr]  = rand(d3);
        end
        if grs[dtr]<0; grs[dtr] = 0; end
        if div_noise; fins[dtr] = max(a*starts[dtr]+b+rand(d1),starts[dtr]);
        else;         fins[dtr] = max(a*fins[i]/2+b+rand(d1),starts[dtr]);
        end
        div_waits[dtr]= div_waits[i] + aNonEl*log2(fins[dtr]/starts[dtr])/grs[dtr] +bNonEl
      end
      if dtr!=i
        starts[i] =fins[i]*tDiv
        if subpops    grs[i]    = rand(d3[sample(1:2,Nwei)]);
        else;         grs[i]    = rand(d3);
        end
        if grs[i]<0; grs[i] = 0;  end
        if div_noise; fins[i]   = max(a*starts[i]+b+rand(d1),starts[i]);
        else;         fins[i]   = max(a*fins[i]/2+b+rand(d1),starts[i]);
        end
        div_waits[i]=   div_waits[i] + aNonEl*log2(fins[i]/starts[i])/grs[i] +bNonEl
      end
    end
    )
  end
  return(fins, starts, grs, div_waits)
end

#nlm function - intermediate layer  (splitting needed for speed reasons)
@everywhere function nlm_unst_OUTloop!(N::Int, tf::Float64, dt::Float64, fins::Vector{Float64}, starts::Vector{Float64}, grs::Vector{Float64}, div_waits::Vector{Float64}, a::Float64,b::Float64, aNonEl::Float64,bNonEl::Float64,d1,d2,d3, Nwei, unbiased::Bool, div_noise::Bool, subpops::Bool)
  t=0
  while t<tf
    t+= dt
    div_waits.-= dt
    (fins, starts, grs, div_waits) = nlm_unst_INloop!(N,fins,starts,grs,div_waits,a,b,aNonEl,bNonEl,d1,d2,d3,Nwei,unbiased,div_noise,subpops)
  end
  return(fins, starts, grs, div_waits)
end

#nlm function - external layer. Elaborates inputs and rearranges output in the desired format. Called for generating a steady-state population
#the boolean variable "div_noise" "subpops" "nonEl_post" are needed to run the appropriate version of the nlm given the input model
@everywhere function nlm_unst_WRAP(params::Vector{Float64}, N::Int64,tf::Float64,dt::Float64, unbiased::Bool=true,fullOut::Bool=false, div_noise::Bool=true,subpops::Bool=false,nonEl_post::Bool=true)
  #READING INPUTS PASSED TO FUNCTION
  a   = params[1];  b   = params[2];    Rab  = b/(2-a)
  if a==2 || b==0; return(0); end
  if nonEl_post;  aNonEl = params[8]+1;             bNonEl = params[9]
  else;           aNonEl = params[6]+params[8]+1;   bNonEl = params[7]+params[9]
  end
  d1 = Normal(0.0,max(Rab*params[3],0.00001))
  d2 = TruncatedNormal(0.5,max(params[4],0.00001),0.0,1.0)

  if subpops
    d3    = Vector{Distributions.Normal{Float64}}(2)
    d3[1] = Normal(1.0,max(params[5],0.00001))
    d3[2] = Normal(params[14],max(params[14]*params[5],0.00001))
    N2W   = params[13];     N2 = round(Int,N*N2W);    Nwei  = weights([1-N2W,N2W])
    grs   = [rand(d3[1],N-N2); rand(d3[2],N2)]; grs[grs.<0] = 0
  else
    d3    = Normal(1.0,max(params[5],0.00001))
    Nwei  = weights([N])
    grs   = rand(d3,N); grs[grs.<0] = 0
  end


  #GENERATING THE REQUIRED SEEDS
  starts    = max.(rand(Normal(Rab,0.1*Rab),N),1)
  fins      = max.(a*starts+b+rand(d1,N),starts)
  div_waits = rand(Uniform(0,1),N).*(aNonEl*log2.(fins./starts)./grs + bNonEl)

  #CORE OF THE SIMULATION (in external functions for speed optimization)
  (fins, starts, grs, div_waits) = nlm_unst_OUTloop!(N,tf,dt,fins,starts,grs,div_waits,a,b,aNonEl,bNonEl,d1,d2,d3,Nwei,unbiased,div_noise,subpops)

  #GENERATING THE REQUIRED OUTPUT
  #tEl = log2.(fins./starts)./grs;    tTot = aNonEl*tEl +bNonEl;    raw = min.(starts.*2.^(grs.*(tTot-div_waits)),fins);
  raw     = min.((fins.^aNonEl).*(starts.^ (1-aNonEl)) .*2.^(grs.*(bNonEl-div_waits)),fins)
  return  hcat(starts,fins,div_waits,grs,raw)
end

#nlm function - innermost layer  (splitting needed for speed reasons)
@everywhere function nlm_dye_INloop!(raN::Float64,aN::Float64,bN::Float64,corr_factor::Float64,N::Int, dyeRsta::Vector{Float64}, dyeRend::Vector{Float64}, dyeDsta::Vector{Float64},dyeDend::Vector{Float64}, netoL::Vector{Float64}, fins::Vector{Float64}, starts::Vector{Float64},
    grs::Vector{Float64}, div_waits::Vector{Float64}, a::Float64,b::Float64, aNonEl::Float64,bNonEl::Float64,d1,d2,d3, Nwei, unbiased::Bool, div_noise::Bool, subpops::Bool)
  for i in 1:N
    if (div_waits[i]<0)
      tDiv=rand(d2)
      luN = fins[i]*(1-dyeRend[i])/2 -dyeDend[i]
      luO = fins[i]*(1-dyeRend[i])/2 +dyeDend[i]

      if unbiased;  dtr=sample(1:(N+1));
      else;         dtr=i;
      end
      if dtr<(N+1)
        starts[dtr]=fins[i]*(1-tDiv)
        if subpops;   grs[dtr] = rand(d3[sample(1:2,Nwei)]);
        else;         grs[dtr] = rand(d3);
        end
        if grs[dtr]<0; grs[dtr] = 0; end

        if div_noise
          netoL[dtr]  = aN*starts[dtr]+bN
          fins[dtr]   = max(a*starts[dtr]+b+rand(d1),starts[dtr])
        else
          netoL[dtr]  = aN*fins[i]/2+bN
          fins[dtr]   = max(a*fins[i]/2+b+rand(d1),starts[dtr])
        end
        div_waits[dtr]=div_waits[i] + (aNonEl+1)*log2(fins[dtr]/starts[dtr])/grs[dtr] +bNonEl

        if dyeRsta[i]!=0
          if (luN<starts[dtr]) & (luO<fins[i]*tDiv)
            dyeRsta[dtr] = (starts[dtr]- luN)/(starts[dtr]+corr_factor)
            dyeDsta[dtr] = luN/2 - corr_factor
          elseif (luN>(starts[dtr]+corr_factor)) | (luO>(fins[i]*tDiv+corr_factor))
            dyeRsta[dtr] = dyeRend[i]*fins[i]/starts[dtr]
            dyeDsta[dtr] = fins[i]*tDiv/2 - abs(dyeDend[i])
          else
            dyeRsta[dtr] = dyeRend[i]*fins[i]/(starts[dtr]+corr_factor)
            dyeDsta[dtr] = max( (fins[i]*tDiv/2 - abs(dyeDend[i]) - corr_factor),-starts[dtr]/2)
            #il sistema e' stabile in assenza di fattori correttivi; il fattore suddetto richiede stabilizzazione esterna
          end
          dyeRend[dtr] = dyeRsta[dtr]*starts[dtr]/fins[dtr]
          dyeDend[dtr] = dyeDsta[dtr] + max((min(netoL[dtr],fins[dtr])-starts[dtr])/2, 0) + max((fins[dtr] -max(netoL[dtr],starts[dtr])),0)*(0.5-raN) #complete
        end
      end
      if dtr!=i
        starts[i]=fins[i]*tDiv
        if subpops;     grs[i] = rand(d3[sample(1:2,Nwei)]);
        else;           grs[i] = rand(d3);
        end
        if grs[i]<0; grs[i] = 0;  end

        if div_noise
          netoL[i]  = aN*starts[i]+bN
          fins[i]   = max(a*starts[i]+b+rand(d1),starts[i])
        else
          netoL[i]  = aN*fins[i]/2+bN
          fins[i]   = max(a*fins[i]/2+b+rand(d1),starts[i])
        end
        div_waits[i]=div_waits[i] + (aNonEl+1)*log2(fins[i]/starts[i])/grs[i] +bNonEl

        if dyeRsta[i]!=0
          if (luN<fins[i]*(1-tDiv)) & (luO<starts[i])
            dyeRsta[i] = (starts[i]- luO)/(starts[i]+corr_factor)
            dyeDsta[i] = luO/2 - corr_factor
          else
            dyeRsta[i] = 0
            dyeDsta[i] = 0
          end
          dyeRend[i] = dyeRsta[i]*starts[i]/fins[i]
          dyeDend[i] = dyeDsta[i] + max((min(netoL[i],fins[i])-starts[i])/2, 0) + max((fins[i] -max(netoL[i],starts[i])),0)*(0.5-raN) #complete
        end
      end
    end
  end
  return(fins,starts,grs,div_waits, dyeRsta,dyeDsta,netoL)
end

#nlm function - intermediate layer  (splitting needed for speed reasons)
@everywhere function nlm_dye_OUTloop!(raN::Float64,aN::Float64,bN::Float64,corr_factor::Float64,N::Int, tf::Float64,dt::Float64, dyeRsta::Vector{Float64}, dyeRend::Vector{Float64}, dyeDsta::Vector{Float64},dyeDend::Vector{Float64}, netoL::Vector{Float64}, fins::Vector{Float64}, starts::Vector{Float64},
    grs::Vector{Float64}, div_waits::Vector{Float64}, a::Float64,b::Float64, aNonEl::Float64,bNonEl::Float64,d1,d2,d3, Nwei, unbiased::Bool, div_noise::Bool, subpops::Bool)
    t=0
    while t<tf
      t+= dt
      div_waits-= dt
      (fins,starts,grs,div_waits, dyeRsta,dyeDsta,netoL) = nlm_dye_INloop!(raN,aN,bN,corr_factor,N,dyeRsta,dyeRend,dyeDsta,dyeDend,netoL,fins,starts,grs,div_waits,a,b,aNonEl,bNonEl,d1,d2,d3,Nwei,unbiased,div_noise,subpops)
    end
    return(fins,starts,grs,div_waits, dyeRsta,dyeDsta,netoL)
end

#nlm function - external layer. Elaborates inputs and rearranges output in the desired format. Called for generating a stained population, sampled after a specific time interval
#the boolean variable "div_noise" "subpops" "nonEl_post" are needed to run the appropriate version of the nlm given the input model
@everywhere function nlm_dye_WRAP(params::Vector{Float64},N::Int64,tf::Float64,dt::Float64,pops,unbiased::Bool=true,fullOut::Bool=false, div_noise::Bool=true,subpops::Bool=false,nonEl_post::Bool=true,corrFact1::Float64=0.045,corrFact3::Float64=2.2)

  #READING INPUTS PASSED TO FUNCTION
  #aNonEl_mit = params[6];   bNonEl_mit = params[7];   aNonEl_post= params[8];  bNonEl_post= params[9]
  if nonEl_post;  aNonEl = params[8];             bNonEl = params[9]
  else;           aNonEl = params[6]+params[8];   bNonEl = params[7]+params[9]
  end
  aN  = params[10]
  bN  = params[11]
  raN = params[12]
  a   = params[1]
  b   = params[2]
  d1  = Normal(0.0,max(params[3]*b/(2-a),0.00001))
  d2  = TruncatedNormal(0.5,max(params[4],0.00001),0.0,1.0)

  if subpops
    d3    = Vector{Distributions.Normal{Float64}}(2)
    d3[1] = Normal(1.0,max(params[5],0.00001))
    d3[2] = Normal(params[14],max(params[14]*params[5],0.00001))
    N2W   = params[13];     N2 = round(Int,N*N2W);    Nwei  = weights([1-N2W,N2W])
  else
    d3    = Normal(1.0,max(params[5],0.00001))
    Nwei  = weights([N])
  end


  #READING THE REQUIRED SEEDS
  starts    = pops[:,1]
  fins      = pops[:,2]
  div_waits = pops[:,3]
  grs       = pops[:,4]
  raw       = pops[:,5]
  netoL     = aN*starts+bN

  dyeRsta   = raw./starts #hypothetical, so that dyeR=1 at the begininning of the run
  dyeR      = ones(N)
  dyeRend   = raw./fins
  dyeDsta   = -max.((min.(raw,netoL)-starts)/2,0) -(0.5-raN)*(max.(raw,netoL)-max.(netoL,starts)) #hypothetical, so that dyeD=0 at the begininning of the run
  dyeD      = zeros(N)
  dyeDend   = max.((min.(netoL,fins)-raw)/2, 0) + max.((fins -max.(netoL,raw)),0)*(0.5-raN)
  ## dyeD   = dyeDsta + max.((min.(raw,netoL)-starts)/2,0) + (0.5-raN)*(max.(raw,netoL)-max.(netoL,starts)) #at the end


  #CORE OF THE SIMULATION (in external functions for speed optimization)
  (fins,starts,grs,div_waits, dyeRsta,dyeDsta,netoL) = nlm_dye_OUTloop!(raN,aN,bN,corrFact3,N,tf,dt,dyeRsta,dyeRend,dyeDsta,dyeDend,netoL,fins,starts,grs,div_waits,a,b,aNonEl,bNonEl,d1,d2,d3,Nwei,unbiased,div_noise,subpops)


  #GENERATING THE REQUIRED OUTPUT
  tEl   = log2.(fins./starts)./grs
  tTot  = (aNonEl+1)*tEl +bNonEl
  raw   = min.(starts.*2.^(grs.*(tTot-div_waits)),fins)
  dyeR  = dyeRsta.*starts./raw
  dyeD  = dyeDsta + max.((min.(raw,netoL)-starts)/2,0) + (0.5-raN)*(max.(raw,netoL)-max.(netoL,starts))
  dyeD[dyeR.==0]=0

  ccPhase = ones(N);
  if nonEl_post;  tBin = tEl*(1-params[6]) - params[7];   tSep = tEl;   #nonEL starts when septum appears
  else            tSep = tEl*(1+params[6]) + params[7];   tBin = tEl;   #nonEL starts when nuclei divide
  end
  ccPhase[div_waits.<=(tTot-tBin)]  = 3
  ccPhase[div_waits.<=(tTot-tSep)]  = 4

  unstInd = zeros(N);   unstCorr = round(Int,corrFact1*N);
  dyeD[1:unstCorr] = 0;     #since the vector isn't ordered, it's the same as sampling but much faster
  dyeR[1:unstCorr] = 0;     #since the vector isn't ordered, it's the same as sampling but much faster
  unstInd[(raw.*dyeR).<1] = 1;

  return hcat(dyeD,raw,dyeR,ccPhase,unstInd, starts,grs)
end


##############
#DNA-RELATED NLMs

#inputs: 1) nlm parameters of all elongating cell cycle stages; 2) %duration of non-elongating stages; 3) DNAint distribution parameters
#output: DNA distribution parameters, for cell that divide in G2 phase
@everywhere function nlm_DNA_G2(params::Vector{Float64},N::Int64,d0::Array{Float64,1})
  # intG2m < intG1b < intG2b   < intSb;	intG2m==intG2b/2
  # intG2m < intG1b < 2*intG2m < intSb;
  # intG2m = 1; intG2b = 2;	intG1b=[1,2];	intSb=[2,4];
  #2) NEW order of input parameters: intN,intG1b,intSb, G1perc,Sperc, G2m_mu,Gxb_mu,Sb_mu

  intN = params[1]; intG1 = params[2]; intS = params[3];
  G1perc = params[4]; Sperc = params[5]*(1-G1perc)
  if params[6]>0; dDNAph1 = Normal(1, params[6]);    	else; dDNAph1 = 1;      end
  if params[7]>0; dDNAph2 = Normal(intG1, params[7]);	else; dDNAph2 = intG1;  end
  dDNAph3 = dDNAph2;
  if params[8]>0; dDNAph4 = Normal(intS, params[8]);	else; dDNAph4 = intS;   end
  if params[7]>0; dDNAph5 = Normal(2, params[7]);     else; dDNAph2 = 2;      end

  extPhsVec = round.(Int,d0)
  extPhsBin = binning(extPhsVec); extIntVec = zeros(Float64,length(extPhsVec))
  phs     = zeros(Int64,5)
  phs[1]  = extPhsBin[1]                     #mononucleate, G2 or M phase
  phs[2]  = extPhsBin[3]                     #binucleate, M phase
  phs[3]  = round(Int,extPhsBin[4]*G1perc)   #binucleate, G1 phase
  phs[4]  = round(Int,extPhsBin[4]*Sperc)    #binucleate, S phase
  phs[5]  = extPhsBin[4] - phs[3] - phs[4]   #binucleate, G2 phase (bef div)

  temp3=[]; temp4=[]; temp5=[];
  if phs[1]!=0;   extIntVec[extPhsVec.==1]= rand(dDNAph1,phs[1]);   end
  if phs[2]!=0;   extIntVec[extPhsVec.==3]= rand(dDNAph2,phs[2]);   end
  if phs[3]!=0;   temp3= rand(dDNAph3,phs[3]);  end
  if phs[4]!=0;   temp4= rand(dDNAph4,phs[4]);  end
  if phs[5]!=0;   temp5= rand(dDNAph5,phs[5]);  end
  extIntVec[extPhsVec.==4] = [temp3; temp4; temp5]
  return intN*extIntVec
end

#inputs: 1) nlm parameters of all elongating cell cycle stages; 2) %duration of non-elongating stages; 3) DNAint distribution parameters
#output: DNA distribution parameters, for cell that divide in G1 phase
@everywhere function nlm_DNA_G1SG2(params::Vector{Float64},N::Int64,d0::Array{Float64,2})
  # intG1m < intG2m < intG1b   < intSm;	intG1m==intG1b/2
  # intG1m < intG2m < 2*intG1m < intSm:
  # intG1m = 1; intG1b = 2;	intG2m=[1,2];	intSm=[2,4];
  #2) NEW order of input parameters: intN,intG2m,intSm, Gxm_mu,G1b_mu,Sm_mu, nlm pars

  intN = params[1];	intG2 = params[2];	intS = params[3];
  aG1 = params[7];	bG1 = params[8];    nG1 = params[9];
  aS  = params[10];	bS  = params[11];	  nS  = params[12];
  aG1S = aG1*aS;    bG1S = aS*bG1+bS;   nG1S = aS*nG1+nS;
  if nG1>0;     dG1 = Normal(0.0, nG1*bG1/(2-aG1));       else; dG1=0.0;      end
  if nS>0;      dS  = Normal(0.0, nS*bS/(2-aS));          else; dS=0.0;       end
  if nG1S>0;    dG1S= Normal(0.0, nG1S*bG1S/(2-aG1S));    else; dG1S=0.0;     end

  if params[4]>0; dDNAph1 = Normal(1.0,  params[4]);    else; dDNAph1 = 1.0;    end
  if params[6]>0; dDNAph2 = Normal(intS, params[6]);  	else; dDNAph2 = intS;   end
  if params[4]>0; dDNAph3 = Normal(intG2,params[4]);  	else; dDNAph3 = intG2;	end
  if params[5]>0; dDNAph4 = Normal(2.0,  params[5]);    else; dDNAph4 = 2.0; 		end
  dDNAph5 = dDNAph4;


  starts  = d0[:,1];  grs = d0[:,2];  raw = d0[:,3];  extPhsVec = round.(Int,d0[:,4])
  extPhsBin = binning(extPhsVec); extIntVec = zeros(Float64,length(extPhsVec))
  lenG1   = max.(aG1*starts + bG1 + rand(dG1),starts)
  lenS    = max.(aS*lenG1 + bS + rand(dS),lenG1)

  phs    = zeros(Int64,5)
  phs[1] = min(sum(raw.<=lenG1),extPhsBin[1])           #mononucleate, G1 phase
  phs[2] = min(sum(raw.<=lenS)-phs[1],(extPhsBin[1]-phs[1]))   #mononucleate, S phase
  phs[3] = extPhsBin[1] - phs[1] - phs[2]               #munonucleate, G2 phase
  phs[4] = extPhsBin[3];                                #binucleate, M phase
  phs[5] = extPhsBin[4];                                #binucleate, G1 phase

  temp1=[]; temp2=[]; temp3=[];
  if phs[1]!=0; temp1= rand(dDNAph1, phs[1]);  end
  if phs[2]!=0; temp2= rand(dDNAph2, phs[2]);  end
  if phs[3]!=0; temp3= rand(dDNAph3, phs[3]);  end
  if phs[4]!=0; extIntVec[extPhsVec.==3] = rand(dDNAph4,phs[4]);    end
  if phs[5]!=0; extIntVec[extPhsVec.==4] = rand(dDNAph5,phs[5]);    end
  extIntVec[extPhsVec.==1] = [temp1; temp2;  temp3]
  return intN*extIntVec
end

#inputs: 1) nlm parameters of all elongating cell cycle stages; 2) %duration of non-elongating stages; 3) DNAint distribution parameters
#output: DNA distribution parameters, for cell that divide in S phase
@everywhere function nlm_DNA_SG2(params::Vector{Float64},N::Int64,d0::Array{Float64,2})
  # intG2m < intSm;	 intG1b<intSb;	intG2m<intG1b;	intSb=2*intSm
  # intG2m < intSm;	 intG2m<intG1b<2*intSm;					#but it's unknown whether intSm<intG1b or not!
  # intSm = 1;	intSb = 2;	intG2m=[0 1];	intG1b=[0 2];	#but it's not sure that intG2m<intG1b
  # intSm = 1;	intSb = 2;	intG2m=[0 1];	intG1b=intG2m+[0 1];
  #2) NEW order of input parameters: intN,intG2m,accrG1b, G1perc, G2m_mu,G1b_mu,Sx_mu, nlm pars

  intN = params[1]; intG2 = params[2]; intG1 = intG2+params[3]; G1perc = params[4];
  aS = params[8];   bS =params[9]
  if params[10]>0;  d1S=Normal(0.0, params[10]*bS/(2-aS)); else; d1S=0.0;       end
  if params[7]>0;   dDNAph1 = Normal(1.0,params[7]);	  else; dDNAph1 = 1.0;		end
  if params[5]>0;   dDNAph2 = Normal(intG2,params[5]);	else; dDNAph2 = intG2;	end
  if params[6]>0;   dDNAph3 = Normal(intG1,params[6]);	else; dDNAph3 = intG1;	end
  dDNAph4 = dDNAph3;
  if params[7]>0;   dDNAph5 = Normal(2.0,params[7]);	  else; dDNAph5 = 2.0;		end

  starts  = d0[:,1];  grs = d0[:,2];  raw = d0[:,3];  extPhsVec = round.(Int,d0[:,4])
  extPhsBin = binning(extPhsVec); extIntVec = zeros(Float64,length(extPhsVec))
  lenS    = max.(aS*starts + bS + rand(d1S),starts)

  phs    = zeros(Int64,5)
  phs[1] = min(sum(raw.<=lenS),extPhsBin[1])  #mononucleate, S phase (aft div)
  phs[2] = extPhsBin[1] - phs[1]              #mononucleate, G2 or M phase
  phs[3] = extPhsBin[3]                       #binucleate, M phase
  phs[4] = round(Int,extPhsBin[4]*G1perc)     #binucleate, G1 phase
  phs[5] = extPhsBin[4] - phs[4]              #binucleate, S phase (bef div)

  temp1=[]; temp2=[]; temp4=[]; temp5=[];
  if phs[1]!=0;	temp1= rand(dDNAph1, phs[1]);	  end
  if phs[2]!=0; temp2= rand(dDNAph2, phs[2]);	  end
  if phs[4]!=0; temp4= rand(dDNAph4, phs[4]); 	end
  if phs[5]!=0; temp5= rand(dDNAph5, phs[5]); 	end
  if phs[3]!=0; extIntVec[extPhsVec.==3] = rand(dDNAph3,phs[3]);    end
  extIntVec[extPhsVec.==1] = [temp1; temp2]
  extIntVec[extPhsVec.==4] = [temp4; temp5]
  return intN*extIntVec
end
