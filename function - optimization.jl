using Distributions

#function needed for removing wrong parameter combinations
#checks are not enforced if not required or if performed somewhere else
@everywhere function check_priori(params::Vector{Float64},changes::Int64)
  if changes==1
    return(false)
  elseif changes==2
    (params[end-1] < params[2]*(1-params[end-2])/(2-params[1])) | #bS > bTot*(1-aS)/(2-aTot)
    (params[end-1] > params[2]*(2-params[end-2])/(2-params[1])) | #bS < bTot*(2-aS)/(2-aTot)
    (params[end-1] > params[end-2]*params[2]/params[1]) |         #bS < aS*bTot/aTot
    (params[end]   > params[end-2]*params[3]/params[1]) |         #μS < aS*μTot/aTot
    (params[end-1] > params[2]) |                                 #bS < bT
    (params[end-2] < params[1]/2)                                 #aS > aT/2
  elseif changes==3
    aX = params[end-2]*params[end-5]
    bX = params[end-2]*params[end-4] + params[end-1]
    mX = params[end-2]*params[end-3] + params[end]

    tempL=((params[end-4]<params[2]*( 1-params[end-5])/(2-params[1])) |   #bG1 > bTot*(1-aG1)/(2-aTot)
    (params[end-4] > bX + params[2]*(aX-params[end-5])/(2-params[1])) |   #bG1 < bX + bTot*(aX-aG1)/(2-aTot)
    (bX < params[2]*(1-aX)/(2-params[1])) |                               #bX  > bTot*(1-aX)/(2-aTot)
    (bX > params[2]*(2-aX)/(2-params[1])) |                               #bX  < bTot*(2-aX)/(2-aTot)
    (bX > aX*params[2]/params[1]) |                                       #bX  < aX*bTot/aTot
    (mX > aX*params[3]/params[1]) |                                       #μX  < aX*μTot/aTot
    (aX < params[1]/2)  |                                                 #aX  > aTot/2
    (bX > params[2])    |                                                 #bX  < bTot
    (params[end-4] > params[2]))                                          #bG1 < bTot
    return(tempL)
  end
end

#rejection sampler
#init_INwrap samples parameters from priori distributions
@everywhere function init_INwrap(models,expd,np::Vector{Int64},rho,changes::Int64)
  #doesn't perform additional checks on priori - already done!
  d=Inf
  count=1
  m=sample(1:length(models))

  params=rand(models[m])
  while check_priori(params,changes)
    m=sample(1:length(models))
    params=rand(models[m])
  end
  d=rho[m](expd,params)
  return vcat(m,params,fill(0,maximum(np)-np[m]),d,count)
end

#cont_INwrap samples and modifies parameters from the sampling done in the previous steps
@everywhere function cont_INwrap(models,pts,wts,expd,np::Vector{Int64},i::Int64,ker,rho::Vector{Function},changes::Int64)
  d=Inf
  count=1
  #while d==Inf
  m=sample(1:length(models))
  while size(pts[m,i-1])[2]==0
    m=sample(1:length(models))
  end
  params=pts[m,i-1][:,sample(1:size(pts[m,i-1])[2],wts[m,i-1])]
  params=params+rand(ker[m])
  while ((pdf(models[m],params)==0) | check_priori(params,changes))
    m=sample(1:length(models))
    while size(pts[m,i-1])[2]==0
      m=sample(1:length(models))
    end
    count=count+1
    params=pts[m,i-1][:,sample(1:size(pts[m,i-1])[2],wts[m,i-1])]
    params=params+rand(ker[m])
  end
  d=rho[m](expd,params)
  # end
  return vcat(m,params,fill(0,maximum(np)-np[m]),d,count)
end


#= Proper script
"models" contains a Vector of models;
"rho" contains a vector of functions to run the model and verify their output
"expd" contains the expected values and other simulation parameters
"changes" are needed to use the right version of check_priori, depending on the dividing phase
all other input parameters are specific to the optimization process =#
@everywhere function APMC(N::Int64,expd,models,prop::Float64,rho::Vector{Function},paccmin::Float64,n::Int64,changes::Int64,prevAPMC)
  lm  = length(models)
  s   = round(Int,N*prop)
  nbs = Array{Integer}(length(models))
  ker = Array{Any}(lm)                  #array for SMC kernel used in weights
  np  = Array{Int64}(length(models))    #array for number of parameters in each model
  for j in 1:lm;  np[j]=length(models[j]);  end

  if !isa(prevAPMC,Float64)
    i=2
    pts   = prevAPMC.pts
    sig   = prevAPMC.sig
    wts   = prevAPMC.wts
    p     = prevAPMC.p

    temp  = prevAPMC.temp
    its   = prevAPMC.its
    epsilon = prevAPMC.epsilon
    pacc  = prevAPMC.pacc
    dists = prevAPMC.dists
  else
    i=1
    template = Array{Any}(lm,1)
    pts = similar(template)   #particles array
    sig = similar(template)   #covariance matrix array
    wts = similar(template)   #weights array
    p   = zeros(lm,1)         #model probability at each iteration array

    temp=@parallel hcat for j in 1:N
      init_INwrap(models,expd,np,rho,changes)
    end
    its = [sum(temp[size(temp)[1],:])]
    epsilon=[quantile(collect(temp[maximum(np)+2,:]),prop)]
    pacc= ones(lm,1)
    println(round.([epsilon[i];its[i]],3))

    temp2   = temp[:,find(temp[maximum(np)+2,:].<=epsilon[i])];
    tNpart  = falses(lm);    for j in 1:lm; tNpart[j] = sum(temp2[1,:].==j) > np[j];  end
    if all(tNpart);   temp=temp2;  end
    #the first line is perfect in theory; however it causes downstream problems
    #indeed, if a model is strongly disadvantaged, it gets particleN<parsN
    #this means that no sig matrix could be calcolated, and the optimization fails
    #therefore the results of the first line are only used if it won't cause troubles.
    temp=temp[:,1:s]
    for j in 1:lm
      pts[j,i]=temp[2:(np[j]+1),find(temp[1,:].==j)]
      wts[j,i]=weights(fill(1.0,sum(temp[1,:].==j)))
    end
    dists=transpose(temp[(maximum(np)+2),:])
    for j in 1:lm
      p[j]=sum(wts[j,1])
    end
    for j in 1:lm
      sig[j,i]=cov(transpose(pts[j,i]),wts[j,i])
    end
    p=p./sum(p)
    for j in 1:lm
      nbs[j]=length(wts[j,i])
      println(round.(hcat(mean(diag(sig[j,i])[1:(np[j])]),pacc[j,i],nbs[j],p[j,i]),3))
    end
    if any(p.==0)
      samp=ABCfit(pts,sig,wts,p,its,dists,epsilon,temp,pacc)
      return(samp)
    end
  end


  while maximum(pacc[:,i])>paccmin
    pts=reshape(pts,i*length(models))
    sig=reshape(sig,i*length(models))
    wts=reshape(wts,i*length(models))
    for j in 1:length(models)
      push!(pts,Array{Any}(1))
      push!(sig,Array{Any}(1))
      push!(wts,Array{Any}(1))
    end
    pts=reshape(pts,length(models),i+1)
    sig=reshape(sig,length(models),i+1)
    wts=reshape(wts,length(models),i+1)
    i=i+1
    for j in 1:lm
      ker[j]=MvNormal(fill(0.0,np[j]),n*sig[j,i-1])
    end

    temp2=@parallel hcat for j in (1:(N-s))
      cont_INwrap(models,pts,wts,expd,np,i,ker,rho,changes)
    end
    its=vcat(its,sum(temp2[size(temp2)[1],:]))
    temp=hcat(temp,temp2)
    inds=sortperm(reshape(temp[maximum(np)+2,:],N))[1:s]
    temp=temp[:,inds]
    dists=hcat(dists,transpose(temp[(maximum(np)+2),:]))
    epsilon=vcat(epsilon,temp[(maximum(np)+2),s])
    pacc=hcat(pacc,zeros(lm))
    for j in 1:lm
      #HOW IT WORKS (for each model)
      #check1: if particles didn't exist and are not created, pacc = NaN
        #optimization stops and model is discarded
      #check2: if particles existed but are not created, pacc = 0.0
        #if pacc==0 in all models, optimization stops
      #if particles are created and some are OK, pacc = Float64 !=0
        #optimization continues
      #if particles are created and none are OK, pacc = 0.0
        #if pacc==0 in all models, optimization stops
      tSum0 = sum(temp[1,:].==j);  tSum1 = sum(temp[1,inds.>s].==j);  tSum2 = sum(temp2[1,:].==j);
      if iszero(tSum0);       pacc[j,i] = NaN;
      elseif iszero(tSum2);   pacc[j,i] = 0;
      else;                   pacc[j,i]=tSum1/tSum2;
      end
    end
    println(round.(vcat(epsilon[i],its[i]),3))
    for j in 1:lm
      pts[j,i]=temp[2:(np[j]+1),find(temp[1,:].==j)]
      if size(pts[j,i])[2]>0
        keep=inds[find(reshape(temp[1,:].==j,s))].<=s
        wts[j,i]= collect(@parallel vcat for k in 1:length(keep)
          if !keep[k]
            pdf(models[j],(pts[j,i][:,k]))/(1/(sum(wts[j,i-1]))*dot(values(wts[j,i-1]),pdf(ker[j],broadcast(-,pts[j,i-1],pts[j,i][:,k]))))
          else
            0.0
          end
        end)
        l=1
        for k in 1:length(keep)
          if keep[k]
            wts[j,i][k]=wts[j,i-1][l]
            l=l+1
          end
        end
        wts[j,i]=weights(wts[j,i])
      else
        wts[j,i]=zeros(0)
      end
    end
    p=hcat(p,zeros(length(models)))
    for j in 1:lm
      p[j,i]=sum(wts[j,i])
    end
    for j in 1:lm
      if (size(pts[j,i])[2]>np[j])
        #EXPLANATION: it's impossible/useless to calculate sig if Nr of particles is less than Nr of variables
        sig[j,i]=cov(transpose(pts[j,i]))
        if isposdef(n*sig[j,i])
          #EXPLANATION: it's impossible to calculate sig if the matrix is not positive definite
          dker=MvNormal(pts[j,i-1][:,1],n*sig[j,i])
          if pdf(dker,pts[j,i][:,1])==Inf
            sig[j,i]=sig[j,i-1]
          end
        else
          sig[j,i]=sig[j,i-1]
        end
      else
        sig[j,i]=sig[j,i-1]
      end
    end
    p[:,i]=p[:,i]./sum(p[:,i])
    for j in 1:lm
      nbs[j]=length(wts[j,i])
      println(round.(hcat(mean(diag(sig[j,i])./diag(sig[j,1])),pacc[j,i],nbs[j],p[j,i]),3))
    end
  end
  samp=ABCfit(pts,sig,wts,p,its,dists,epsilon,temp,pacc)
  return(samp)
end
