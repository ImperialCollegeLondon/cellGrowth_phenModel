#ABC algorithm output structure
@everywhere type ABCfit
  pts::Array{Any,2}
  sig::Array{Any,2}
  wts::Array{Any,2}
  p::Array{Float64,2}
  its::Array{Int64,1}
  dists::Array{Float64,2}
  epsilon::Array{Float64,1}
  temp::Array{Float64,2}
  pacc::Array{Float64}
end
