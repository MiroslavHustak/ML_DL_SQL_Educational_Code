namespace NeuralNetworks

open TorchSharp

open type torch.nn

module RMSNorm = 

    // Root Mean Square Layer Normalization
    type internal RMSNorm(normalizedShape: int64[], eps: float32) as self =

        inherit Module<torch.Tensor, torch.Tensor>("RMSNorm")
    
        let weight = torch.nn.Parameter(torch.ones normalizedShape)
        let eps = torch.tensor eps
        do self.RegisterComponents()
    
        override _.forward (x: torch.Tensor) =
            use norm = x.pow(torch.tensor(2.0f)).mean([|-1L|], keepdim = true).add(eps).sqrt() 
            x / norm * weight