namespace NeuralNetworks

open TorchSharp

open type torch.nn

module RMSNorm = 

    // Root Mean Square Layer Normalization

    type RMSNorm(normalizedShape: int64[], eps: float32, device: torch.Device) as self =

        inherit Module<torch.Tensor, torch.Tensor>("RMSNorm")
    
        let weight = torch.nn.Parameter(torch.ones(normalizedShape).``to``(device))
        let eps = torch.tensor(eps).``to``(device)
        do self.RegisterComponents()
        do self.``to``(device) |> ignore<RMSNorm>        
    
        override _.forward (x: torch.Tensor) =
            use norm = x.pow(torch.tensor(2.0f).``to``(x.device)).mean([|-1L|], keepdim = true).add(eps).sqrt()
            x / norm * weight