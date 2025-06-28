namespace NeuralNetworks

module LoRa =
    
    open TorchSharp
    open type torch.nn
    
    type LoRALinear(inFeatures: int64, outFeatures: int64, rank: int64, alpha: float32, device: torch.Device) as this =

        inherit Module<torch.Tensor, torch.Tensor>("LoRALinear")
    
        // Main weights
        let weight = torch.empty([| outFeatures; inFeatures |], device = device).AsParameter()
        let bias = torch.zeros(outFeatures, device = device).AsParameter()
    
        // LoRA weights (A: random small init, B: zeros, per LoRA paper suggestion)
        let A : Modules.Parameter = (torch.randn([| rank; inFeatures |], device = device) * 0.01f).AsParameter()
        let B : Modules.Parameter = torch.zeros([| outFeatures; rank |], device = device).AsParameter()
    
        let scaling = alpha / (Microsoft.FSharp.Core.Operators.float32 rank)
    
        do
            this.register_parameter("weight", weight) |> ignore
            this.register_parameter("bias", bias) |> ignore
            this.register_parameter("A", A) |> ignore
            this.register_parameter("B", B) |> ignore
    
            init.kaiming_uniform_(weight) |> ignore
            init.zeros_(bias) |> ignore
            // LoRA: A is already initialized above, B is already zeros    
            
        override this.forward(input: torch.Tensor) =
            // LoRA adapter: (B @ A) * scaling
            let lora : torch.Tensor = B.matmul(A).mul(torch.tensor(scaling))
            let effectiveWeight = weight.add(lora)
            input.matmul(effectiveWeight.transpose(0L, 1L)).add(bias)
