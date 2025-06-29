namespace NeuralNetworks

open System.Collections.Generic

module LoRa =
    
    open TorchSharp
    open type torch.nn
    
    //LoRA Low-Rank Adaptation
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

            A.requires_grad <- true
            B.requires_grad <- true

            init.kaiming_uniform_(weight) |> ignore
            init.zeros_(bias) |> ignore
            // LoRA: A is already initialized above, B is already zeros    
            
        override this.forward(input: torch.Tensor) =
           let projA = input.matmul(A.transpose(0L, 1L)) // [batch, rank]
           let projB = projA.matmul(B.transpose(0L, 1L)) // [batch, outF]
           let scaled = projB.mul(torch.tensor(scaling, dtype=input.dtype, device=input.device))
       
           input.matmul(weight.transpose(0L, 1L)).add(scaled).add(bias)

    let save_LoRA_adapters (model: torch.nn.Module) (path: string) =

        model.named_parameters()
        |> Seq.filter 
            (fun struct (name, _) -> name.EndsWith(".A") || name.EndsWith(".B"))
        |> Seq.iter
            (fun struct (name, param)
                ->
                let filePath = sprintf "%s_%s.pt" path (name.Replace('.', '_'))
                torch.save(param, filePath)
            )
    
    let load_LoRA_adapters (model: torch.nn.Module) (path: string) =

        use _ = torch.no_grad()
    
        model.named_parameters()
        |> Seq.filter (fun struct (name, _) -> name.EndsWith(".A") || name.EndsWith(".B"))
        |> Seq.iter 
            (fun struct (name, param) 
                ->
                let filePath = sprintf "%s_%s.pt" path (name.Replace('.', '_'))

                match System.IO.File.Exists(filePath) with
                | true  ->   
                        let loaded = torch.load(filePath)
                        param.copy_(loaded) |> ignore<torch.Tensor> //Side effect: model updating
                | false ->
                        printfn "File not found: %s" filePath
            )

        model //Just added to show that this fn updates the model
                
