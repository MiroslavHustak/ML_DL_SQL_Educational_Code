namespace NeuralNetworks

open System.Collections.Generic

module LoRA =
    
    open TorchSharp
    open type torch.nn
    
    //LoRA Low-Rank Adaptation
    type LoRALinear(inFeatures: int64, outFeatures: int64, rank: int64, alpha: float32, device: torch.Device) as this =

        inherit Module<torch.Tensor, torch.Tensor>("LoRALinear")
    
        // Main weights (allocated on specified device)
        let weight = torch.empty([| outFeatures; inFeatures |], device = device).AsParameter()
        let bias = torch.zeros(outFeatures, device = device).AsParameter()
            
        // LoRA weights (A: random small init, B: zeros, per LoRA paper, allocated on specified device)
        let A : Modules.Parameter = (torch.randn([| rank; inFeatures |], device = device) * 0.01f).AsParameter()
        let B : Modules.Parameter = torch.zeros([| outFeatures; rank |], device = device).AsParameter()
    
        let scaling = alpha / (Microsoft.FSharp.Core.Operators.float32 rank)
        let scalingTensor = torch.tensor(scaling, dtype = torch.float32, device = device).AsParameter()
    
        do
            this.register_parameter("weight", weight) |> ignore<unit>
            this.register_parameter("bias", bias) |> ignore<unit>
            this.register_parameter("A", A) |> ignore<unit>
            this.register_parameter("B", B) |> ignore<unit>
            this.register_parameter("scaling", scalingTensor) |> ignore<unit>

            A.requires_grad <- true
            B.requires_grad <- true

            init.kaiming_uniform_(weight) |> ignore<torch.Tensor>
            init.zeros_(bias) |> ignore<torch.Tensor>
            // LoRA: A is already initialized above, B is already zeros    
            
        override this.forward(input: torch.Tensor) =

           let projA = input.matmul(A.transpose(0L, 1L)) // [batch, rank]
           let projB = projA.matmul(B.transpose(0L, 1L)) // [batch, outF]
           let scaled = projB.mul(scalingTensor.to_type(input.dtype))
       
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