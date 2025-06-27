namespace NeuralNetwork

module LoRa =

    open System
    open TorchSharp
    open TorchSharp.Modules
    
    open type TorchSharp.torch
    open type TorchSharp.torch.nn

    type LoRALinear(inFeatures: int64, outFeatures: int64, rank: int64, alpha: float32, device: Device) as this =
        inherit Module("LoRALinear")

        // Main weights
        let weight = torch.empty([| outFeatures; inFeatures |], device = device).AsParameter()
        let bias = torch.zeros(outFeatures, device = device).AsParameter()

        // LoRA weights
        let A = torch.empty([| rank; inFeatures |], device = device).AsParameter()
        let B = torch.empty([| outFeatures; rank |], device = device).AsParameter()

        let scaling = alpha / (Microsoft.FSharp.Core.Operators.float32 rank)

        do
            this.register_parameter("weight", weight) |> ignore
            this.register_parameter("bias", bias) |> ignore
            this.register_parameter("A", A) |> ignore
            this.register_parameter("B", B) |> ignore

            init.kaiming_uniform_(weight) |> ignore
            init.zeros_(bias) |> ignore
            init.kaiming_uniform_(A) |> ignore
            init.zeros_(B) |> ignore

        member this.forward(input: Tensor) : Tensor =
            let lora = B.matmul(A).mul(torch.tensor(scaling, dtype = float32, device = device))
            let effectiveWeight = weight.add(lora)
            input.matmul(effectiveWeight.transpose(0L, 1L)).add(bias)
