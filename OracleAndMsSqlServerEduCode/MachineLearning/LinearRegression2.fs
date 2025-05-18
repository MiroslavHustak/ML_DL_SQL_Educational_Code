namespace MachineLearning


open TorchSharp
open TorchSharp.Modules

open type torch.nn
open type torch.nn.functional

open type TorchSharp.TensorExtensionMethods


module TorchLinearRegression = 

    let torchSharpLR () = 

        let learningRate = 0.01
        let epochs = 1000
        let samples = 100000
    
        // Generate synthetic data: y = 2 * x1 + 3 * x2 + noise
        let generateMockData numSamples =
            let noise = torch.randn([| int64 numSamples; 1L |]) * 0.1f
            let x1 = torch.rand([| int64 numSamples; 1L |]) * 10.0f
            let x2 = torch.rand([| int64 numSamples; 1L |]) * 10.0f
            let y = 2.0f * x1 + 3.0f * x2 + noise
            (torch.cat([|x1; x2|], 1), y)
    
        // Define the model: Single Linear Layer
        let model = torch.nn.Linear(2L, 1L)
        let optimizer = torch.optim.SGD(model.parameters(), learningRate)
    
        // Mean Squared Error Loss
        let criterion = torch.nn.functional.mse_loss
    
        // Training loop (Functional Style)
        let train (X: torch.Tensor) (y: torch.Tensor) =
            [1..epochs]
            |> List.iter (fun epoch ->
                optimizer.zero_grad()
                let output = model.forward(X)
                let loss = criterion(output, y)
                loss.backward()
                optimizer.step() |> ignore<torch.Tensor>
    
                match epoch % 100 with
                | 0 -> printfn "Epoch %d - Loss: %f" epoch (loss.item<float32>())
                | _ -> ()
            )
    
        // Generate Data
        let (X, y) = generateMockData samples
    
        // Train the model
        train X y
    
        // Print the learned parameters (weights and bias)
        let weights = model.weight.detach().``to``(torch.float32).reshape(-1)
        let bias = model.bias.detach().``to``(torch.float32)
        
        printfn "Learned Weights: %A" (weights.``to``(torch.float32).data<float32>()) 
        printfn "Learned Bias: %A" (bias.``to``(torch.float32).data<float32>()) 
    
        // Make a prediction for input [5.0; 5.0]
        let newInput = torch.tensor([| 5.0f; 5.0f |]).reshape([| 1L; 2L |])
        let prediction = model.forward(newInput)
        printfn "Prediction for [5.0; 5.0]: %f" (prediction.item<float32>())
    