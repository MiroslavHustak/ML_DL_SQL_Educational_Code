namespace MachineLearning


open TorchSharp
open TorchSharp.Modules

open type torch.nn
open type torch.nn.functional

open type TorchSharp.TensorExtensionMethods

//*******************************************************************
(* 
This code is still under development and is intended for educational purposes only.
And yes, I know that TorchSharp is overkill for linear regression.

Option and Result types have not been applied yet.
Separation of data and data manipulation not fully implemented yet.
Disposals not fully implemented yet.
The code architecture will be figured out later.
*)
//*******************************************************************

module TorchLinearRegression = 

    let torchSharpLR () = 

        let learningRate = 0.01
        let epochs = 2000
        let samples = 100000
    
        // Generate synthetic data: y = 2 * x1 + 3 * x2 + noise
        let generateMockData numSamples =

            let noise = torch.randn([| int64 numSamples; 1L |]) * 0.1f
            let x1 = torch.rand([| int64 numSamples; 1L |]) * 10.0f
            let x2 = torch.rand([| int64 numSamples; 1L |]) * 10.0f
            let y = 2.0f * x1 + 3.0f * x2 + noise
            (torch.concat([|x1; x2|], 1), y)
    
        // 1) Define the model: Single Linear Layer
        let model = torch.nn.Linear(2L, 1L)

        // 2) Nutno vzdy dumat, jaku optimizer zvolit
        let optimizer = torch.optim.SGD(model.parameters(), learningRate)  
    
        // 3) Volba loss function Mean Squared Error Loss
        let criterion = torch.nn.functional.mse_loss
    
        let train (x: torch.Tensor) (y: torch.Tensor) =

            // model.zero_grad() neni tady striktne nutna because of the single call, ale staci volat train podruhe a uz nutna je.
            model.zero_grad() //ensures that any existing gradients from previous iterations or function calls are cleared

            [1..epochs]
            |> List.iter 
                (fun epoch
                    ->
                    optimizer.zero_grad() //Clear old gradients  //Resets the gradients of all parameters managed by the optimizer to zero.                          
                    let output = model.forward x //Compute predictions (weighs, biases)          
                    let loss = criterion(output, y) //Compute loss (tady metoda nejmensich ctvercu)

                    //The optimizer and backpropagation steps alter (mutate under the hood) the state of the model.
                    loss.backward()  //Compute new gradients
                    optimizer.step() |> ignore<torch.Tensor>  //Update parameters using gradients

                    match epoch % 100 with
                    | 0 -> printfn "Epoch %d - Loss: %f" epoch (loss.item<float32>())
                    | _ -> ()

                    System.GC.Collect()
            )
    
        // Generate Data
        let (x, y) = generateMockData samples
    
        // Train the model
        train x y  
    
        // Print the learned parameters (weights and bias)
        let weights = model.weight.detach().``to``(torch.float32).reshape(-1)
        let bias = model.bias.detach().``to``(torch.float32)       
        
        printfn "Learned Weights: %A" (weights.``to``(torch.float32).data<float32>()) 
        printfn "Learned Bias: %A" (bias.``to``(torch.float32).data<float32>()) 

        // Make a prediction for input [5.0; 5.0]
        let newInput = torch.tensor([| 19.0f; 15.0f |]).reshape([| 1L; 2L |])
        let prediction = model.forward newInput
        printfn "Prediction for [5.0; 5.0]: %f" (prediction.item<float32>())

        //TODO: Result type
        model.save("tutorial6.model.bin") |> ignore<torch.nn.Module> //saving the weights and biases, not the model itself !!!
         
        //*********************************************************************** 
        let model1 = torch.nn.Linear(2L, 1L) //musi byt stejny typ/struktura, jako u modelu, ktery byl ulozen

        //TODO: Result type
        model1.load("tutorial6.model.bin") |> ignore<torch.nn.Module> 

        // Print the learned parameters (weights and bias)
        let weights1 = model1.weight.detach().``to``(torch.float32).reshape(-1)
        let bias1 = model1.bias.detach().``to``(torch.float32)       
        
        printfn "Learned Weights1: %A" (weights1.``to``(torch.float32).data<float32>()) 
        printfn "Learned Bias1: %A" (bias1.``to``(torch.float32).data<float32>()) 
    
        // Make a prediction for input [5.0; 5.0]
        let newInput1 = torch.tensor([| 19.0f; 15.0f |]).reshape([| 1L; 2L |])
        let prediction1 = model1.forward newInput1
        printfn "Prediction1 for [5.0; 5.0]: %f" (prediction1.item<float32>())

//For experimenting only!!!!
module TorchLinearRegressionSequential =

    let learningRate = 0.001 // Reduced for better convergence
    let epochs = 3000
    let samples = 100000

    // Generate synthetic data: y = 2 * x1 + 3 * x2 + noise
    let generateMockData numSamples =

        let noise = torch.randn([| int64 numSamples; 1L |]) * 0.1f
        let x1 = torch.rand([| int64 numSamples; 1L |]) * 10.0f
        let x2 = torch.rand([| int64 numSamples; 1L |]) * 10.0f
        let y = 2.0f * x1 + 3.0f * x2 + noise
        (torch.cat([|x1; x2|], 1), y)

    // For educational purposes only 
    // Overkill for a linear regression
    // Define the model using nn.Sequential for a clean, stackable architecture 
    let createModel () =

        torch.nn.Sequential(
            Linear(2L, 10L),
            ReLU(),  //ReLU (Rectified Linear Unit) is a non-linear activation function //akorat to zhorsuje, ale je to tady jen pro learning purposes
            Linear(10L, 5L),
            ReLU(),
            Linear(5L, 1L)
        )

    let createModel2 () = //GELU to nepatrne zlepsi, ale stale je to overparameterised 
    
            torch.nn.Sequential(
                Linear(2L, 10L),
                GELU(),  
                Linear(10L, 5L),
                GELU(),
                Linear(5L, 1L)
            )

    let train (model: Sequential) (x: torch.Tensor) (y: torch.Tensor) epochs learningRate =
        
        //use optimizer : SGD = torch.optim.SGD(model.parameters(), learningRate)  
        use optimizer : Adam = torch.optim.Adam(model.parameters(), learningRate)

        //Learning Rate Scheduler - s tim je treba experimentovat, jestli to pomuze, v mem pripade ne
        //let scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 25, 0.95, verbose=true)
        
        let criterion = torch.nn.functional.mse_loss

        [1..epochs]
        |> List.iter
            (fun epoch
                ->
                optimizer.zero_grad()
                let output : torch.Tensor = model.forward x
                let loss = criterion(output, y)
                loss.backward()
                optimizer.step() |> ignore<torch.Tensor>

                match epoch % 100 with
                | 0 -> printfn "Epoch %d - Loss: %f" epoch (loss.item<float32>())
                | _ -> ()

                (*
                // Step the learning rate every 25 epochs as defined in the scheduler
                match epoch % 25 = 0 with
                | true  -> scheduler.step()
                | false -> ()
                *)

                System.GC.Collect()
            )

    let torchSharpLR () =        

        // Generate Data
        let (x, y) = generateMockData samples

        // Create and train the model
        //let model = createModel()
        let model = createModel2()
        train model x y epochs learningRate  

        // Print learned parameters (weights and biases)
        model.named_children()
        |> Seq.iter
            (fun (struct (name, module_)) //u for loop by to proslo bez struct, tam neni az takova type safety
                ->
                match module_ with
                | :? Linear as linear
                    ->
                    use weights = linear.weight.detach().``to``(torch.float32)
                    use bias = linear.bias.detach().``to``(torch.float32)
                    
                    let count = weights.data<float32>() |> Seq.length
                    printfn "Layer %s Weight count: %i" name count
                    printfn "Layer %s Weights: %A" name (weights.data<float32>())
                    printfn "Layer %s Bias: %A" name (bias.data<float32>())
                    //the weights and biases not resembling the expected [2.0; 3.0] 
                    //from the data-generating function y = 2 * x1 + 3 * x2 + noise) is expected 
                    //due to the non-linear, overparameterized nature of the Sequential model.                     
                | _ 
                    ->
                    () // Skip non-Linear layers (e.g., ReLU)        
        )       

        // TODO: Result type
        model.save("tutorial6_sequential.model.bin") |> ignore<torch.nn.Module>

        // Reload the model for inference
        //let model1 = createModel()
        let model1 = createModel2()

        // TODO: Result type
        model1.load("tutorial6_sequential.model.bin") |> ignore<torch.nn.Module>

        // Make a prediction for input [5.0; 5.0]
        let newInput = torch.tensor([| 19.0f; 15.0f |]).reshape([| 1L; 2L |])
        let prediction = model1.forward newInput
        printfn "Prediction for [5.0; 5.0]: %f" (prediction.item<float32>())