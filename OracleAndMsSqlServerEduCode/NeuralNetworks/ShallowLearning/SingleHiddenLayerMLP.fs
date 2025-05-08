namespace NeuralNetworks

open System
open Synapses

//pouze vyukovy kod !!!

module MLP_XOR_Synapses =

    (*
    The [2; 4; 1] architecture defines:
    Input layer: 2 neurons (not counted in layer count for MLPs).    
    Hidden layer: 4 neurons (the only hidden layer).    
    Output layer: 1 neuron.        
    *)

    // XOR dataset
    let data = 
        [
            ([0.0; 0.0], [0.0]) // (x1, x2), y
            ([0.0; 1.0], [1.0])
            ([1.0; 0.0], [1.0])
            ([1.0; 1.0], [0.0])
        ]

    // Create neural network: 2 inputs -> 4 hidden (sigmoid) -> 1 output (sigmoid)
    let createNetwork () =
        NeuralNetwork.init [2; 4; 1]

    (*
    toto uz patri mezi multiple hidden layer perceptron
    let createNetwork () =
        NeuralNetwork.init [2; 4; 4; 1]

    The [2; 4; 4; 1] architecture defines:
    Input layer: 2 neurons (not counted in layer count for MLPs).    
    First hidden layer: 4 neurons    
    Second hidden layer: 4 neurons    
    Output layer: 1 neuron.  
    *) 

    // Train the network
    let train (network : NeuralNetwork) (learningRate : float) (iterations : int) =

        let rec trainLoop iter net lossAcc =

            match iter with
            | i 
                when i >= iterations 
                -> (net, lossAcc / float iterations)

            | i ->
                let trainedNet, loss =
                    data 
                    |> List.fold 
                        (fun (n, l) (x, y)
                            ->
                            let newNet = NeuralNetwork.fit(n, learningRate, x, y)
                            let pred = NeuralNetwork.prediction(newNet, x)
                            let loss = -((y |> List.head) * log((pred |> List.head) + 1e-15) + (1.0 - (y |> List.head)) * log(1.0 - (pred |> List.head) + 1e-15))  
                            (newNet, l + loss)
                        ) (net, 0.0)
                let avgLoss = loss / float data.Length
                
                match i % 1000 with
                | 0 -> printfn "Iteration %d, Loss: %.4f" i avgLoss
                | _ -> ()
                
                trainLoop (i + 1) trainedNet avgLoss

        trainLoop 0 network 0.0

    // Predict: Classify input as 0 or 1 with pattern matching
    let predict (network : NeuralNetwork) (x : float list) =

        NeuralNetwork.prediction(network, x) 
        |> List.head
        |> function       
            | output 
                when output >= 0.5
                -> 1.0
            | _ 
                -> 0.0

    // Main function to train and test
    let run () =

        let learningRate = 0.1
        let iterations = 5000

        printfn "Training MLP for XOR (Synapses)..."
        let network = createNetwork ()
        let (trainedNetwork, avgLoss) = train network learningRate iterations

        // Test predictions
        printfn "\nXOR Predictions:"
        let correct =
            data 
            |> List.fold 
                (fun acc (inputs, yTrue) 
                    ->
                    let yPred = predict trainedNetwork inputs
                    printfn "Input: %A, True: %.0f, Predicted: %.0f" inputs (List.head yTrue) yPred
                    acc + (match abs (yPred - List.head yTrue) < 1e-10 with true -> 1 | false -> 0)
                ) 0

        let accuracy = float correct / float data.Length

        printfn "Accuracy: %.2f%%" (accuracy * 100.0)
        printfn "Average Loss: %.4f" avgLoss