namespace NeuralNetworks

open System
open Synapses

//pouze vyukovy kod !!!

module MLP_Churn_Synapses =

    type private ChurnRecord =
        {
            MonthlyBill : float
            ContractLength : float
            BiasFeature : float
            Churn : float
        }

    let private data =
        [
            { MonthlyBill = 50.0; ContractLength = 12.0; BiasFeature = 1.0; Churn = 0.0 }
            { MonthlyBill = 80.0; ContractLength = 6.0; BiasFeature = 1.0; Churn = 1.0 }
            { MonthlyBill = 60.0; ContractLength = 24.0; BiasFeature = 1.0; Churn = 0.0 }
            { MonthlyBill = 90.0; ContractLength = 3.0; BiasFeature = 1.0; Churn = 1.0 }
            { MonthlyBill = 55.0; ContractLength = 18.0; BiasFeature = 1.0; Churn = 0.0 }
            { MonthlyBill = 70.0; ContractLength = 9.0; BiasFeature = 1.0; Churn = 1.0 }
            { MonthlyBill = 45.0; ContractLength = 15.0; BiasFeature = 1.0; Churn = 0.0 }
        ]

    // Normalize features to [0, 1]
    let private normalize (values : float list) =

        let minVal = values |> List.min
        let maxVal = values |> List.max
        let range = maxVal - minVal

        match range = 0.0 with
        | true  -> values
        | false -> values |> List.map (fun x -> (x - minVal) / range)

    let private normalizedData =
        let bills = data |> List.map (fun r -> r.MonthlyBill) |> normalize
        let lengths = data |> List.map (fun r -> r.ContractLength) |> normalize
        data |> List.mapi (fun i r -> ([bills |> List.item i; lengths |> List.item i; r.BiasFeature], [r.Churn]))

    // Create neural network: 3 inputs -> 8 hidden -> 4 hidden -> 1 output (sigmoid)
    let private createNetwork () =
        NeuralNetwork.init [3; 8; 4; 1]

    (*
    NeuralNetwork.init [3; 18; 9; 3; 1]
    Potential Downsides
    Overfitting: With such a small dataset (only 7 samples), this larger network might memorise the data rather than generalise.
    
    Slower training: More parameters mean more computation per iteration.
    
    Vanishing gradients (rare here but theoretically more likely as depth increases).   
    
    *)

    // Train the network
    let private train (network : NeuralNetwork) (learningRate : float) (iterations : int) =

        let rec trainLoop iter net lossAcc =

            match iter with
            | i 
                when i >= iterations 
                -> (net, lossAcc / float iterations)
            | i 
                ->
                let trainedNet, batchLoss =
                    normalizedData 
                    |> List.fold 
                        (fun (n, l) (x, y) 
                            ->
                            let newNet = NeuralNetwork.fit(n, learningRate, x, y)
                            let pred = NeuralNetwork.prediction(newNet, x)
                            let loss = -((y |> List.head) * log((pred |> List.head) + 1e-15) + (1.0 - (y |> List.head)) * log(1.0 - (pred |> List.head) + 1e-15))                        
                            (newNet, l + loss)
                        ) (net, 0.0)
    
                let avgBatchLoss = batchLoss / float normalizedData.Length

                match i % 1000 with
                | 0 -> printfn "Iteration %d, Loss: %.4f" i avgBatchLoss
                | _ -> ()
    
                // Accumulate full loss
                trainLoop (i + 1) trainedNet (lossAcc + avgBatchLoss)
    
        trainLoop 0 network 0.0

    // Predict: Classify as churn (1) or no churn (0)
    let private predict (network : NeuralNetwork) (bill : float) (length : float) (bias : float) =
        // Normalize input
        let bills = data |> List.map (fun r -> r.MonthlyBill)
        let lengths = data |> List.map (fun r -> r.ContractLength)
        let normBill = normalize bills |> List.item (bills |> List.findIndex ((=) bill))
        let normLength = normalize lengths |> List.item (lengths |> List.findIndex ((=) length))
        let input = [normBill; normLength; bias]
        
        NeuralNetwork.prediction(network, input) 
        |> List.head
        |> function
            | output 
                when output >= 0.5 
                -> 1.0
            | _ 
                -> 0.0

    let run () =

        let learningRate = 0.1
        let iterations = 5000

        printfn "Training MLP for Churn Prediction (Synapses)..."
        let network = createNetwork ()
        let (trainedNetwork, avgLoss) = train network learningRate iterations

        // Test predictions
        printfn "\nChurn Predictions:"
        let correct =
            data 
            |> List.fold 
                (fun acc r
                    ->
                    let yPred = predict trainedNetwork r.MonthlyBill r.ContractLength r.BiasFeature
                    printfn "Input: [Bill=%.0f, Length=%.0f], True: %.0f, Predicted: %.0f" r.MonthlyBill r.ContractLength r.Churn yPred
                    acc + (match abs (yPred - r.Churn) < 1e-10 with true -> 1 | false -> 0)
                ) 0

        // Accuracy
        let accuracy = float correct / float data.Length
        printfn "Accuracy: %.2f%%" (accuracy * 100.0)
        printfn "Average Loss: %.4f" avgLoss

        // Prediction for a new customer
        let newBill = 75.0
        let newLength = 8.0
        let newBias = 1.0
        let prob =
            let bills = data |> List.map (fun r -> r.MonthlyBill)
            let lengths = data |> List.map (fun r -> r.ContractLength)
            let minBill, maxBill = List.min bills, List.max bills
            let minLength, maxLength = List.min lengths, List.max lengths            
            let normBill = (newBill - minBill) / (maxBill - minBill)
            let normLength = (newLength - minLength) / (maxLength - minLength)            
            let input = [normBill; normLength; newBias]

            NeuralNetwork.prediction(trainedNetwork, input) |> List.head

        let predicted =
            match prob >= 0.5 with true -> "Customer will churn" | false -> "Customer will not churn"

        printfn "\nNew Customer Prediction:"
        printfn "  MonthlyBill: %.2f, ContractLength: %.2f" newBill newLength
        printfn "  Churn Probability: %.4f" prob
        printfn "  Prediction: %s" predicted
