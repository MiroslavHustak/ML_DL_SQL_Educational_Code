namespace NeuralNetworks

open System
open TorchSharp // a wrapper for LibTorch written in C++
open type TorchSharp.torch.nn
open type TorchSharp.torch.nn.functional

//pouze vyukovy kod !!!

module MLP_Churn_TorchSharp =

    type private ChurnRecord =
        {
            MonthlyBill: float32
            ContractLength: float32
            BiasFeature: float32
            Churn: float32
        }

    let private data =
        [
            { MonthlyBill = 50.0f; ContractLength = 12.0f; BiasFeature = 1.0f; Churn = 0.0f }
            { MonthlyBill = 80.0f; ContractLength = 6.0f; BiasFeature = 1.0f; Churn = 1.0f }
            { MonthlyBill = 60.0f; ContractLength = 24.0f; BiasFeature = 1.0f; Churn = 0.0f }
            { MonthlyBill = 90.0f; ContractLength = 3.0f; BiasFeature = 1.0f; Churn = 1.0f }
            { MonthlyBill = 55.0f; ContractLength = 18.0f; BiasFeature = 1.0f; Churn = 0.0f }
            { MonthlyBill = 70.0f; ContractLength = 9.0f; BiasFeature = 1.0f; Churn = 1.0f }
            { MonthlyBill = 45.0f; ContractLength = 15.0f; BiasFeature = 1.0f; Churn = 0.0f }
        ]

    let private bills = data |> List.map (fun r -> r.MonthlyBill)
    let private lengths = data |> List.map (fun r -> r.ContractLength)
    let private minBill, maxBill = List.min bills, List.max bills
    let private minLength, maxLength = List.min lengths, List.max lengths

    let private normalizeValue value minVal maxVal =
        match minVal = maxVal with
        | true -> 0.0f
        | false -> (value - minVal) / (maxVal - minVal)

    let private inputTensor =
        let inputData =
            data
            |> List.collect (fun r -> 
                [
                    normalizeValue r.MonthlyBill minBill maxBill
                    normalizeValue r.ContractLength minLength maxLength
                    r.BiasFeature
                ])
            |> Array.ofList
        torch.tensor(inputData, dtype = TorchSharp.torch.ScalarType.Float32).reshape([| 7L; 3L |])

    let private targetTensor =
        data
        |> List.map (fun r -> r.Churn)
        |> Array.ofList
        |> fun arr -> torch.tensor(arr, dtype = TorchSharp.torch.ScalarType.Float32).reshape([| 7L; 1L |])

    type private ChurnMLP() =
        inherit Module("ChurnMLP")
        let fc1 = Linear(3L, 8L)
        let fc2 = Linear(8L, 4L)
        let fc3 = Linear(4L, 1L)
        do base.RegisterComponents()

        member _.forward(input: TorchSharp.torch.Tensor) =
            input
            |> fc1.forward
            |> relu
            |> fc2.forward
            |> relu
            |> fc3.forward
            |> sigmoid

    let private train (model: ChurnMLP) (learningRate: float) (iterations: int) =
        let optimizer = TorchSharp.torch.optim.SGD(model.parameters(), learningRate)

        let rec trainLoop iter lossAcc =
            match iter >= iterations with
            | true -> (model, lossAcc / float32 iterations)
            | false ->
                optimizer.zero_grad()
                use output = model.forward(inputTensor)
                use loss = functional.binary_cross_entropy(output, targetTensor)
                loss.backward()
                optimizer.step() |> ignore
                let batchLoss = loss.item<float32>()
                match iter % 1000 with
                | 0 -> printfn "Iteration %d, Loss: %.4f" iter batchLoss
                | _ -> ()
                trainLoop (iter + 1) (lossAcc + batchLoss)

        model.train()
        trainLoop 0 0.0f

    let private predict (model: ChurnMLP) (bill: float32) (length: float32) (bias: float32) =
        model.eval()
        let input =
            [| 
                normalizeValue bill minBill maxBill
                normalizeValue length minLength maxLength
                bias 
            |]
            |> torch.tensor
            |> fun t -> t.reshape([| 1L; 3L |])

        use output = model.forward(input)
        let prob = output.item<float32>()
        let label =
            match prob with
            | p when p >= 0.5f -> 1.0
            | _ -> 0.0
        (prob, label)

    let private computeAUC (model: ChurnMLP) =
        let sortedProbs =
            data
            |> List.map (fun r -> 
                let (prob, _) = predict model r.MonthlyBill r.ContractLength r.BiasFeature
                (prob, float r.Churn))
            |> List.sortByDescending fst

        let pos = float (sortedProbs |> List.filter (fun (_, y) -> y = 1.0) |> List.length)
        let neg = float (sortedProbs |> List.filter (fun (_, y) -> y = 0.0) |> List.length)

        let foldAUC (tpr, fpr, auc, prevTpr, prevFpr) (prob, y) =
            match y with
            | 1.0 ->
                let newTpr = tpr + 1.0 / pos
                (newTpr, fpr, auc, newTpr, fpr)
            | _ ->
                let newFpr = fpr + 1.0 / neg
                let newAUC = auc + (tpr + prevTpr) * (newFpr - prevFpr) / 2.0
                (tpr, newFpr, newAUC, tpr, newFpr)

        let (_, _, auc, _, _) =
            sortedProbs
            |> List.fold foldAUC (0.0, 0.0, 0.0, 0.0, 0.0)

        auc

    let run () =
        torch.random.manual_seed(42L) |> ignore
        let learningRate = 0.1
        let iterations = 5000

        printfn "Training MLP for Churn Prediction (TorchSharp)..."
        use model = new ChurnMLP()
        let (trainedModel, avgLoss) = train model learningRate iterations

        let accuracy =
            data
            |> List.sumBy (fun r ->
                let (_, yPred) = predict trainedModel r.MonthlyBill r.ContractLength r.BiasFeature
                match yPred = float r.Churn with
                | true -> 1.0
                | false -> 0.0)
            |> fun sum -> sum / float data.Length

        let auc = computeAUC trainedModel

        printfn "\nModel Evaluation Metrics:"
        printfn "  Log Loss: %.4f" avgLoss
        printfn "  Accuracy: %.4f" accuracy
        printfn "  Area Under ROC Curve (AUC): %.4f" auc

        printfn "\nChurn Predictions:"
        data 
        |> List.iter 
            (fun r 
                -> 
                let (prob, yPred) = predict trainedModel r.MonthlyBill r.ContractLength r.BiasFeature
                printfn "Input: [Bill=%.0f, Length=%.0f], True: %.0f, Predicted: %.0f, Probability: %.4f"
                    r.MonthlyBill r.ContractLength r.Churn yPred prob
            )

        let testCustomer = { MonthlyBill = 75.0f; ContractLength = 8.0f; BiasFeature = 1.0f; Churn = 0.0f }
        let (testProb, testLabel) = predict trainedModel testCustomer.MonthlyBill testCustomer.ContractLength testCustomer.BiasFeature

        printfn "\nNew Customer Prediction:"
        printfn "  MonthlyBill: %.2f, ContractLength: %.2f" testCustomer.MonthlyBill testCustomer.ContractLength
        printfn "  Churn Probability: %.4f" testProb
        printfn "  Prediction: %s" (
            match testLabel with
            | 1.0 -> "Customer will churn"
            | _ -> "Customer will not churn")
