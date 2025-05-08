namespace MachineLearning

open System

open Microsoft.ML
open Microsoft.ML.Data
open Microsoft.ML.Trainers
open Microsoft.ML.Calibrators

//pouze vyukovy kod !!!

module ManualLogisticRegression =

    type Customer =
        {
            MonthlyBill : float
            ContractLength : float
            BiasFeature : float
            Churn : float // 1.0 = churn, 0.0 = no churn
        }

    let trainAndPredictManual () =

        let sigmoid z = 1.0 / (1.0 + exp (-z))

        let features customer = [ customer.MonthlyBill; customer.ContractLength; customer.BiasFeature ]

        let dotProduct (weights : float list) (features : float list) =

            List.fold2 (fun acc w f -> acc + w * f) 0.0 weights features

        let predict (weights : float list) customer = sigmoid (dotProduct weights (features customer))

        let logLoss (weights : float list) customer =

            let y = customer.Churn
            let p = predict weights customer
            -(y * log p + (1.0 - y) * log (1.0 - p))

        let gradient (weights : float list) customer =

            let y = customer.Churn
            let p = predict weights customer
            let error = p - y
            
            features >> List.map (fun f -> error * f) <| customer

        let updateWeights (weights : float list) (gradients : float list) learningRate =

            List.map2 (fun w g -> w - learningRate * g) weights gradients

        let train (data : Customer list) learningRate iterations =

            let featureCount = 3
            let initialWeights = List.init featureCount (fun _ -> 0.0)
    
            let rec trainIter weights iter =
                match iter with
                | 0 -> 
                    weights

                | _ ->
                    let gradients =
                        data
                        |> List.map (gradient weights)
                        |> List.fold (fun acc g -> List.map2 (+) acc g) (List.init featureCount (fun _ -> 0.0))
                        |> List.map (fun g -> g / float data.Length)
                    
                    let newWeights = updateWeights weights gradients learningRate

                    trainIter newWeights (iter - 1)
    
            trainIter initialWeights iterations

        let trainModel (data : Customer list) = train data 0.001 10000

        let predictChurn weights customer = predict weights customer

        let data =
            [
                { MonthlyBill = 50.0; ContractLength = 12.0; BiasFeature = 1.0; Churn = 0.0 }
                { MonthlyBill = 80.0; ContractLength = 6.0; BiasFeature = 1.0; Churn = 1.0 }
                { MonthlyBill = 60.0; ContractLength = 24.0; BiasFeature = 1.0; Churn = 0.0 }
                { MonthlyBill = 90.0; ContractLength = 3.0; BiasFeature = 1.0; Churn = 1.0 }
                { MonthlyBill = 55.0; ContractLength = 18.0; BiasFeature = 1.0; Churn = 0.0 }
                { MonthlyBill = 70.0; ContractLength = 9.0; BiasFeature = 1.0; Churn = 1.0 }
                { MonthlyBill = 45.0; ContractLength = 15.0; BiasFeature = 1.0; Churn = 0.0 }
            ]
    
        let weights = trainModel data
    
        printfn "Trained Weights:"
        printfn "  MonthlyBill: %.4f" (weights |> List.head)
        printfn "  ContractLength: %.4f" (weights |> List.item 1)
        printfn "  BiasFeature: %.4f" (weights |> List.item 2)
    
        printfn "\nPer-Customer Results:"

        data
        |> List.iteri
            (fun i customer 
                ->
                let prob = predictChurn weights customer
                let loss = logLoss weights customer

                printfn "Customer %d:" (i + 1)
                //printfn "  MonthlyBill: %.2f, ContractLength: %.2f" customer.MonthlyBill customer.ContractLength
                //printfn "  Actual Churn: %.0f" customer.Churn
                printfn "  Predicted Churn Probability: %.4f" prob
                printfn "  Log Loss: %.4f" loss
        )
    
        let avgLogLoss =
            data
            |> List.map (logLoss weights)
            |> List.average
        
        printfn "\nAverage Log Loss: %.4f" avgLogLoss
    
        // Prediction for a new customer
        let testCustomer = { MonthlyBill = 75.0; ContractLength = 8.0; BiasFeature = 1.0; Churn = 0.0 }
        let testProb = predictChurn weights testCustomer
        
        printfn "\nNew Customer Prediction:"
        //printfn "  MonthlyBill: %.2f, ContractLength: %.2f" testCustomer.MonthlyBill testCustomer.ContractLength
        printfn "  Churn Probability: %.4f" testProb
        
        match testProb >= 0.5 with
        | true  -> printfn "  Prediction: Customer will churn"
        | false -> printfn "  Prediction: Customer will not churn"

//************************************************************************

 module MLNETLogisticRegression = 
        
    [<CLIMutable>]
    type Customer = 
        {
            [<ColumnName("MonthlyBill")>] MonthlyBill : float32
            [<ColumnName("ContractLength")>] ContractLength : float32
            [<ColumnName("Label")>] Churn : bool // ML.NET expects bool for binary classification
        }
    
    [<CLIMutable>]
    type Prediction =
        {
            [<ColumnName("PredictedLabel")>] PredictedLabel : bool
            [<ColumnName("Probability")>] Probability : float32
        }

    let trainAndPredictML_NET () =    
    
        // Convert float-based data to ML.NET-compatible data
        let convertData (data : ManualLogisticRegression.Customer list) : Customer list =
            data
            |> List.map
                (fun c 
                    ->
                    { 
                        MonthlyBill = float32 c.MonthlyBill
                        ContractLength = float32 c.ContractLength
                        Churn =
                            match c.Churn with
                            | 1.0 -> true
                            | 0.0 -> false
                            | _   -> failwith "Invalid Churn value" 
                    }
                )
    
        // Train logistic regression model and create a prediction engine
        let train (data : Customer list) =

            let mlContext = MLContext()
            let dataView = mlContext.Data.LoadFromEnumerable data
        
            let pipeline = 
                EstimatorChain()
                    .Append(mlContext.Transforms.Concatenate("Features", [| "MonthlyBill"; "ContractLength" |]))
                    .Append(mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression())
        
            let model = pipeline.Fit dataView

             // Attempt to extract weights (handling calibrated models) //tohle neber prilis vazne
            let tryGetWeights () =
                try
                    let lastTransformer = model.LastTransformer
                    let calibratedModel = lastTransformer.Model
                    let linearModel = calibratedModel.SubModel
                    let weights = linearModel.Weights |> Seq.toList
                    let bias = linearModel.Bias

                    Some (weights, bias)                  
                with
                | _ -> None

            match tryGetWeights () with
            | Some (weights, bias) 
                ->
                printfn "Trained Weights:"
                printfn "  MonthlyBill: %.4f" (weights |> List.head)
                printfn "  ContractLength: %.4f" (weights |> List.item 1)
                printfn "  Bias: %.4f" bias

            | None 
                ->
                printfn "Warning: Could not extract weights from the model. Model structure may not support direct weight access."

            let predictionEngine = mlContext.Model.CreatePredictionEngine<Customer, Prediction> model
            (mlContext, model, predictionEngine)
    
        // Predict churn for a new customer
        // Note: predictionEngine is not thread-safe; synchronize access if used concurrently
        let predictChurn (mlContext : MLContext, model : ITransformer, predictionEngine : PredictionEngine<Customer, Prediction>) (customer : Customer) =
            
            let prediction = predictionEngine.Predict(customer)
            (prediction.Probability, prediction.PredictedLabel)

        // Dataset (using float32 and bool)
        let data =
            [
                { MonthlyBill = 50.0f; ContractLength = 12.0f; Churn = false }
                { MonthlyBill = 80.0f; ContractLength = 6.0f; Churn = true }
                { MonthlyBill = 60.0f; ContractLength = 24.0f; Churn = false }
                { MonthlyBill = 90.0f; ContractLength = 3.0f; Churn = true }
                { MonthlyBill = 55.0f; ContractLength = 18.0f; Churn = false }
                { MonthlyBill = 70.0f; ContractLength = 9.0f; Churn = true }
                { MonthlyBill = 45.0f; ContractLength = 15.0f; Churn = false }
            ]
    
        let (mlContext, model, predictionEngine) = train data
    
        printfn "ML.NET Logistic Regression Results:"
        
        data
        |> List.iteri
            (fun i customer 
                ->
                let (prob, label) = predictChurn (mlContext, model, predictionEngine) customer
                
                printfn "Customer %d:" (i + 1)
                //printfn "  MonthlyBill: %.2f, ContractLength: %.2f" customer.MonthlyBill customer.ContractLength
                //printfn "  Actual Churn: %b" customer.Churn
                printfn "  Predicted Churn: %b" label
                printfn "  Churn Probability: %.4f" prob
            )
    
        let predictions = model.Transform(mlContext.Data.LoadFromEnumerable data)
        let metrics = mlContext.BinaryClassification.Evaluate(predictions, labelColumnName = "Label")
        
        printfn "\nModel Evaluation Metrics:"
        printfn "  Log Loss: %.4f" metrics.LogLoss
        printfn "  Accuracy: %.4f" metrics.Accuracy
        printfn "  Area Under ROC Curve (AUC): %.4f" metrics.AreaUnderRocCurve
    
        // Predict for a new customer
        let testCustomer = { MonthlyBill = 75.0f; ContractLength = 8.0f; Churn = false }
        let (testProb, testLabel) = predictChurn (mlContext, model, predictionEngine) testCustomer
        
        printfn "\nNew Customer Prediction:"
        //printfn "  MonthlyBill: %.2f, ContractLength: %.2f" testCustomer.MonthlyBill testCustomer.ContractLength
        printfn "  Churn Probability: %.4f" testProb
        printfn "  Prediction: %s" (match testLabel with true -> "Customer will churn" | false -> "Customer will not churn")