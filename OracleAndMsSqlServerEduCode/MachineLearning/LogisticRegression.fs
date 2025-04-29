namespace MachineLearning

open System

module ChurnPrediction =

    type private Customer = {
        MonthlyBill : float
        ContractLength : float
        BiasFeature : float
        Churn : float // 1.0 = churn, 0.0 = no churn
    }

    let private data = [
        { MonthlyBill = 50.0; ContractLength = 12.0; BiasFeature = 1.0; Churn = 0.0 }
        { MonthlyBill = 80.0; ContractLength = 6.0; BiasFeature = 1.0; Churn = 1.0 }
        { MonthlyBill = 60.0; ContractLength = 24.0; BiasFeature = 1.0; Churn = 0.0 }
        { MonthlyBill = 90.0; ContractLength = 3.0; BiasFeature = 1.0; Churn = 1.0 }
        { MonthlyBill = 55.0; ContractLength = 18.0; BiasFeature = 1.0; Churn = 0.0 }
        { MonthlyBill = 70.0; ContractLength = 9.0; BiasFeature = 1.0; Churn = 1.0 }
        { MonthlyBill = 45.0; ContractLength = 15.0; BiasFeature = 1.0; Churn = 0.0 }
    ]

    let private sigmoid (z: float) : float = 1.0 / (1.0 + exp(-z))

    let private normalizeWith (minVal: float) (maxVal: float) (x: float) : float =
        if maxVal - minVal = 0.0 then 0.0 else (x - minVal) / (maxVal - minVal)

    let private normalizeData (data : Customer list) =
        let bills = data |> List.map (fun c -> c.MonthlyBill)
        let contracts = data |> List.map (fun c -> c.ContractLength)
        let minBill, maxBill = List.min bills, List.max bills
        let minContract, maxContract = List.min contracts, List.max contracts

        let normalizedData =
            data
            |> List.map (fun c -> {
                c with
                    MonthlyBill = normalizeWith minBill maxBill c.MonthlyBill
                    ContractLength = normalizeWith minContract maxContract c.ContractLength
            })

        (normalizedData, minBill, maxBill, minContract, maxContract)

    let private predict (beta0 : float) (beta1 : float) (beta2 : float) (customer : Customer) : float =
        let z = beta0 * customer.BiasFeature + beta1 * customer.MonthlyBill + beta2 * customer.ContractLength
        sigmoid z

    let private logLoss (beta0 : float) (beta1 : float) (beta2 : float) (data : Customer list) (lambda: float) : float =
        let epsilon = 1e-15
        let loss =
            data
            |> List.map (fun customer ->
                let y = customer.Churn
                let p = predict beta0 beta1 beta2 customer
                let clippedP = max epsilon (min (1.0 - epsilon) p)
                - (y * log clippedP + (1.0 - y) * log (1.0 - clippedP))
            )
            |> List.average
        loss + (lambda / 2.0) * (beta1 ** 2.0 + beta2 ** 2.0)

    let private computeGradients (beta0 : float) (beta1 : float) (beta2 : float) (data : Customer list) (lambda: float) =
        let grad0 = data |> List.averageBy (fun c -> (predict beta0 beta1 beta2 c - c.Churn) * c.BiasFeature)
        let grad1 = (data |> List.averageBy (fun c -> (predict beta0 beta1 beta2 c - c.Churn) * c.MonthlyBill)) + (lambda * beta1)
        let grad2 = (data |> List.averageBy (fun c -> (predict beta0 beta1 beta2 c - c.Churn) * c.ContractLength)) + (lambda * beta2)
        (grad0, grad1, grad2)

    let rec private gradientDescent (data : Customer list) beta0 beta1 beta2 learningRate iterations currentIter lambda : float * float * float =
        if currentIter >= iterations then (beta0, beta1, beta2)
        else
            let grad0, grad1, grad2 = computeGradients beta0 beta1 beta2 data lambda
            let newBeta0 = beta0 - learningRate * grad0
            let newBeta1 = beta1 - learningRate * grad1
            let newBeta2 = beta2 - learningRate * grad2
            gradientDescent data newBeta0 newBeta1 newBeta2 learningRate iterations (currentIter + 1) lambda

    let trainAndPredict () =
        let learningRate = 0.01
        let iterations = 1000
        let lambda = 0.1

        let normalizedData, minBill, maxBill, minContract, maxContract = normalizeData data

        let beta0, beta1, beta2 = gradientDescent normalizedData 0.0 0.0 0.0 learningRate iterations 0 lambda
        let loss = logLoss beta0 beta1 beta2 normalizedData lambda
        printfn "Final Log-Loss: %.4f" loss

        let newCustomer = { MonthlyBill = 75.0; ContractLength = 8.0; BiasFeature = 1.0; Churn = 0.0 }
        let normNewCustomer = {
            newCustomer with
                MonthlyBill = normalizeWith minBill maxBill newCustomer.MonthlyBill
                ContractLength = normalizeWith minContract maxContract newCustomer.ContractLength
        }

        let churnProbability = predict beta0 beta1 beta2 normNewCustomer
        printfn "Churn Probability for new customer: %.2f" churnProbability

        match churnProbability >= 0.5 with
        | true -> printfn "Prediction: Customer will churn"
        | false -> printfn "Prediction: Customer will not churn"

        printfn "Learned Weights:"
        printfn "  beta0 (bias): %.4f" beta0
        printfn "  beta1 (MonthlyBill): %.4f" beta1
        printfn "  beta2 (ContractLength): %.4f" beta2

    //************************************************************************
module ChurnPredictionMLNET = 

    open System

    open Microsoft.ML
    open Microsoft.ML.Data
    open Microsoft.ML.Trainers    

    [<CLIMutable>]
    type Customer =
        {
            [<ColumnName("MonthlyBill")>] MonthlyBill : float32
            [<ColumnName("ContractLength")>] ContractLength : float32
            [<ColumnName("Label")>] Churn : bool // ML.NET expects bool for binary classification
        }

    // Define the prediction output structure
    [<CLIMutable>]
    type ChurnPrediction = {
        [<ColumnName("PredictedLabel")>] PredictedLabel: bool
        [<ColumnName("Probability")>] Probability: float32
        [<ColumnName("Score")>] Score: float32
    }

    // Mock dataset
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

    // Main function to train and predict
    let trainAndPredict () =
        // Initialize ML.NET context
        let mlContext = MLContext(seed = 0)

        // Convert data to ML.NET format
        let dataView = mlContext.Data.LoadFromEnumerable(data)

        // Define the pipeline
        // Define the pipeline with regularization matching manual lambda = 0.1
        let pipeline =
            EstimatorChain()
                .Append(mlContext.Transforms.Concatenate("Features", [| "MonthlyBill"; "ContractLength" |]))
                .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                .Append(mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(l2Regularization = 0.1f))

        // Train the model
        let model : ITransformer = pipeline.Fit dataView

        // Evaluate the model on training data
        let predictions = model.Transform dataView
        let metrics = mlContext.BinaryClassification.Evaluate predictions

        printfn "Training Metrics:"
        printfn "  Log-Loss: %.4f" metrics.LogLoss
        printfn "  Accuracy: %.2f" metrics.Accuracy
        printfn "  AUC: %.2f" metrics.AreaUnderRocCurve

        // Predict for a new customer
        let newCustomer: Customer = { MonthlyBill = 75.0f; ContractLength = 8.0f; Churn = false }

        // Create prediction engine
        let predEngine: PredictionEngine<Customer, ChurnPrediction> = mlContext.Model.CreatePredictionEngine<Customer, ChurnPrediction>(model)

        let prediction: ChurnPrediction = 
            predEngine.Predict(newCustomer)

        // Output prediction
        printfn "\nNew Customer Prediction:"
        printfn "  Churn Probability: %.2f" prediction.Probability
        match prediction.PredictedLabel with
        | true -> printfn "  Prediction: Customer will churn"
        | false -> printfn "  Prediction: Customer will not churn"