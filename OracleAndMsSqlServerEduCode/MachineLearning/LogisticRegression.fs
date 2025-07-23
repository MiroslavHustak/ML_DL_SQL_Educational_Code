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
            BiasFeature : float // Represents the constant feature x0 = 1 (intercept term) as described in the text
            Churn : float // 1.0 = churn, 0.0 = no churn; binary variable Y ∈ {0, 1}
        }

    module ManualLogisticRegression =
    
        type Customer =
            {
                MonthlyBill : float
                ContractLength : float
                BiasFeature : float // Represents the constant feature x0 = 1 (intercept term) as described in the text
                Churn : float // 1.0 = churn, 0.0 = no churn; binary variable Y ∈ {0, 1}
            }
    
        let trainAndPredictManual () =
    
            // 1. Sigmoid is the core function to transform linear combinations into probabilities, used in prediction
            // Sigmoid function transforms a real number into the interval (0, 1), used to estimate P(Y = 1 | x, w)
            let sigmoid z = 1.0 / (1.0 + exp (-z))
    
            // 2. Features extracts the input vector, needed before computing linear combinations
            // Extracts features [x1, x2, x0] for a customer, where x0 is the bias feature (x0 = 1)
            let features customer = [ customer.MonthlyBill; customer.ContractLength; customer.BiasFeature ]
    
            // 3. DotProduct computes the linear combination wT*x, a prerequisite for prediction
            // Computes the linear combination w0*x0 + w1*x1 + w2*x2 (wT*x in the text)
            let dotProduct (weights : float list) (features : float list) =
                List.fold2 (fun acc w f -> acc + w * f) 0.0 weights features
    
            // 4. Predict combines dotProduct and sigmoid to compute P(Y = 1 | x, w), the main model output
            // Predicts P(Y = 1 | x, w) by applying sigmoid to the linear combination wT*x
            let predict (weights : float list) customer = sigmoid (dotProduct weights (features customer))
    
            // 5. PredictChurn is a task-specific wrapper for predict, logically follows prediction
            // Predicts churn probability for a customer using trained weights
            let predictChurn weights customer = predict weights customer
    
            // 6. LogLoss (Cross-Entropy Loss) evaluates prediction error, used for training and evaluation (tady se uz nepouziva MSE)
            // Computes log loss (negative log-likelihood) for a customer, measuring the error in predicted probability
            // Log loss = -[y * log(p) + (1-y) * log(1-p)], where y is actual churn and p is predicted probability
            let logLoss (weights : float list) customer =

                let y = customer.Churn
                let p = predict weights customer
                -(y * log p + (1.0 - y) * log (1.0 - p))
    
            // 7. Gradient computes the derivative of the loss, needed for optimization //eqv. MLE
            // Computes the gradient of the log loss with respect to weights
            // Gradient for each weight = (p - y) * xi, where p is predicted probability and y is actual churn
            let gradient (weights : float list) customer =

                let y = customer.Churn
                let p = predict weights customer
                let error = p - y // Difference between predicted probability and actual label
                features >> List.map (fun f -> error * f) <| customer
    
            // 8. UpdateWeights applies gradient descent, part of the optimization process
            // Updates weights using gradient descent: w_new = w_old - learningRate * gradient
            let updateWeights (weights : float list) (gradients : float list) learningRate =
                
                List.map2 (fun w g -> w - learningRate * g) weights gradients
    
            // 9. Train orchestrates the training process, using gradient descent to optimize weights
            // Trains the model by iteratively updating weights to minimize log loss
            let train (data : Customer list) learningRate iterations =
                
                let featureCount = 3 // Number of features: MonthlyBill, ContractLength, BiasFeature
                let initialWeights = List.init featureCount (fun _ -> 0.0) // Initialize weights w = [w0, w1, w2] to 0
    
                let rec trainIter weights iter =

                    match iter with
                    | 0 -> 
                        weights // Return final weights after all iterations
    
                    | _ ->
                        // Compute average gradient across all customers
                        let gradients =
                            data
                            |> List.map (gradient weights) // Gradients for each customer
                            |> List.fold (fun acc g -> List.map2 (+) acc g) (List.init featureCount (fun _ -> 0.0)) // Sum gradients
                            |> List.map (fun g -> g / float data.Length) // Average gradients
                    
                        let newWeights = updateWeights weights gradients learningRate // Update weights
    
                        trainIter newWeights (iter - 1) // Recursive call for next iteration
    
                trainIter initialWeights iterations
    
            // 10. TrainModel sets training hyperparameters, initiating the training process
            // Trains the model with specified learning rate and number of iterations
            let trainModel (data : Customer list) = train data 0.001 10000
    
            // Sample dataset with customers, each with features and churn label
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
    
            let weights = trainModel data // Train the model to find optimal weights w
    
            printfn "Trained Weights:"
            printfn "  MonthlyBill: %.4f" (weights |> List.head)
            printfn "  ContractLength: %.4f" (weights |> List.item 1)
            printfn "  BiasFeature: %.4f" (weights |> List.item 2)
    
            printfn "\nPer-Customer Results:"
    
            // Evaluate each customer in the dataset
            data
            |> List.iteri
                (fun i customer 
                    ->
                    let prob = predictChurn weights customer // Predicted probability P(Y = 1 | x, w)
                    let loss = logLoss weights customer // Log loss for this customer
    
                    printfn "Customer %d:" (i + 1)
                    printfn "  Predicted Churn Probability: %.4f" prob
                    printfn "  Log Loss: %.4f" loss
            )
    
            // Compute average log loss across all customers
            let avgLogLoss =
                data
                |> List.map (logLoss weights)
                |> List.average
        
            printfn "\nAverage Log Loss: %.4f" avgLogLoss
    
            // Prediction for a new customer
            let testCustomer = { MonthlyBill = 75.0; ContractLength = 8.0; BiasFeature = 1.0; Churn = 0.0 }
            let testProb = predictChurn weights testCustomer // Predict P(Y = 1 | x, w) for new customer
        
            printfn "\nNew Customer Prediction:"
            printfn "  Churn Probability: %.4f" testProb
        
            // Decision rule: if P(Y = 1 | x, w) >= 0.5, predict Y = 1 (churn); otherwise, Y = 0 (no churn)
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


        (*
        Defining the model components (e.g., sigmoid function, feature extraction, linear combination).
        
        Making predictions (computing probabilities using the sigmoid function).
        
        Evaluating the model (calculating the loss function, such as log loss).
        
        Optimizing the model (computing gradients and updating weights via gradient descent).
        
        Training the model (iteratively applying optimization).
        
        Applying the model (training on data, making predictions, and evaluating results).
        *)
       
        