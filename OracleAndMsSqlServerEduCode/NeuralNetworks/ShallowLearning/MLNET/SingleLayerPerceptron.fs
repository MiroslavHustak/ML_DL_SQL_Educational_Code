namespace NeuralNetworks

open System
open Microsoft.ML
open Microsoft.ML.Data

//machine learning (ML) using logistic regression and linear regression is equivalent to a Single Layer Perceptron (SLP) 
//with sigmoid and linear activation, respectively, differing primarily in naming and framing (neural network vs. statistical perspective).

//pouze vyukovy kod !!!

module SingleLayerPerceptron2 =

    type Customer =
        {
            MonthlyBill : float
            ContractLength : float
            BiasFeature : float
            Churn : float // 1.0 = churn, 0.0 = no churn
        }

    let main2 () =

        let rnd = Random()
        let learningRate = 0.001 // Matches original code
        let initWeights = [| rnd.NextDouble() * 0.0; rnd.NextDouble() * 0.0; rnd.NextDouble() * 0. |] // 3 weights (MonthlyBill, ContractLength, BiasFeature)
        let iterations = 10000 // Matches original code

        // Sigmoid activation function
        let sigmoid (x : float) = 1.0 / (1.0 + Math.Exp(-x))

        // Derivative of sigmoid for backpropagation
        let sigmoidDerivative (x : float) = x * (1.0 - x)

        // Forward pass: Compute output (single neuron)
        let forward (weights: float[]) (inputs: float[]) =
            let weightedSum = Array.sum (Array.map2 (*) inputs weights)
            sigmoid weightedSum

        // Log-loss for evaluation
        let logLoss (weights: float[]) (inputs: float[]) (expected: float) =
            let p = forward weights inputs
            -(expected * log p + (1.0 - expected) * log (1.0 - p))

        // Update weights for one training example
        let updateParameters (learningRate : float) (weights : float[]) (inputs : float[]) (expected : float) =

            let output = forward weights inputs
            let error = output - expected // Gradient of loss w.r.t. output
            let delta = error * sigmoidDerivative output
            let newWeights = Array.map2 (fun w x -> w - learningRate * delta * x) weights inputs // Negative gradient for minimization
            newWeights

        // Train over one epoch (all training examples)
        let trainEpoch (learningRate: float) (trainingData: (float[] * float)[]) (weights: float[]) =
            trainingData
            |> Array.fold (fun w (inputs, expected) ->  updateParameters learningRate w inputs expected) weights 

        // Train for multiple epochs
        let train (epochs : int) (learningRate : float) (trainingData : (float[] * float)[]) (initWeights : float[]) =

            let rec trainRec epoch w =
                match epoch >= epochs with
                | true  -> w
                | false -> trainRec (epoch + 1) (trainEpoch learningRate trainingData w)
            trainRec 0 initWeights

        // Predict churn (threshold at 0.5)
        let predict (weights : float[]) (inputs : float[]) =

            let output = forward weights inputs
            if output >= 0.5 then 1.0 else 0.0

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

        // Prepare training data as (inputs, expected) pairs
        let trainingData =
            data
            |> List.map (fun c -> ([| c.MonthlyBill; c.ContractLength; c.BiasFeature |], c.Churn))
            |> List.toArray

        printfn "Training SLP for churn prediction..."
        let finalWeights = train iterations learningRate trainingData initWeights

        printfn "\nTrained Weights:"
        printfn "  MonthlyBill: %.4f" finalWeights.[0]
        printfn "  ContractLength: %.4f" finalWeights.[1]
        printfn "  BiasFeature: %.4f" finalWeights.[2]

        printfn "\nPer-Customer Results:"
        data
        |> List.iteri
            (fun i customer 
                ->
                let inputs = [| customer.MonthlyBill; customer.ContractLength; customer.BiasFeature |]
                let prob = forward finalWeights inputs
                let loss = logLoss finalWeights inputs customer.Churn
                printfn "Customer %d:" (i + 1)
                printfn "  Predicted Churn Probability: %.4f" prob
                printfn "  Log Loss: %.4f" loss
            )

        let avgLogLoss =
            data
            |> List.map (fun c -> logLoss finalWeights [| c.MonthlyBill; c.ContractLength; c.BiasFeature |] c.Churn)
            |> List.average

        printfn "\nAverage Log Loss: %.4f" avgLogLoss

        // Prediction for a new customer
        let testCustomer = { MonthlyBill = 75.0; ContractLength = 8.0; BiasFeature = 1.0; Churn = 0.0 }
        let testInputs = [| testCustomer.MonthlyBill; testCustomer.ContractLength; testCustomer.BiasFeature |]
        let testProb = forward finalWeights testInputs
        let testPrediction = predict finalWeights testInputs

        printfn "\nNew Customer Prediction:"
        printfn "  Churn Probability: %.4f" testProb
        printfn "  Prediction: %s" (if testPrediction = 1.0 then "Customer will churn" else "Customer will not churn")

module SingleLayerPerceptron3 =

    [<CLIMutable>]
    type Customer =
        {
            [<ColumnName("MonthlyBill")>] MonthlyBill: float32
            [<ColumnName("ContractLength")>] ContractLength: float32
            [<ColumnName("Label")>] Churn: bool // ML.NET expects bool for binary classification
        }

    [<CLIMutable>]
    type Prediction =
        {
            [<ColumnName("PredictedLabel")>] PredictedLabel: bool
            [<ColumnName("Probability")>] Probability: float32
        }

    let main3 () =

        // Forward pass: Compute the neuron's output (sigmoid probability)
        let forward (predictionEngine : PredictionEngine<Customer, Prediction>) (inputs : Customer) =
            let prediction = predictionEngine.Predict(inputs)
            prediction.Probability // Sigmoid output (0 to 1)

        // Predict: Convert probability to binary label (threshold at 0.5)
        let predict (predictionEngine : PredictionEngine<Customer, Prediction>) (inputs : Customer) =
            let prediction = predictionEngine.Predict(inputs)
            (prediction.Probability, prediction.PredictedLabel)

        // Train: Optimize the neuron's weights using ML.NET
        let train (data : Customer list) =

            let mlContext = MLContext(seed = 42) // Fixed seed for reproducibility
            let dataView = mlContext.Data.LoadFromEnumerable data

            // Pipeline: Concatenate features and train single neuron
            let pipeline =
                EstimatorChain()
                    .Append(mlContext.Transforms.Concatenate("Features", [| "MonthlyBill"; "ContractLength" |]))
                    .Append(mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression())

            let model = pipeline.Fit dataView

            // Extract weights (neuron's parameters)
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
                printfn "Trained Weights (Single Neuron):"
                printfn "  MonthlyBill: %.4f" weights.[0]
                printfn "  ContractLength: %.4f" weights.[1]
                printfn "  Bias: %.4f" bias
            | None
                ->
                printfn "Warning: Could not extract weights. Model may not support direct weight access."

            let predictionEngine = mlContext.Model.CreatePredictionEngine<Customer, Prediction> model
            (mlContext, model, predictionEngine)

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

        printfn "ML.NET Single Layer Perceptron Results:"

        data
        |> List.iteri
            (fun i customer
                ->
                let (prob, label) = predict predictionEngine customer
                printfn "Customer %d:" (i + 1)
                printfn "  Predicted Churn: %b" label
                printfn "  Churn Probability: %.4f" prob
            )

        // Evaluate model
        let predictions = model.Transform(mlContext.Data.LoadFromEnumerable data)
        let metrics = mlContext.BinaryClassification.Evaluate(predictions, labelColumnName = "Label")

        printfn "\nModel Evaluation Metrics:"
        printfn "  Log Loss: %.4f" metrics.LogLoss
        printfn "  Accuracy: %.4f" metrics.Accuracy
        printfn "  Area Under ROC Curve (AUC): %.4f" metrics.AreaUnderRocCurve

        // Predict for a new customer
        let testCustomer = { MonthlyBill = 75.0f; ContractLength = 8.0f; Churn = false }
        let (testProb, testLabel) = predict predictionEngine testCustomer

        printfn "\nNew Customer Prediction:"
        printfn "  Churn Probability: %.4f" testProb
        printfn "  Prediction: %s" (if testLabel then "Customer will churn" else "Customer will not churn")

        (*
        The LbfgsLogisticRegression trainer models a single artificial neuron:
        Inputs: MonthlyBill, ContractLength.
        
        Weights: w1, w2, and a bias term.
        
        Computation: sigmoid(w1*MonthlyBill + w2*ContractLength + bias), computed by the forward function via the prediction engine.
        
        Output: Binary classification (true/false) via predict (threshold at 0.5).
        
        This mirrors the manual SLP’s forward function, but ML.NET handles the weighted sum and sigmoid internally. 
        
        *)

open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.Distributions

module SingleLayerPerceptron4 =

    type Message =
        | UpdateState of int
        | CheckState of AsyncReplyChannel<int>

    let actor2 () =
        MailboxProcessor.Start
            (fun inbox
                ->
                let rec loop counter =
                    async
                        {
                            match! inbox.Receive() with
                            | UpdateState n 
                                -> 
                                return! loop (counter + n)

                            | CheckState replyChannel 
                                ->
                                replyChannel.Reply counter
                                return! loop counter
                        }
                loop 0
            )

    let machineLearningSLP () =

        let actor = actor2 ()

        // Generate synthetic data: y = 2 * x1 + 3 * x2 + noise
        let generateMockData numSamples =

            let rand = Random(42)
            let noise = Normal(0.0, 0.1) // Gaussian noise

            List.init numSamples
                (fun _
                    ->
                    let x1 = rand.NextDouble() * 10.0
                    let x2 = rand.NextDouble() * 10.0
                    let noiseSample = noise.Sample()
                    let y = 2.0 * x1 + 3.0 * x2 + noiseSample
                    ([1.0; x1; x2], y)
                ) // Include bias term (1.0)

        // Forward pass: Compute the neuron's output (linear activation)
        let forward (weights : Vector<float>) (inputs : float list) =

            let inputVector = DenseVector.ofList inputs
            weights * inputVector // Linear: w0*1 + w1*x1 + w2*x2

        // Mean Squared Error loss
        let meanSquaredError (X : Matrix<float>) (y : Vector<float>) (weights : Vector<float>) =

            let predictions = X * weights
            let errors = predictions - y
            (errors.PointwisePower 2).Sum() / (2.0 * float X.RowCount)

        // Train: Optimize the neuron's weights via gradient descent
        let train (X : Matrix<float>) (y : Vector<float>) (initialWeights : Vector<float>) (learningRate : float) (maxIterations : int) (minIterations : int) (tolerance : float) =

            let m = float X.RowCount

            let rec loop currentWeights previousCost iteration =

                match iteration >= maxIterations with
                | true 
                    ->
                    actor.Post <| UpdateState 1
                    currentWeights

                | false 
                    ->   
                    actor.Post <| UpdateState 1
                    let predictions : Vector<float> = X * currentWeights
                    let errors = predictions - y
                    let gradient = (X.Transpose() * errors) / m
                    let newWeights = currentWeights - learningRate * gradient
                    let newCost = meanSquaredError X y newWeights

                    match iteration >= minIterations && abs (previousCost - newCost) < tolerance with
                    | true  -> newWeights
                    | false -> loop newWeights newCost (iteration + 1)

            loop initialWeights (meanSquaredError X y initialWeights) 0

        // Predict: Compute the neuron's output for new inputs
        let predict (weights : Vector<float>) (inputs : float list) = forward weights inputs

        let samples = 100000
        let learningRate = 0.01
        let maxIterations = 100000
        let minIterations = 1000
        let tolerance = 1e-6

        // Generate data
        let data = generateMockData samples

        let X =
            DenseMatrix.ofRowList 
                (
                    List.init data.Length (fun i -> fst (data |> List.item i))
                )

        let y =
            DenseVector.ofList
                (
                    List.init data.Length (fun i -> snd (data |> List.item i))
                )

        let initialWeights = DenseVector.zero 3 // [bias; weight1; weight2]
        
        // Train the neuron
        let trainedWeights = train X y initialWeights learningRate maxIterations minIterations tolerance

        // Print results
        printfn "Learned parameters (Single Neuron): %A" trainedWeights
        printfn "Counter: %i" <| actor.PostAndReply (fun replyChannel -> CheckState replyChannel)
        printfn "Final cost (MSE): %f" (meanSquaredError X y trainedWeights)

        // Predict for a new input [1.0; 5.0; 5.0]
        let newInput = [1.0; 5.0; 5.0]
        let prediction = predict trainedWeights newInput

        printfn "Prediction for input [1.0; 5.0; 5.0]: %f" prediction

        actor.Dispose()
