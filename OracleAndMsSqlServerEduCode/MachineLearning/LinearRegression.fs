namespace MachineLearning

open System

open Microsoft.ML
open Microsoft.ML.Data
open Microsoft.ML.Trainers

open MathNet.Numerics.Distributions
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double

//pouze vyukovy kod !!!

[<CLIMutable>]
type private DataPoint =
    {
        X1 : float32
        X2 : float32
        Label : float32
    }
   
type private Increment =
    | UpdateState of int  
    | CheckState  of AsyncReplyChannel<int>

module MachineLearning = 

    let [<Literal>] private tolerance = 1e-6
    let [<Literal>] private learningRate = 0.01
    let [<Literal>] private maxIterations = 100000
    let [<Literal>] private minIterations = 1000
    let [<Literal>] private samples = 100000
    
    let private actor2 () =

        MailboxProcessor<Increment>
            .StartImmediate
                <|
                fun inbox 
                    ->
                    let rec loop n = 
                        async
                            { 
                               match! inbox.Receive() with
                               | UpdateState i 
                                   ->
                                   let updated = (+) n i
                                   return! loop updated

                               | CheckState  replyChannel 
                                   ->
                                   replyChannel.Reply n
                                   return! loop n
                            }
                    loop 0  
   
    let machineLearningMLdotNET () =

        let mlContext = MLContext(seed = 42)
        
        let generateMockData (numSamples : int) =

            let rand = Random 42
            let noise = Normal(0.0, 0.1)

            Array.init numSamples 
                (fun _ 
                    ->
                    let x1 = float32 (rand.NextDouble() * 10.0)
                    let x2 = float32 (rand.NextDouble() * 10.0)
                    let noiseSample = float32 (noise.Sample())
                    let y = 2.0f * x1 + 3.0f * x2 + noiseSample
                    { X1 = x1; X2 = x2; Label = y }
                )
    
        let data = generateMockData samples
        let dataView = mlContext.Data.LoadFromEnumerable data

        //Poisson regression is designed for modeling count data, and the labels (dependent variable) in Poisson regression should always be non-negative.
        //V mem pripade to vubec nefunguje
        //let data = data |> Array.filter (fun sample -> sample.Label >= 0.0f)
        //let dataView = mlContext.Data.LoadFromEnumerable(data)
    
        let featureEngineering =
            mlContext.Transforms.Concatenate("Features", [| "X1"; "X2" |])
        
        //Sdca stands for Stochastic Dual Coordinate Ascent (a kind of solver similar to stochastic gradient descent).      
        let trainer_SDCA () = mlContext.Regression.Trainers.Sdca(labelColumnName = "Label", featureColumnName = "Features")
        
        //SGD (stochastic gradient descent) neumoznuje stanovit tolerance 1e-40f 
        let trainer_SGD () = 

            mlContext.Regression.Trainers.OnlineGradientDescent(
                labelColumnName = "Label",
                featureColumnName = "Features",
                learningRate = 0.01f,
                numberOfIterations = maxIterations,
                lossFunction = SquaredLoss()  // <<<<< !!! EXPLICITLY specify squared loss
            )

        //LBFGS Poisson Regression //Limited-memory Broyden-Fletcher-Goldfarb-Shanno algorithm
        let trainerOptions = 
            LbfgsPoissonRegressionTrainer.Options(
                LabelColumnName = "Label",
                FeatureColumnName = "Features",
                L2Regularization = 0.0f,
                HistorySize = 100,
                MaximumNumberOfIterations = maxIterations,
                OptimizationTolerance = float32 tolerance
            )

        //LBFGS Poisson Regression //Limited-memory Broyden-Fletcher-Goldfarb-Shanno algorithm
        let trainer_LBFGS () = mlContext.Regression.Trainers.LbfgsPoissonRegression trainerOptions
        
        //let trainer = trainer_SDCA ()
        let trainer = trainer_SGD ()        
        //let trainer = trainer_LBFGS ()  //nema to ocekavane vysledky
        
        let pipeline = EstimatorChain().Append(featureEngineering).Append(trainer)      
        
        let model = pipeline.Fit dataView
    
        // Extract model parameters
        let linearModel = model.LastTransformer.Model
    
        let weightsArray = linearModel.Weights |> Array.ofSeq
        let bias = linearModel.Bias
    
        printfn "Learned parameters:"
        printfn "Bias: %f" bias
        printfn "Weight for X1: %f" (weightsArray |> Array.head)
        printfn "Weight for X2: %f" (weightsArray |> Array.item 1)
    
        // Prepare data for mean squared error calculation
        let X =
            DenseMatrix.ofRowArrays (
                data |> Array.map (fun p -> [| p.X1; p.X2 |])
            )
        
        let y =
            DenseVector.ofArray (
                data |> Array.map (fun p -> p.Label)
            )

        let weightsArray = linearModel.Weights |> Seq.map float32 |> Array.ofSeq
        let theta = DenseVector.ofArray weightsArray
        let predictions = X * theta + DenseVector.create samples bias
        let errors = predictions - y
        let mse = (errors.PointwisePower 2.0f).Sum() / (2.0f * (float32 samples))
    
        printfn "Final cost (Mean Squared Error): %f" mse
    
        // Prediction for [5.0; 5.0]
        let input = DenseVector.ofArray ([| 5.0; 5.0 |] |> Array.map float32)
        let prediction = input * theta + bias
        printfn "Prediction for input [5.0; 5.0]: %f" prediction
        
    let machineLearningArray () =

        let actor = actor2 ()

        // Generate synthetic data: y = 2 * x1 + 3 * x2 + noise
        let generateMockData numSamples =

            let rand = System.Random 42
            let noise = Normal(0.0, 0.1) // Gaussian noise //v ML staci takto maly noise, pri linearni regresi v realnem svete muze byt daleko vetsi
              
            Array.init numSamples
                (fun _ 
                    ->
                    let x1 = rand.NextDouble() * 10.0
                    let x2 = rand.NextDouble() * 10.0
                    let noiseSample = noise.Sample()
                    let y = 2.0 * x1 + 3.0 * x2 + noiseSample
                    ([|1.0; x1; x2|], y) // Include bias term (1.0)
                )           
        
        // metoda nejmensich ctvercu (least squares method, squared error) method
        let meanSquaredError (X : Matrix<float>) (y : Vector<float>) (theta : Vector<float>) =

            let predictions = X * theta
            let errors = predictions - y
            (errors.PointwisePower 2).Sum() / (2.0 * float X.RowCount) //mean
         
        (*
        let gradientDescent3 (X : Matrix<float>) (y : Vector<float>) (theta : Vector<float>) (alpha : float) (iterations : int) =

            let m = float X.RowCount

            [ 0 .. iterations - 1 ]
            |> List.fold 
                (fun currentTheta _
                    ->
                    let predictions : Vector<float> = X * currentTheta
                    let errors = predictions - y
                    let gradient = (X.Transpose() * errors) / m
                    currentTheta - alpha * gradient
                ) theta
        *) 
        (*
        It just mechanically updates theta using the gradient.           
        It does not check if the model is getting better or worse.           
        It does not stop early based on precision; it just runs for exactly iterations rounds.   
        *)     
        
        let gradientDescent (X : Matrix<float>) (y : Vector<float>) (theta : Vector<float>) (alpha : float) (maxIterations : int) (minIterations : int) (tolerance : float) =
        
            let m = float X.RowCount
        
            let rec loop currentTheta previousCost iteration =

                match iteration >= maxIterations with
                | true 
                    ->
                    actor.Post <| UpdateState 1

                    currentTheta

                | false 
                    ->
                    actor.Post <| UpdateState 1

                    let predictions : Vector<float> = X * currentTheta
                    let errors = predictions - y
                    let gradient = (X.Transpose() * errors) / m
                    let newTheta = currentTheta - alpha * gradient
                    let newCost = meanSquaredError X y newTheta
        
                    match iteration >= minIterations && abs (previousCost - newCost) < tolerance with
                    | true  -> newTheta
                    | false -> loop newTheta newCost (iteration + 1)
        
            loop theta (meanSquaredError X y theta) 0        

        // Generate mock data samples
        let data = generateMockData samples 

        let X = 
            DenseMatrix.ofRowArrays (
                Array.init data.Length 
                    (fun i -> fst (data |> Array.item i))
            )
        
        let y = 
            DenseVector.ofArray (
                Array.init data.Length
                    (fun i -> snd (data |> Array.item i))
            )
        
        // Initialize parameters
        let initialTheta = DenseVector.zero 3 // [bias; weight1; weight2]
              
        // Train the model
        let theta : Vector<float> = gradientDescent X y initialTheta learningRate maxIterations minIterations tolerance 
        
        // Print results
        printfn "Learned parameters: %A" theta
        printfn "Counter: %i"  <| actor.PostAndReply (fun replyChannel -> CheckState replyChannel)
        printfn "Final cost: %f" (meanSquaredError X y theta)
        
        // Make a prediction for a new input [1.0; 5.0; 5.0]
        let newInput = DenseVector.ofArray [|1.0; 5.0; 5.0|]
        let prediction = newInput * theta
        printfn "Prediction for input [1.0; 5.0; 5.0]: %f" prediction    

        actor.Dispose()

    let machineLearningList () =
    
        let actor = actor2 ()
    
        // Generate synthetic data: y = 2 * x1 + 3 * x2 + noise
        let generateMockData numSamples =
    
            let rand = System.Random 42
            let noise = Normal(0.0, 0.1) // Gaussian noise //v ML staci takto maly noise, pri linearni regresi v realnem svete muze byt daleko vetsi
                  
            List.init numSamples
                (fun _ 
                    ->
                    let x1 = rand.NextDouble() * 10.0
                    let x2 = rand.NextDouble() * 10.0
                    let noiseSample = noise.Sample()
                    let y = 2.0 * x1 + 3.0 * x2 + noiseSample
                    ([1.0; x1; x2], y) // Include bias term (1.0)
                )           
           
        let meanSquaredError (X : Matrix<float>) (y : Vector<float>) (theta : Vector<float>) =

            let predictions = X * theta // Forward pass: Compute predictions (X * theta)
            let errors = predictions - y // Compute errors for loss calculation
            (errors.PointwisePower 2).Sum() / (2.0 * float X.RowCount) // Mean squared error (MSE) loss
            // Remark: The loss function (MSE) is used in backpropagation to compute gradients.
            // The forward pass (predictions = X * theta) sets up the computation graph needed for backpropagation.
              
        let gradientDescent (X : Matrix<float>) (y : Vector<float>) (theta : Vector<float>) (alpha : float) (maxIterations : int) (minIterations : int) (tolerance : float) =
        
            let m = float X.RowCount
        
            let rec loop currentTheta previousCost iteration =
                match iteration >= maxIterations with
                | true 
                    ->
                    actor.Post <| UpdateState 1

                    currentTheta
    
                | false
                    ->
                    actor.Post <| UpdateState 1
                    
                    let predictions : Vector<float> = X * currentTheta // Forward pass: Compute predictions
                    // Remark: This forward pass is the first step of backpropagation, computing the model's output (X * theta) needed for the loss and gradient.
                    let errors = predictions - y // Compute errors (difference between predictions and true values)
                    // Remark: The errors (predictions - y) are used in backpropagation to compute the gradient of the loss with respect to predictions.
                    let gradient = (X.Transpose() * errors) / m // Compute gradient of loss w.r.t. theta
                    // Remark: This is the core of backpropagation: computing the gradient of the loss w.r.t. theta using the chain rule.
                    // The gradient is calculated as (1/m) * X^T * (predictions - y), propagating the error backward to the parameters.
                    let newTheta = currentTheta - alpha * gradient // Update parameters using gradient descent
                    // Remark: This step uses the gradients from backpropagation to update theta, completing one iteration of optimization.
                    let newCost = meanSquaredError X y newTheta // Compute new loss for convergence check
        
                    match iteration >= minIterations && abs (previousCost - newCost) < tolerance with
                    | true  -> newTheta
                    | false -> loop newTheta newCost (iteration + 1)        
                    
            loop theta (meanSquaredError X y theta) 0     
            // Remark: The gradientDescent function implements backpropagation iteratively.
            // Each iteration involves a forward pass (computing predictions), a backward pass (computing gradients via the chain rule), and a parameter update.
    
        let data = generateMockData samples 
    
        let X = 
            DenseMatrix.ofRowList (
                List.init data.Length (fun i -> fst (data |> List.item i))
            )
            
        let y = 
            DenseVector.ofList(
                List.init data.Length (fun i -> snd (data |> List.item i))
            )
            
        let initialTheta = DenseVector.zero 3 // [bias; weight1; weight2]
           
        let theta = gradientDescent X y initialTheta learningRate maxIterations minIterations tolerance
        // Remark: The call to gradientDescent triggers the backpropagation process, optimizing theta by repeatedly computing gradients and updating parameters.
            
        // Print results
        printfn "Learned parameters: %A" theta
        printfn "Counter: %i"  <| actor.PostAndReply (fun replyChannel -> CheckState replyChannel)
        printfn "Final cost: %f" (meanSquaredError X y theta)
            
        // Make a prediction for a new input [1.0; 5.0; 5.0]
        let newInput = DenseVector.ofList [1.0; 5.0; 5.0]
        let prediction = newInput * theta // Forward pass for prediction
        // Remark: This is a forward pass, similar to the one used in backpropagation, but here it's for inference, not gradient computation.
        
        printfn "Prediction for input [1.0; 5.0; 5.0]: %f" prediction
    
        actor.Dispose()

    //********************************************************************************
       

    let solveLinearSystem () =

        let A1 = matrix [| [| 2.0; 3.0 |]
                           [| 4.0; -1.0 |] |]
    
        // Constant vector b
        let b1 = DenseVector.ofArray [| 8.0; 7.0 |]

        let m = 
            matrix [[ 1.0; 2.0 ]
                    [ 3.0; 4.0 ]]
        
        let m' = m.Inverse()
        
        // Coefficient matrix A
        let A = 
            matrix [[ 2.0; 3.0 ]
                    [ 4.0; -1.0 ]]
        
        // Constant vector b
        let b = DenseVector.ofList [ 8.0; 7.0 ]
        
        // Solve x = A^(-1) * b
        let x = A.Inverse() * b
        
        // Print the solution
        printfn "Solution x: %A" x

        printfn "Solution: x = %f, y = %f" x.[0] x.[1]
        
        // Verify by computing A * x
        let verification = A * x
        printfn "Verification (should be close to [8; 7]): %A" verification

        //************************************************************************