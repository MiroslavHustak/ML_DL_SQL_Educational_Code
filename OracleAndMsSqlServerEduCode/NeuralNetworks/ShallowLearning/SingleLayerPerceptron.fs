namespace NeuralNetworks

open System

module SingleLayerPerceptron = 

    // Sigmoid activation function
    let sigmoid (x : float) = 1.0 / (1.0 + Math.Exp(-x))

    // Derivative of sigmoid for backpropagation
    let sigmoidDerivative (x : float) = x * (1.0 - x)

    // Forward pass: Compute output
    let forward (weights : float[]) (bias : float) (inputs : float[]) =

        let weightedSum = Array.sum (Array.map2 (*) inputs weights) + bias
        sigmoid weightedSum

    // Update weights and bias for one training example
    let updateParameters (learningRate : float) (weights : float[]) (bias : float) (inputs : float[]) (expected : float) =

        let output = forward weights bias inputs
        let error = expected - output
        let delta = error * sigmoidDerivative output
        let newWeights = Array.map2 (fun w x -> w + learningRate * delta * x) weights inputs
        let newBias = bias + learningRate * delta
        (newWeights, newBias)

    // Train over one epoch (all training examples)
    let trainEpoch (learningRate : float) (trainingData : (float[] * float)[]) (weights : float[]) (bias : float) =

        Array.fold
            (fun (w, b) (inputs, expected)
                -> updateParameters learningRate w b inputs expected
            ) (weights, bias) trainingData

    // Train for multiple epochs
    let train (epochs : int) (learningRate : float) (trainingData : (float[] * float)[]) (initWeights : float[]) (initBias : float) =

        let rec trainRec epoch (w, b) =
            match epoch >= epochs with
            | true  -> (w, b)
            | false -> trainRec (epoch + 1) (trainEpoch learningRate trainingData w b)

        trainRec 0 (initWeights, initBias)

    // Test prediction
    let predict (weights : float[]) (bias : float) (inputs : float[]) =

        let output = forward weights bias inputs
        if output >= 0.5 then 1.0 else 0.0

    // Main function
    let main () =

        let rnd = Random()
        let learningRate = 0.1
        let initWeights = [| rnd.NextDouble() * 0.1; rnd.NextDouble() * 0.1 |]
        let initBias = rnd.NextDouble() * 0.1

        // Training data: [study hours; previous test score], expected output (1 = pass, 0 = fail)
        let trainingData =
            [|
                ([| 2.0; 60.0 |], 0.0)  // Few hours, low score -> fail
                ([| 6.0; 80.0 |], 1.0)  // More hours, high score -> pass
                ([| 3.0; 50.0 |], 0.0)  // Moderate hours, low score -> fail
                ([| 8.0; 90.0 |], 1.0)  // Many hours, high score -> pass
            |]

        printfn "Training neural network..."
        let (finalWeights, finalBias) = train 1000 learningRate trainingData initWeights initBias

        // Test cases
        let testCases =
            [|
                [| 5.0; 75.0 |]  // Moderate hours, decent score
                [| 1.0; 40.0 |]  // Few hours, low score
            |]

        testCases
        |> Array.iter 
            (fun inputs 
                ->
                let prediction = predict finalWeights finalBias inputs
                printfn "Inputs: %.1f hours, %.1f score -> Predicted: %s" 
                         inputs.[0] inputs.[1] (match prediction = 1.0 with true -> "Pass" | false -> "Fail")
            )

  
