namespace NeuralNetworks

open System

open TorchSharp
open TorchSharp.Modules

open type torch.nn
open type torch.nn.functional

module Option = 

    let internal ofNull (value : 'nullableValue) =

        match System.Object.ReferenceEquals(value, null) with //The "value" type can be even non-nullable, and ReferenceEquals will still work.
        | true  -> None
        | false -> Some value       
    
module Settings =
    
    let internal vocabulary = [ "The"; "Sun"; "is"; "yellow"; "black"; "sky"; "blue"; "<eos>"; "<pad>" ]
    let internal vocabSize = vocabulary |> List.length

    let internal eosTokenIdx = vocabulary |> List.findIndex ((=) "<eos>") |> int64
    let internal padTokenIdx = vocabulary |> List.findIndex ((=) "<pad>") |> int64

    let internal batch = 32L

    let [<Literal>] internal dModel = 48L //  Embeddings of size 48
    let [<Literal>] internal epochs = 400  // Fewer epochs for reasonable CPU training
    let [<Literal>] internal fineTuneEpochs = 10
    let [<Literal>] internal fineTuneBatch = 10L
    let [<Literal>] internal nHeads = 4L   // Smaller head count for small models
    let [<Literal>] internal numLayers = 2 //number of transformer decoder layers
    let [<Literal>] internal dropoutRate = 0.1f
    let [<Literal>] internal topK = 3L
    let [<Literal>] internal contextSize = 32 // Smaller context, fits short sequences
    let [<Literal>] internal learningRate = 0.01 // Faster learning for quick overfitting
    let [<Literal>] internal strategy = "top-k"   //"top-k" //"greedy"
    
    
    
     