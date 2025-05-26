namespace NeuralNetworks

open System
open TorchSharp
open TorchSharp.Modules
open type torch.nn
open type torch.nn.functional

module Settings = 

    let internal vocabulary = ["The"; "Sun"; "is"; "yellow"; "black"; "sky"; "blue"; "<eos>"]
    let internal vocabSize = vocabulary |> List.length

    let internal batch = 32L

    let [<Literal>] internal dModel = 72L // Embeddings of size 72
    let [<Literal>] internal epochs = 20000
    let [<Literal>] internal fineTuneEpochs = 4000 // Max new tokens   
    let [<Literal>] internal fineTuneBatch = 10L
    let [<Literal>] internal nHeads = 12L
    let [<Literal>] internal numLayers = 2
    let [<Literal>] internal dropoutRate = 0.1f
    let [<Literal>] internal topK = 3L
    let [<Literal>] internal contextSize = 1024
    let [<Literal>] internal learningRate = 0.01
    let [<Literal>] internal strategy = "greedy"  // "top-k" //