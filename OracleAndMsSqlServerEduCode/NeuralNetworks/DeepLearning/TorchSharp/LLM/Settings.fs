namespace NeuralNetworks

open System

open TorchSharp
open TorchSharp.Modules

open type torch.nn
open type torch.nn.functional

type Strategy = 
    | Top_k
    | Top_p
    | Greedy
    | S
    
module Settings =
    
    let internal vocabulary = 
        [
            "The"
            "Sun"
            "is"
            "yellow"
            "black"
            "sky"
            "blue"
            "<eos>"
            "What"
            "the"
            "colour"
            "of"
            "?"
            "."
            "<sep>"
            "<pad>"
            "<unk>"
            "Is"
            "Yes"
            "No"
        ]

    let internal vocabSize = vocabulary |> List.length

    let internal eosTokenIdx = vocabulary |> List.findIndex ((=) "<eos>") |> int64
    let internal padTokenIdx = vocabulary |> List.findIndex ((=) "<pad>") |> int64

    let internal batch = 32L

    let [<Literal>] internal dModel = 48L //  Embeddings of size 48
    let [<Literal>] internal epochs = 240  
    let [<Literal>] internal fineTuneEpochs = 60
    let [<Literal>] internal fineTuneBatch = 10L
    let [<Literal>] internal nHeads = 4L   
    let [<Literal>] internal numLayers = 4 //number of transformer decoder layers
    let [<Literal>] internal dropoutRate = 0.01f
   
    let [<Literal>] internal learningRate = 0.001     

    //INFERENCE SETTINGS
    let [<Literal>] initialStep = 0
    let [<Literal>] maxSteps = 64

    let internal acc = []

    let [<Literal>] internal temp = 0.3f
    let [<Literal>] internal topK = 5L
    let [<Literal>] internal contextSize = 32 

    let [<Literal>] internal p = 0.9 //top-p threshold       
    
    let internal strategy = Top_p  // Top_k  // Top_p  // Greedy