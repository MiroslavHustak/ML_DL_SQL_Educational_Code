namespace NeuralNetworks2

type Strategy = 
    | Top_k
    | Top_p
    | Greedy
    | S
    
module Settings2 =
    
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

    let internal prompt = "What is the colour of the Sun? <sep>"

    let internal vocabSize = vocabulary |> List.length

    let internal eosTokenIdx = vocabulary |> List.findIndex ((=) "<eos>") |> int64
    let internal padTokenIdx = vocabulary |> List.findIndex ((=) "<pad>") |> int64

    let [<Literal>] internal dModel = 48L //  Embeddings of size 48
    let [<Literal>] internal epochs = 240  
    let [<Literal>] internal trainingBatch = 32L 

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

    //let [<Literal>] internal temp = 0.3f  //hight accuracy
    let [<Literal>] internal temp = 0.7f //common default, gives more creative, diverse, and less predictable responses.
    let [<Literal>] internal topK = 5L
    let [<Literal>] internal contextSize = 32 

    let [<Literal>] internal p = 0.8 //top-p (nucleus sampling) probability threshold 
    (*
    | p value | Behavior       | Diversity | Determinism |
    |---------|----------------|-----------|-------------|
    | 0.7     | Focused        | Low       | High        |
    | 0.8     | Balanced       | Medium    | Medium      |
    | 0.9     | Diverse        | High      | Lower       |
    | 1.0     | Pure random    | Highest   | Lowest      |
    *) 
    
    let internal strategy = Top_p  // Top_k  // Top_p  // Greedy