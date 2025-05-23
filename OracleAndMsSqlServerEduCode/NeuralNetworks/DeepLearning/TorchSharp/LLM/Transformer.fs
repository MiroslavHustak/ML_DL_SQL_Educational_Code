namespace NeuralNetworks

open System

open TorchSharp
open TorchSharp.Modules

open type torch.nn
open type torch.nn.functional

module Transformer_TorchSharp =

    //*******************************************************************
    (*    
    This code is still under development and is intended for educational purposes only.
    Submitting issues and pull requests is welcome.

    Option and Result types have not been applied yet.
    Separation of data and data manipulation not fully implemented yet.
    Disposals not fully implemented yet.
    The code architecture will be figured out later.
    
    Acknowledgments:
    - A lecture on creating an LLM with TorchSharp by Tomáš Herceg (https://www.youtube.com/watch?v=tW5RiP765hw&t=12s).
    - Sebastian Raschka, "Build a Large Language Model (From Scratch)".
    *)
    //*******************************************************************

    //TODO!!! zrobit record a presunout do dat    
    let private vocabulary = [ "The"; "Sun"; "is"; "yellow"; "black"; "sky"; "blue"; "<eos>" ]   
    let private vocabSize = vocabulary |> List.length
    
    //TODO!!! zrobit record a presunout do dat
    let [<Literal>] private dModel = 72L //embeddings of size 72
    let [<Literal>] private epochs = 20000
    let [<Literal>] private fineTuneEpochs = 2000 //max new tokens
    let private batch = 32L
    let [<Literal>] private fineTuneBatch = 10L
    let [<Literal>] private nHeads = 12L  
    let [<Literal>] private numLayers = 2
    let [<Literal>] private dropoutRate = 0.1f
    let [<Literal>] private topK = 3L
    let [<Literal>] private contextSize = 1024
    let [<Literal>] private learningRate = 0.001    

    //sinusoidal non-learnable positional encodings 
    let private getPositionalEncodings (seqLen: int64) (dModel: int64) : torch.Tensor =

        let position = torch.arange(seqLen, dtype = torch.float32).unsqueeze(1)              
        let divTerm = torch.exp(torch.arange(0L, dModel, 2L, dtype = torch.float32) * (- Math.Log 10000.0 / float dModel))
        
        let encodings = torch.zeros([|seqLen; dModel|])
        encodings.index_copy_(1, torch.arange(0L, dModel, 2L), torch.sin(position * divTerm)) |> ignore
        encodings.index_copy_(1, torch.arange(1L, dModel, 2L), torch.cos(position * divTerm)) |> ignore
        encodings
  
    type private TransformerDecoderLayer(dModel: int64, nHeads: int64, dropoutRate: float32) as self =
           
        inherit Module<torch.Tensor, torch.Tensor>("TransformerDecoderLayer")
                      
        let qkvProjection = Linear (dModel, dModel * 3L) 
        let outputProjection = Linear (dModel, dModel) 
        let feedForward1 = Linear (dModel, dModel * 4L)
        let feedForward2 = Linear (dModel * 4L, dModel)                 
        let layerNorm1 = LayerNorm [|dModel|]        
        let layerNorm2 = LayerNorm [|dModel|]        
        let dropout = Dropout (float dropoutRate)      

        do self.RegisterComponents()
    
        override _.forward x =
          
            let (batch, seq, _) = x.shape |> Array.head, x.shape |> Array.item 1, x.shape |> Array.item 2
           
            match dModel % nHeads <> 0L with
            | true  -> failwithf "dModel (%d) must be divisible by nHeads (%d) to compute headDim." dModel nHeads
            | false -> ()

            let headDim = dModel / nHeads 
            let reshapedShape = [|batch; seq; 3L; nHeads; headDim|]   
          
            use qkv = qkvProjection.forward x           
            use qkvReshaped = qkv.view reshapedShape 
            
            use q = qkvReshaped.select(2, 0L).transpose(1, 2) 
            use k = qkvReshaped.select(2, 1L).transpose(1, 2) 
            use v = qkvReshaped.select(2, 2L).transpose(1, 2) 
            
            use attentionScores = torch.matmul(q, k.transpose(-2, -1)) / sqrt(float headDim) 
              
            use mask = torch.triu(torch.ones([|seq; seq|], device=attentionScores.device), diagonal=1L).to_type(torch.ScalarType.Bool) 
            use maskedScores = attentionScores.masked_fill(mask.unsqueeze(0).unsqueeze(0), System.Single.NegativeInfinity) 
           
            use attentionWeights = softmax(maskedScores, -1L) |> dropout.forward 
               
            use contextVector = torch.matmul(attentionWeights, v)               
            use contextVector = contextVector.transpose(1, 2).contiguous().view(batch, seq, dModel) 
                 
            contextVector
            |> outputProjection.forward 
            |> fun output -> x + output
            |> layerNorm1.forward 
            |> fun output
                ->
                output
                |> feedForward1.forward 
                |> gelu
                |> feedForward2.forward 
                |> dropout.forward 
                |> fun ffOutput -> output + ffOutput 
                |> layerNorm2.forward 
                       
    type private Transformer(vocabSize: int64, dModel: int64, nHeads: int64, numLayers: int) as self =

        inherit Module<torch.Tensor, torch.Tensor>("Transformer")
      
        let embedding = Embedding (vocabSize, dModel)  
        let posEnc = getPositionalEncodings 1024L dModel   
        let dropout = Dropout (float dropoutRate)
        
        let decoderLayers = 
            let createLayer _ = new TransformerDecoderLayer(dModel, nHeads, dropoutRate) :> torch.nn.Module<torch.Tensor, torch.Tensor>
            let decoderLayersList = List.init numLayers createLayer
            let decoderLayersArray = List.toArray decoderLayersList        
            ModuleList<torch.nn.Module<torch.Tensor, torch.Tensor>>(decoderLayersArray)
        
        let outputLayer = Linear (dModel, vocabSize)
        let norm = LayerNorm [|dModel|]

        do self.RegisterComponents() 

        override _.forward x =

            let emb = embedding.forward x 
            
            let seqLen = x.shape |> Array.item 1
            
            let embWithPos = emb + posEnc.narrow(0L, 0L, seqLen).``to``(x.device)
            let embWithPos = dropout.forward embWithPos

            let decoderLayersOutput = Seq.fold (fun x (layer: torch.nn.Module<torch.Tensor, torch.Tensor>) -> layer.forward(x)) embWithPos decoderLayers
                    
            let normOut = norm.forward decoderLayersOutput           
            
            outputLayer.forward(normOut).to_type(torch.ScalarType.Float32) 

    let private trainEpoch (model: torch.nn.Module<torch.Tensor, torch.Tensor>) (optimizer: torch.optim.Optimizer) (lossFn: CrossEntropyLoss) (input: torch.Tensor) (target: torch.Tensor) maxEpochs phase =
       
        [ 0 .. maxEpochs - 1 ]
        |> List.iter
            (fun epoch
                ->
                optimizer.zero_grad()  
                
                use output = model.forward input
                use loss = lossFn.forward(output.view(-1L, vocabSize), target.view(-1L))
                let perplexity = torch.exp(loss).item<float32>()
               
                try
                    loss.backward() 
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) |> ignore
                with
                | :? System.StackOverflowException as ex
                    ->
                    printfn "StackOverflowException in %s, epoch %d: %s" phase (epoch + 1) (string ex.Message) 
                    Console.ReadLine() |> ignore
                | ex 
                    -> 
                    printfn "%s" (string ex.Message) 
               
                optimizer.step() |> ignore  
                               
                let counter = (+) epoch 1
                match counter % 100 = 0 && counter > 100 with
                | true  -> printfn "%s Epoch %d, Loss: %.4f, Perplexity: %.4f" phase counter (loss.item<float32>()) perplexity
                | false -> ()

                System.GC.Collect()
            )        
   
    let rec [<TailCall>] private generate (model: torch.nn.Module<torch.Tensor, torch.Tensor>) (inputSeq: torch.Tensor) steps maxSteps acc contextSize (temp: float32) (topK: int64) (strategy: string) =
           
        let trimInput (input: torch.Tensor) (contextSize: int) =

            match input.shape.[1] > contextSize with
            | true  -> input.narrow(1, -contextSize, contextSize) // Keep only the last contextSize tokens
            | false -> input
       
        let sampleLogits (logits: torch.Tensor) (temp: float32) (topK: int64) (strategy: string) (vocabSize: int64) =

            let effectiveTopK = min topK vocabSize
            
            match effectiveTopK <= 0L with
            | true  -> failwith "topK must be positive and not exceed vocabulary size"
            | false -> ()
            
            match strategy with
            | "top-k" 
                ->
                let struct (probs, indices) = torch.topk(logits / temp, int effectiveTopK, dim=0)
                let probs = softmax(probs, dim=0L)               
                let idx = torch.multinomial(probs, 1).item<int64>() //It randomly picks one category index from probs
                indices.[idx].item<int64>()

            | "greedy" 
                ->
                torch.argmax(logits, dim=0).item<int64>()

            | _ 
                ->
                failwith $"Unsupported sampling strategy: {strategy}"

        match steps with
        | s
            when s >= maxSteps
            ->
            acc 
        | _ 
            ->
            let _ = torch.no_grad() 
            
            let adjustedLogit: torch.Tensor =
                let trimmedInput = trimInput inputSeq contextSize  
                let logits: torch.Tensor = model.forward trimmedInput 
                let lastLogit: torch.Tensor = logits.select(0, 0L).select(0, -1L) 
                
                match steps with
                | 0 -> lastLogit.index_fill_(0, torch.tensor([|7L|], device=lastLogit.device), System.Single.NegativeInfinity) 
                | _ -> lastLogit 
                        
            let nextToken: int64 = sampleLogits adjustedLogit temp topK strategy (int64 vocabSize) 
            let newAcc: int64 list = nextToken :: acc 
            
            match nextToken with
            | 7L 
                ->
                newAcc 
            | _ ->
                let newInput: torch.Tensor = torch.cat([|inputSeq; torch.tensor([|nextToken|], device=inputSeq.device).unsqueeze(0L)|], dim=1L) 
                generate model newInput (steps + 1) maxSteps newAcc contextSize temp topK strategy  
    
    let internal main () =     
        
        //CUDA® is a parallel computing platform and programming model developed by NVIDIA for general computing on graphical processing units (GPUs).
        let device = match torch.cuda.is_available() with true -> torch.CUDA | false -> torch.CPU
        printfn "Using device: %A" <| (string device).ToUpper()

        let inputData = 
            Array2D.init 320 3
                (fun i k 
                    ->
                    match (i, k) with
                    | i, k when i < 310 -> [|0L; 1L; 2L|] |> Array.item k
                    | i, k when i >= 310 && i <= 318 -> [|0L; 5L; 2L|] |> Array.item k
                    | i, k when i = 319 -> [|0L; 1L; 2L|] |> Array.item k
                    | _ -> failwith "Invalid index"
                )
         //TODO presunout do dat   
     
        let targetData =  
            Array2D.init 320 3
                (fun i k
                    ->
                    match (i, k) with
                    | i, k when i < 310 -> [|2L; 3L; 7L|] |> Array.item k
                    | i, k when i >= 310 && i <= 318 -> [|2L; 6L; 7L|] |> Array.item k
                    | i, k when i = 319 -> [|2L; 4L; 7L|] |> Array.item k
                    | _ -> failwith "Invalid index"
                )
     
        let dataset = TextData.getSequences()

        //Diky shadowing nyni pouzivame inputData a targetData bud z custom-made tokenizatoru nebo z TikTokTokenizeru
        let (inputData, targetData) = Tokenizer.createInputTargetPairs dataset 
        //let (inputData, targetData) = TikTokTokenizer.createInputTargetPairs dataset //TikTokTokenizer //nelze quli male delky     
        
        use input = torch.tensor(inputData, device = device)     
        use target = torch.tensor(targetData, device = device) 
       
        printfn "Starting pre-training..."        
              
        use model = (new Transformer(int64 vocabSize, dModel, nHeads, numLayers)).``to``(device) 

        use lossFn = new CrossEntropyLoss() //viz přednáška Tomáše Hercega
        use optimizer = torch.optim.Adam(model.parameters(), lr = learningRate) 
     
        model.train()

        trainEpoch model optimizer lossFn input target epochs "Pre-training"        
        
        let fineTuneInputData = Array2D.init 10 3 (fun i k -> [|0L; 1L; 2L|] |> Array.item k)       
        let fineTuneTargetData = Array2D.init 10 3 (fun i k -> [|2L; 3L; 7L|] |> Array.item k) 

        use fineTuneInput = torch.tensor(fineTuneInputData, device = device) 
        use fineTuneTarget = torch.tensor(fineTuneTargetData, device = device) 

        printfn "Starting fine-tuning..."
        use fineTuneOptimizer = torch.optim.Adam(model.parameters(), lr = learningRate)

        model.train()
        
        trainEpoch model fineTuneOptimizer lossFn fineTuneInput fineTuneTarget fineTuneEpochs "Fine-tuning"

        printfn "Generating sequence after fine-tuning..."
               
        model.eval()  

        use inputSeq = torch.tensor([|0L; 1L; 2L|], device = device).unsqueeze 0L 
        
        printf "Generated sequence (token IDs): "
        let generated = generate model inputSeq 0 2 [] contextSize 0.7f topK "top-k" |> List.rev 

        generated |> List.iter (printf "%d ")
        printfn "\n"
        
        printf "Generated sequence (words): "
        generated |> List.iter (fun id -> printf "%s " (vocabulary |> List.item (int id)))
        printfn "\n"