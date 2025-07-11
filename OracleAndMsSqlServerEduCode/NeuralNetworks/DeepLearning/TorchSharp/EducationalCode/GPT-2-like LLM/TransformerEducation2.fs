namespace NeuralNetworks2

open System

open TorchSharp
open TorchSharp.Modules

open type torch.nn
open type torch.nn.functional

open Settings2

module Transformer_TorchSharp =
        
    type private TransformerDecoderLayer(dModel: int64, nHeads: int64, dropoutRate: float32) as self =

        inherit Module<torch.Tensor, torch.Tensor>("TransformerDecoderLayer")
        
        let qkvProjection = Linear(dModel, dModel * 3L)  // LOAD PRE-TRAINED GPT-2 QKV PROJECTION WEIGHTS AND BIASES HERE
        let outputProjection = Linear(dModel, dModel)    // LOAD PRE-TRAINED GPT-2 OUTPUT PROJECTION WEIGHTS AND BIASES HERE
        let feedForward1 = Linear(dModel, dModel * 4L)   // LOAD PRE-TRAINED GPT-2 FEED-FORWARD LAYER 1 WEIGHTS AND BIASES HERE
        let feedForward2 = Linear(dModel * 4L, dModel)   // LOAD PRE-TRAINED GPT-2 FEED-FORWARD LAYER 2 WEIGHTS AND BIASES HERE
        let layerNorm1 = LayerNorm [|dModel|]            // LOAD PRE-TRAINED GPT-2 LAYER NORM 1 SCALE AND SHIFT HERE
        let layerNorm2 = LayerNorm [|dModel|]            // LOAD PRE-TRAINED GPT-2 LAYER NORM 2 SCALE AND SHIFT HERE       
        let dropout = Dropout(float dropoutRate)

        do self.RegisterComponents()
    
        override _.forward x =

            // PROCESSING INPUT EMBEDDINGS THROUGH MULTI-HEAD ATTENTION AND FF LAYERS TO PRODUCE CONTEXTUALIZED REPRESENTATIONS
            (*
            Contextualized representations = the refined embeddings output by a transformer decoder layer,
            with shape [batch, seq, dModel] (e.g., [1, 3, 72]), where each token’s 72-dimensional vector (e.g., [0.25, -0.15, ..., 0.20] for “is”)
            encodes its meaning and context from other tokens in the sequence. 
            They are produced by:
            Multi-head attention: Incorporating inter-token relationships (e.g., “is” attending to “Sun”).
            Feed-forward processing: Applying position-wise non-linear transformations.            
            Residual connections and layer normalization stabilize the output.            
            These representations are passed to subsequent layers or, after final normalization, used to compute logits = emb @ W^T + b for token prediction (e.g., “yellow”).
            *) 

            match x.shape with
            | [|batch; seq; dmodel|]
                when dmodel = dModel
                    ->
                    let (batch, seq, _) = x.shape |> Array.head, x.shape |> Array.item 1, x.shape |> Array.last

                    //printfn "batch %i" batch

                    match dModel % nHeads <> 0L with
                    | true  -> failwithf "dModel (%d) must be divisible by nHeads (%d) to compute headDim." dModel nHeads
                    | false -> ()
                
                    // MULI-HEAD ATTENTION
                    let headDim = dModel / nHeads
                    let reshapedShape = [|batch; seq; 3L; nHeads; headDim|]
                
                    use qkv = qkvProjection.forward x
                    use qkvReshaped = qkv.view reshapedShape //The qkv.view method performs the actual splitting into the heads
                
                    use q = qkvReshaped.select(2, 0L).transpose(1, 2)
                    use k = qkvReshaped.select(2, 1L).transpose(1, 2)
                    use v = qkvReshaped.select(2, 2L).transpose(1, 2)
                    // Each of these (q, k, v) ends up with shape: [batch; nHeads; seq; headDim]                    
                
                    // batched matrix multiplication matmul
                    use attentionScores = torch.matmul(q, k.transpose(-2, -1)) / sqrt(float headDim)
                
                    use mask = torch.triu(torch.ones([|seq; seq|], device = attentionScores.device), diagonal = 1L).to_type(torch.ScalarType.Bool)
                    use maskedScores = attentionScores.masked_fill(mask.unsqueeze(0).unsqueeze(0), System.Single.NegativeInfinity) //Hiding future words with negative infinity
                
                    use attentionWeights = //softmax -> normalization (sum of attention weights to be 1) //negative infinity -> 0
                        softmax(maskedScores, -1L)
                        |> dropout.forward //zeroing out additional attention weights to reduce overfitting
                
                    // batched matrix multiplications matmul
                    let contextVector = torch.matmul(attentionWeights, v) //use could lead to premature disposal 

                    //Back to single representation //Concatenating the heads (from [batch; nHeads; seq; headDim]) back into [batch; seq; dModel]
                    use contextVector = contextVector.transpose(1, 2).contiguous().view(batch, seq, dModel)
                    //The concatenated output of the multi-head attention mechanism, where each head’s output (shape [batch; seq; headDim]) has been reshaped back to dModel
                                    
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
            
            | shapeArr 
                    ->
                    failwithf "Input tensor must have shape [batch; seq; dModel] but got %A" shapeArr            
    
    // Sinusoidal non-learnable positional encodings
    let private getPositionalEncodings (seqLen: int64) (dModel: int64) : torch.Tensor =

        let position = torch.arange(seqLen, dtype=torch.float32).unsqueeze(1)
        let divTerm = torch.exp(torch.arange(0L, dModel, 2L, dtype = torch.float32) * (-Math.Log 10000.0 / float dModel))
        
        let encodings = torch.zeros([|seqLen; dModel|])
        encodings.index_copy_(1, torch.arange(0L, dModel, 2L), torch.sin(position * divTerm)) |> ignore
        encodings.index_copy_(1, torch.arange(1L, dModel, 2L), torch.cos(position * divTerm)) |> ignore
        encodings
    
    type private Transformer(vocabSize: int64, dModel: int64, nHeads: int64, numLayers: int) as self =

        inherit Module<torch.Tensor, torch.Tensor>("Transformer")
        
        let embedding = Embedding(vocabSize, dModel)  // LOAD PRE-TRAINED GPT-2 EMBEDDING WEIGHTS HERE
        let posEnc = getPositionalEncodings 128L dModel
        let dropout = Dropout(float dropoutRate)
        
        //hidden layers of the transformer neural network (performing the core transformations between the input embeddings and the output logits)
        let decoderLayers =
            let createLayer _ = new TransformerDecoderLayer(dModel, nHeads, dropoutRate) :> torch.nn.Module<torch.Tensor, torch.Tensor>
            let decoderLayersList = List.init numLayers createLayer
            let decoderLayersArray = List.toArray decoderLayersList
            
            ModuleList<torch.nn.Module<torch.Tensor, torch.Tensor>> decoderLayersArray
        
        let outputLayer = Linear(dModel, vocabSize) // LOAD PRE-TRAINED GPT-2 OUTPUT LAYER WEIGHTS AND BIASES HERE (MAY BE TIED TO EMBEDDING WEIGHTS)
        let norm = LayerNorm [|dModel|] // LOAD PRE-TRAINED GPT-2 FINAL LAYER NORM SCALE AND SHIFT HERE

        do outputLayer.weight <- embedding.weight //in case of using pre-trained weights //GPT-2 typically ties embedding and output weights     
        
        do self.RegisterComponents()
        
        override _.forward x =

            let emb = embedding.forward x
            let seqLen = x.shape |> Array.item 1
            
            let embWithPos = emb + posEnc.narrow(0L, 0L, seqLen).``to``(x.device)
            let embWithPos = dropout.forward embWithPos
            
            // PRODUCING CONTEXTUALIZED EMBEDDINGS FROM DECODER LAYERS, INCORPORATING MULTI-HEAD ATTENTION AND FEED-FORWARD PROCESSING
            (*
            Feed-forward = the position-wise feed-forward neural network (FFN) in each TransformerDecoderLayer, 
            consisting of two linear layers (feedForward1, feedForward2), GELU activation, dropout, a residual connection, and layer normalization. 
            It transforms each token’s embedding (e.g., [0.23, -0.13, ..., 0.18] for “is”) after attention, refining it to capture complex patterns. 
            Alongside multi-head attention, it produces contextualized embeddings that are normalized and used to compute logits = emb @ W^T + b. 
            The FFN’s parameters ([288, 72], [72, 288], biases) are updated by the Adam optimizer to improve predictions.
            *)
            //layer.forward executes all the operations defined in the TransformerDecoderLayer, including self-attention, feed-forward networks, residual connections, and normalization
                        
            let decoderLayersOutput = Seq.fold (fun x (layer: torch.nn.Module<torch.Tensor, torch.Tensor>) -> layer.forward(x)) embWithPos decoderLayers
            
            // OBTAINING NORMALIZED CONTEXTUALIZED EMBEDDINGS READY FOR LOGIT COMPUTATION
            let normOut = norm.forward decoderLayersOutput
            
            outputLayer.forward(normOut).to_type(torch.ScalarType.Float32)
    
    // Trains the model (or fine-tunes it respectively)
    let private trainEpoch (model: torch.nn.Module<torch.Tensor, torch.Tensor>) (optimizer: torch.optim.Optimizer) (lossFn: CrossEntropyLoss) (input: torch.Tensor) (target: torch.Tensor) maxEpochs phase =
       
       [0 .. maxEpochs - 1]
        |> List.iter
            (fun epoch
                ->  
                let counter = (+) epoch 1

                optimizer.zero_grad()
                
                // COMPUTING LOGITS (emb @ W^T + b), THEN UPDATES EMBEDDING VECTORS, OUTPUT LAYER WEIGHTS, AND BIASES BASED ON LOSS
                use output = model.forward input
                
                // COMPUTING LOSS BY COMPARING LOGITS TO TARGET TOKEN IDs (e.g., ID 3 for “yellow”)
                use loss = lossFn.forward(output.view(-1L, vocabSize), target.view(-1L))
                
                let perplexity = torch.exp(loss).item<float32>() //Lower perplexity (closer to 1) indicates a better model   
                //Perplexity measures the average number of tokens the model considers as equally likely for each prediction.                
                
                try
                    // COMPUTING GRADIENTS OF THE LOSS TO UPDATE MODEL PARAMETERS
                    (*
                    Embedding weights ([8, 72])
                    Decoder layer weights and biases (QKV projection, output projection, FFN, layer norms) for 2 layers                    
                    Final layer norm scale and shift ([72] each)                    
                    Output layer weight (W, [8, 72]) and bias (b, [8])             
                    *)
                    loss.backward() //Computes gradients of the loss with respect to model parameters (e.g., embedding weights, output layer’s W and b)                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0) |> ignore<float>
                with
                | :? System.StackOverflowException 
                    as ex
                        ->
                        printfn "StackOverflowException in %s, epoch %d: %s" phase counter (string ex.Message)
                        Console.ReadLine() |> ignore
                | ex 
                        ->
                        printfn "%s" (string ex.Message)
                
                optimizer.step() |> ignore                
               
                match counter % 20 = 0 with
                | true  -> printfn "%s Epoch %d, Loss: %.4f, Perplexity: %.4f" phase counter (loss.item<float32>()) perplexity                       
                | false -> ()
            )
    
    // Generates tokens as part of the inference process
    // not tail-recursive
    let rec [<TailCall>] private generate (model: torch.nn.Module<torch.Tensor, torch.Tensor>) (inputSeq: torch.Tensor) steps maxSteps acc contextSize (temp: float32) (topK: int64) (strategy: Strategy) =
        
        let trimInput (input: torch.Tensor) =

            match input.shape.[1] > contextSize with
            | true -> input.narrow(1, -contextSize, contextSize)
            | false -> input

        let sampleLogits (logits: torch.Tensor) =
            
            match strategy with
            | Top_k 
                ->
                let struct (probs, indices) = torch.topk(logits / temp, int (min topK (int64 vocabSize)), dim = 0)
                let probs = softmax(probs, dim = 0L)
                let idx = torch.multinomial(probs, 1).item<int64>()
                indices.[idx].item<int64>()

            | Top_p ->
                // Apply temperature to logits
                let scaledLogits: torch.Tensor = logits / temp
                // Compute probabilities via softmax
                use probs: torch.Tensor = softmax(scaledLogits, dim = 0L)
                // Sort probabilities in descending order with corresponding indices
                let struct (sortedProbs: torch.Tensor, sortedIndices: torch.Tensor) = torch.sort(probs, dim = 0L, descending = true)
                // Compute cumulative sum of sorted probabilities
                use cumulativeProbs: torch.Tensor = torch.cumsum(sortedProbs, dim = 0L)
                // Create a boolean mask where cumulative probability exceeds p
                let pTensor =
                    match cumulativeProbs.dtype with
                    | torch.ScalarType.Float32 ->
                        torch.tensor(0.9f, dtype=torch.ScalarType.Float32, device=cumulativeProbs.device)
                    | torch.ScalarType.Float64 ->
                        torch.tensor(0.9, dtype=torch.ScalarType.Float64, device=cumulativeProbs.device)
                    | _ ->
                        failwithf "Unsupported dtype for cumulativeProbs: %A" cumulativeProbs.dtype
                use mask: torch.Tensor = torch.gt(cumulativeProbs, pTensor)
                // Get the first index to exclude (or use all if none exceed p)
                use nonzero: torch.Tensor = torch.nonzero(mask)
                let cutoff: int64 =
                   match nonzero.shape.[0] with
                   | 0L -> sortedProbs.shape.[0]  // Use all tokens
                   | _  -> nonzero.[0].item<int64>()
                // Slice to keep only tokens up to cutoff (nucleus)
                use probsTopP: torch.Tensor = sortedProbs.narrow(0L, 0L, cutoff)
                use indicesTopP: torch.Tensor = sortedIndices.narrow(0L, 0L, cutoff)
                // Renormalize probabilities to sum to 1
                use probsRenormalised: torch.Tensor = probsTopP / probsTopP.sum()
                // Sample an index from the nucleus
                let idx: int64 = torch.multinomial(probsRenormalised, 1).item<int64>()
                // Map sampled index to original token ID
                indicesTopP.[idx].item<int64>() 
            
            | Greedy
                ->
                torch.argmax(logits, dim = 0).item<int64>()
            
            | S 
                ->
                failwithf "Unsupported sampling strategy: %A" S

        match steps >= maxSteps with
        | true  ->
                List.rev acc
        | false ->
                use _ = torch.no_grad()
                
                // PREDICTING THE NEXT TOKEN BY COMPUTING LOGITS AND APPLYING SAMPLING (TOP-K, TEMPERATURE) 
                use trimmedInput = trimInput inputSeq
                use logits: torch.Tensor = model.forward trimmedInput
                use lastLogit: torch.Tensor = logits.select(0, 0L).select(0, -1L)
                let nextToken = sampleLogits lastLogit
                
                match nextToken with
                | tok 
                    when tok = Settings2.eosTokenIdx || tok = Settings2.padTokenIdx 
                        ->
                        List.rev (nextToken::acc)
                | _ 
                        ->
                        use newInput = torch.cat([|inputSeq; torch.tensor([|nextToken|], device = inputSeq.device).unsqueeze(0L)|], dim = 1L)
                        generate model newInput (steps + 1) maxSteps (nextToken::acc) contextSize temp topK strategy
                
    let internal main () =

        use scope = torch.NewDisposeScope()
        
        // CUDA® is a parallel computing platform and programming model developed by NVIDIA for general computing on graphical processing units (GPUs).
        let device = match torch.cuda.is_available() with true -> torch.CUDA | false -> torch.CPU

        printfn "Using device: %A" <| (string device).ToUpper()
        
        let dataset = TextData.getSequences()
        
        // CREATING INPUT-TARGET PAIRS FROM TEXT DATA USING A TOKENIZER
        let (inputData, targetData) = Tokenizer21.createInputTargetPairs dataset
        
        // LOADING INPUT FOR MODEL TRAINING
        use input = torch.tensor(inputData, device = device)
        
        // LOADING TARGET (SHIFTED ONE POSITION TO PREDICT THE NEXT TOKEN) FOR MODEL TRAINING
        use target = torch.tensor(targetData, device = device)
        
        printfn "Starting pre-training..."
       
        // INITIALIZING THE MODEL WITH EMBEDDINGS AND DECODER LAYERS
        use model = (new Transformer(int64 vocabSize, dModel, nHeads, numLayers)).``to``(device)
        // OPTIONALLY LOAD PRE-TRAINED GPT-2 WEIGHTS INTO MODEL HERE (AFTER INITIALIZATION), 
        //for example model.load_state_dict(torch.load("gpt2_checkpoint.pt"))        
        
        // INITIALIZING THE LOSS FUNCTION USED IN TRAINING
        // The CrossEntropyLoss (lossFn) computes the loss out of the logits using mathematical operations (softmax and negative log-likelihood)    
        // The output of CrossEntropyLoss is a continuous function with respect to the input logits        
        // Loss functions are typically continuous (and often differentiable) to enable gradient-based optimization.        
        use lossFn = new CrossEntropyLoss()
        
        // INITIALIZING THE ADAM OPTIMIZER FOR PARAMETER UPDATES
        use optimizer = torch.optim.Adam(model.parameters(), lr = learningRate) //gradient-based optimization
        (*
        Parameters:
        learnable weights and biases of the embedding layer,
        decoder layers (QKV projection, output projection, FFN, layer norms), 
        final layer norm, 
        and output layer (W, b)       
        *)
        
        // SETTING THE MODEL TO TRAINING MODE FOR PRE-TRAINING
        model.train()
        
        // PRE-TRAINING THE MODEL ON INPUT-TARGET DATA (SEQUENCES)
        trainEpoch model optimizer lossFn input target epochs "Pre-training"
        
        printfn "Starting fine-tuning..."
        
        let (fineTuneInputData, fineTuneTargetData) = TextData.getFineTuningSequences ()
        
        use fineTuneInput = torch.tensor(fineTuneInputData, device = device)
        use fineTuneTarget = torch.tensor(fineTuneTargetData, device = device)
        
        // INITIALIZING THE ADAM OPTIMIZER FOR FINE-TUNING
        use fineTuneOptimizer = torch.optim.Adam(model.parameters(), lr = learningRate)     

        // SETTING THE MODEL TO TRAINING MODE FOR FINE-TUNING
        model.train()
        
        // FINE-TUNING THE MODEL ON FINE-TUNING INPUT-TARGET DATA (SEQUENCES)
        trainEpoch model fineTuneOptimizer lossFn fineTuneInput fineTuneTarget fineTuneEpochs "Fine-tuning"
        
        printfn "Generating sequence after fine-tuning..."

        model.``to``("cpu") |> ignore<torch.nn.Module<torch.Tensor, torch.Tensor>> //redundant here, but kept for future use
        
        // SETTING THE MODEL TO EVALUATION MODE FOR INFERENCE
        model.eval()
        
        // DEFINING THE INPUT SEQUENCE (PROMPT) FOR INFERENCE
        use inputSeq = (torch.tensor([|0L; 1L; 2L|], device = device).unsqueeze 0L).``to``(torch.CPU) //torch.CPU redundant here, but kept for future use
                
        printf "Generated sequence (token IDs): "
        
        // GENERATING THE OUTPUT SEQUENCE (EXPECTED TO BE [yellow, <eos>]) USING THE TRAINED MODEL
        let generated = generate model inputSeq 0 2 [] contextSize temp topK strategy // |> List.rev
        
        generated |> List.iter (printf "%d ")

        printfn "\n"

        printf "Generated sequence (words): "

        let results = generated |> List.map (fun id -> vocabulary |> List.item (int id))

        generated |> List.iter (fun id -> printf "%s " (vocabulary |> List.item (int id)))

        printfn "\n"

        model.Dispose()

        torch.CurrentDisposeScope.DisposeEverything()
        
        System.GC.Collect()