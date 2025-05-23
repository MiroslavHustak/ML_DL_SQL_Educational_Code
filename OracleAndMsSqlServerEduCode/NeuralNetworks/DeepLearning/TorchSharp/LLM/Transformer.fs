﻿namespace NeuralNetworks

open System
open TorchSharp
open TorchSharp.Modules

open type torch.nn
open type torch.nn.functional

open Settings

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
            
            use mask = torch.triu(torch.ones([|seq; seq|], device = attentionScores.device), diagonal = 1L).to_type(torch.ScalarType.Bool)
            use maskedScores = attentionScores.masked_fill(mask.unsqueeze(0).unsqueeze(0), System.Single.NegativeInfinity)
            
            use attentionWeights = softmax(maskedScores, -1L) |> dropout.forward //softmax -> normalization (sum of attention weights to be 1)
            
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
        let posEnc = getPositionalEncodings 1024L dModel
        let dropout = Dropout(float dropoutRate)
        
        //hidden layers of the transformer neural network (performing the core transformations between the input embeddings and the output logits)
        let decoderLayers =

            let createLayer _ = new TransformerDecoderLayer(dModel, nHeads, dropoutRate) :> torch.nn.Module<torch.Tensor, torch.Tensor>
            let decoderLayersList = List.init numLayers createLayer
            let decoderLayersArray = List.toArray decoderLayersList
            
            ModuleList<torch.nn.Module<torch.Tensor, torch.Tensor >> decoderLayersArray
        
        let outputLayer = Linear(dModel, vocabSize) // LOAD PRE-TRAINED GPT-2 OUTPUT LAYER WEIGHTS AND BIASES HERE (MAY BE TIED TO EMBEDDING WEIGHTS)
        let norm = LayerNorm [|dModel|] // LOAD PRE-TRAINED GPT-2 FINAL LAYER NORM SCALE AND SHIFT HERE

        //do outputLayer.weight <- embedding.weight //in case of using pre-trained weights //GPT-2 typically ties embedding and output weights        
        
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
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0) |> ignore
                with
                | :? System.StackOverflowException as ex ->
                    printfn "StackOverflowException in %s, epoch %d: %s" phase (epoch + 1) (string ex.Message)
                    Console.ReadLine() |> ignore
                | ex -> printfn "%s" (string ex.Message)
                
                optimizer.step() |> ignore
                
                let counter = (+) epoch 1
                match counter % 100 = 0 && counter > 100 with
                | true  -> printfn "%s Epoch %d, Loss: %.4f, Perplexity: %.4f" phase counter (loss.item<float32>()) perplexity
                | false -> ()
                
                System.GC.Collect()
            )
    
    // Generates tokens as part of the inference process
    let rec [<TailCall>] private generate (model: torch.nn.Module<torch.Tensor, torch.Tensor>) (inputSeq: torch.Tensor) steps maxSteps acc contextSize (temp: float32) (topK: int64) (strategy: string) =
        
        let trimInput (input: torch.Tensor) (contextSize: int) =

            match input.shape.[1] > contextSize with
            | true  -> input.narrow(1, -contextSize, contextSize) 
            | false -> input
        
        let sampleLogits (logits: torch.Tensor) (temp: float32) (topK: int64) (strategy: string) (vocabSize: int64) =

            let effectiveTopK = min topK vocabSize
            
            match effectiveTopK <= 0L with
            | true  -> failwith "topK must be positive and not exceed vocabulary size"
            | false -> ()
            
            match strategy with
            | "top-k"
                ->
                let struct (probs, indices) = torch.topk(logits / temp, int effectiveTopK, dim = 0)
                let probs = softmax(probs, dim = 0L)
                let idx = torch.multinomial(probs, 1).item<int64>()
                indices.[idx].item<int64>()

            | "greedy" 
                ->
                torch.argmax(logits, dim=0).item<int64>()

            | _ 
                ->
                failwith $"Unsupported sampling strategy: {strategy}"
        
        match steps with
        | s when s >= maxSteps 
            ->
            acc
        | _ ->
            let _ = torch.no_grad()
      
            // PREDICTING THE NEXT TOKEN BY COMPUTING LOGITS AND APPLYING SAMPLING (TOP-K, TEMPERATURE) 
            let adjustedLogit: torch.Tensor =
                let trimmedInput = trimInput inputSeq contextSize
                let logits: torch.Tensor = model.forward trimmedInput
                let lastLogit: torch.Tensor = logits.select(0, 0L).select(0, -1L)
                
                match steps with
                | 0 -> lastLogit.index_fill_(0, torch.tensor([|7L|], device = lastLogit.device), System.Single.NegativeInfinity)
                | _ -> lastLogit
            
            let nextToken: int64 = sampleLogits adjustedLogit temp topK strategy (int64 vocabSize)
            let newAcc: int64 list = nextToken :: acc
            
            match nextToken with
            | 7L 
                ->
                newAcc
            | _ 
                ->
                let newInput: torch.Tensor = torch.cat([|inputSeq; torch.tensor([|nextToken|], device = inputSeq.device).unsqueeze(0L)|], dim = 1L)
                generate model newInput (steps + 1) maxSteps newAcc contextSize temp topK strategy
    
    let internal main () =
        
        // CUDA® is a parallel computing platform and programming model developed by NVIDIA for general computing on graphical processing units (GPUs).
        let device = match torch.cuda.is_available() with true -> torch.CUDA | false -> torch.CPU

        printfn "Using device: %A" <| (string device).ToUpper()
        
        let dataset = TextData.getSequences()
        
        // CREATING INPUT-TARGET PAIRS FROM TEXT DATA USING A TOKENIZER
        let (inputData, targetData) = Tokenizer.createInputTargetPairs dataset
        
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
        
        // SETTING THE MODEL TO EVALUATION MODE FOR INFERENCE
        model.eval()
        
        // DEFINING THE INPUT SEQUENCE (PROMPT) FOR INFERENCE
        use inputSeq = torch.tensor([|0L; 1L; 2L|], device = device).unsqueeze 0L
        
        printf "Generated sequence (token IDs): "
        
        // GENERATING THE OUTPUT SEQUENCE (EXPECTED TO BE [yellow, <eos>]) USING THE TRAINED MODEL

        let generated = generate model inputSeq 0 2 [] contextSize 0.7f topK strategy |> List.rev
        
        generated |> List.iter (printf "%d ")
        printfn "\n"
        
        printf "Generated sequence (words): "
        generated |> List.iter (fun id -> printf "%s " (vocabulary |> List.item (int id)))
        printfn "\n"