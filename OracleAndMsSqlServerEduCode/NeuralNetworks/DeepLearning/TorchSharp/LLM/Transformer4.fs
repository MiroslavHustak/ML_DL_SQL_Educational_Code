namespace NeuralNetworks //275

open System
open System.Threading.Tasks

open TorchSharp
open TorchSharp.Modules

open type torch.nn
open type torch.nn.functional

open Settings

//*******************************************************************
// GPT-4 in architecture (not in scale) without calling external tools 

// This code is still under development and is intended for educational purposes only.
// Submitting issues and pull requests is welcome.
//
// Option and Result types have not been applied yet.
// Separation of data and data manipulation not fully implemented yet.
// Disposals not fully implemented yet.
// The code architecture will be figured out later.
//
// Acknowledgments:
// - A lecture on creating an LLM with TorchSharp by Tomáš Herceg (https://www.youtube.com/watch?v=tW5RiP765hw&t=12s).
// - Sebastian Raschka, "Build a Large Language Model (From Scratch)".
//*******************************************************************

module Transformer_TorchSharp4 =  

    //*************************************************************  
    // MODEL ARCHITECTURE DEFINITION SECTION
    // GPT-4-like architecture (decoder-only Transformer with RoPE and RMSNorm)
    // Does not include FlashAttention, fused ops, production-grade tokenizer, checkpointing, LoRA, or quantisation
    //*************************************************************
        
    type private RMSNorm(normalizedShape: int64[], eps: float32) as self =

        inherit Module<torch.Tensor, torch.Tensor>("RMSNorm")

        let weight = torch.nn.Parameter(torch.ones(normalizedShape))
        let eps = torch.tensor eps
        do self.RegisterComponents()

        override _.forward (x: torch.Tensor) =

            use norm = x.pow(torch.tensor(2.0f)).mean([|-1L|], keepdim = true).add(eps).sqrt() 
            x / norm * weight

    type private TransformerDecoderLayer(dModel: int64, nHeads: int64, dropoutRate: float32) as self =

        inherit Module<torch.Tensor, torch.Tensor>("TransformerDecoderLayer")       

        let headDim = dModel / nHeads
        let kvHeads = nHeads / 4L |> max 1L
        let qProjection = Linear(dModel, dModel)
        let kProjection = Linear(dModel, kvHeads * headDim)
        let vProjection = Linear(dModel, kvHeads * headDim)
        let outputProjection = Linear(dModel, dModel)

        let feedForward1 = Linear(dModel, dModel * 4L)
        let feedForward2 = Linear(dModel * 4L, dModel)
        let layerNorm1 = new RMSNorm([|dModel|], 1e-5f)
        let layerNorm2 = new RMSNorm([|dModel|], 1e-5f)
        let dropout = Dropout(float dropoutRate)      

        do self.RegisterComponents()

        override _.forward x =

            //Rotary Positional Embeddings
            let applyRotary (q: torch.Tensor) (k: torch.Tensor) : torch.Tensor * torch.Tensor =

                let lastDim = q.shape.[q.shape.Length - 1]
            
                match lastDim % 2L <> 0L with true -> failwithf "The last dimension (%d) is not even, cannot apply rotary split." lastDim | false -> ()
            
                let dim = lastDim / 2L
            
                use theta = torch.pow(10000.0f, torch.arange(0L, dim, device=q.device, dtype=torch.float32) / float32 dim)
                let seqLen = q.shape.[q.shape.Length - 2]
                use positionIds = torch.arange(seqLen, device=q.device, dtype=torch.float32).unsqueeze(-1)
                use freqs = positionIds / theta
                use sin = torch.sin freqs
                use cos = torch.cos freqs
    
                let reshapeForRotation (x: torch.Tensor) =

                    let split = x.split([|dim; dim|], -1L)
                    match split.Length <> 2 with
                    | true  -> failwithf "Split did not return two tensors. Split length: %d, dim: %A, x.shape: %A" split.Length dim x.shape
                    | false -> ()
                
                    use a = split |> Array.head
                    use b = split |> Array.last

                    torch.cat([|(a * cos) - (b * sin); (a * sin) + (b * cos)|], dim = -1)
    
                reshapeForRotation q, reshapeForRotation k 

            let getAlibiBias (nHeads: int64) (seq: int64) (device: torch.Device) = //Attention with linear biases

                use slopes = torch.linspace(1.0, 0.0, int nHeads, dtype = torch.float32, device = device).unsqueeze(-1).unsqueeze(-1)
                use bias = torch.arange(seq, device = device).unsqueeze(0).unsqueeze(0).float()
                (*) slopes bias

            match x.shape with
            | [|batch; seq; dmodel|] 
                when dmodel = dModel
                    ->
                    use normedInput = layerNorm1.forward x

                    //With my model, no big difference when run sequentially, with .NET tasks (directly or with Array.Parallel), or with async workflows
                    let result = 
                        match torch.cuda.is_available() with
                        | true 
                            -> 
                            // GPU path: batched execution to reduce kernel overhead
                            //TODO: VERIFY THIS CODE WITH GPUs
                            let q = qProjection.forward(normedInput.view([|batch * seq; dModel|])).view([|batch; seq; nHeads; headDim|]).transpose(1, 2)
                            let k = kProjection.forward(normedInput.view([|batch * seq; dModel|])).view([|batch; seq; kvHeads; headDim|]).transpose(1, 2)
                            let v = vProjection.forward(normedInput.view([|batch * seq; dModel|])).view([|batch; seq; kvHeads; headDim|]).transpose(1, 2)
                            let q = q.cuda()
                            let k = k.cuda()
                            let v = v.cuda()
                            let qkv = torch.cat([| q; k; v |], dim=2L) 

                            [
                                qkv.slice(1L, 0L, int64 nHeads, headDim)  // q: [batch, nHeads, seq, headDim]
                                qkv.slice(1L, int64 nHeads, int64 (nHeads + kvHeads), headDim)  // k: [batch, kvHeads, seq, headDim]
                                qkv.slice(1L, int64 (nHeads + kvHeads), int64 (nHeads + 2L * kvHeads), headDim)  // v: [batch, kvHeads, seq, headDim]
                            ]

                        | false 
                            -> //CPU path
                            [
                                (fun () -> qProjection.forward(normedInput.view([|batch * seq; dModel|])).view([|batch; seq; nHeads; headDim|]).transpose(1, 2))
                                (fun () -> kProjection.forward(normedInput.view([|batch * seq; dModel|])).view([|batch; seq; kvHeads; headDim|]).transpose(1, 2))
                                (fun () -> vProjection.forward(normedInput.view([|batch * seq; dModel|])).view([|batch; seq; kvHeads; headDim|]).transpose(1, 2))
                            ]
                            |> List.Parallel.map (fun f -> f())
                    
                    use q = result |> List.head
                    use k = result |> List.item 1
                    use v = result |> List.last
                    
                    (*
                    let qTask = Task.Run (fun _ -> qProjection.forward ...)
                    let kTask = Task.Run (fun _ -> kProjection.forward ...)                      
                    let vTask = Task.Run (fun _ -> vProjection.forward ...)   

                    Task.WaitAll([| qTask :> Task; kTask :> Task; vTask :> Task |])
                    
                    use q = qTask.Result
                    use k = kTask.Result
                    use v = vTask.Result  
                    *)

                    let qRoPE, kRoPE = applyRotary q k

                    use qRoPE = qRoPE
                    use kRoPE = kRoPE

                    //positional information is integrated directly into the attention mechanism via RoPE (for q and k) and ALiBi (for attention scores)                    
                    use attentionScores : torch.Tensor = torch.matmul(qRoPE, kRoPE.transpose(-2, -1)) / sqrt(float headDim)
                    use alibiBias = getAlibiBias nHeads seq attentionScores.device
                    use attentionScoresWithBias : torch.Tensor = (+) attentionScores alibiBias

                    //preventing tokens from attending to future ones
                    use mask : torch.Tensor = torch.triu(torch.ones([|seq; seq|], device = attentionScores.device), diagonal = 1L).to_type(torch.ScalarType.Bool)
                    use maskedScores = attentionScoresWithBias.masked_fill(mask.unsqueeze(0).unsqueeze(0), System.Single.NegativeInfinity)
                    use attentionWeights = softmax(maskedScores, -1L) |> dropout.forward
                       
                    let result = 
                        [
                            (fun () -> torch.matmul(attentionWeights, v).transpose(1, 2).contiguous().view(batch, seq, dModel) |> outputProjection.forward)
                            (fun () -> normedInput |> layerNorm2.forward |> feedForward1.forward |> gelu |> feedForward2.forward |> dropout.forward)
                        ]
                        |> List.Parallel.map (fun f -> f())

                    use attnResult = result |> List.head
                    use ffnResult = result |> List.last 
                   
                    (*
                    let attnTask = 
                        Task.Run
                            (fun _
                                -> torch.matmul(attentionWeights, v).transpose(1, 2).contiguous().view(batch, seq, dModel) |> outputProjection.forward
                            )

                    let ffnTask = 
                        Task.Run
                            (fun _
                                -> normedInput |> layerNorm2.forward |> feedForward1.forward |> gelu |> feedForward2.forward |> dropout.forward
                            )

                    Task.WaitAll([| attnTask :> Task; ffnTask :> Task |])

                    use attnResult = attnTask.Result
                    use ffnResult = ffnTask.Result
                     *)

                    //Skip (residual) connections with scaling - prevent exploding activations in very deep networks (not used in my GPT-2-like model)
                    use residualScale = 1.0f / float32 (2 * numLayers) |> torch.tensor

                    use out1 = x + attnResult * residualScale //The first skip (residual) connection after multi-head attention

                    out1 + ffnResult * residualScale //The second residual connection after feed-forward network

            | shapeArr -> failwithf "Invalid input shape: %A" shapeArr

    type private Transformer(vocabSize: int64, dModel: int64, nHeads: int64, numLayers: int, device: torch.Device) as self =

        inherit Module<torch.Tensor, torch.Tensor>("Transformer")

        let embedding = Embedding(vocabSize, dModel)
        let dropout = Dropout(float dropoutRate)

        let decoderLayers =
            List.init numLayers (fun _ -> new TransformerDecoderLayer(dModel, nHeads, dropoutRate) :> torch.nn.Module<torch.Tensor, torch.Tensor>)
            |> List.toArray
            |> ModuleList<torch.nn.Module<torch.Tensor, torch.Tensor>>

        let outputLayer = Linear(dModel, vocabSize)
        let norm = new RMSNorm([|dModel|], 1e-5f)

        do outputLayer.weight <- embedding.weight

        do self.RegisterComponents()

        override _.forward x =

            use emb = embedding.forward x
            use embWithDropout = dropout.forward emb //embeddings remain position-free            
            use decoderOut = decoderLayers |> Seq.fold (fun acc (layer: torch.nn.Module<torch.Tensor, torch.Tensor>) -> layer.forward acc) embWithDropout 
            use normOut = norm.forward decoderOut

            outputLayer.forward(normOut).to_type(torch.ScalarType.Float32)    
   
    //*************************************************************   
    // For mathematicians:
    // PARAMETER ESTIMATION AND GRADIENT-BASED OPTIMISATION SECTION

    // For others:
    // MODEL TRAINING SECTION (USED FOR PRE-TRAINING AND FINE-TUNING)
    //*************************************************************

    let private trainEpoch (model: torch.nn.Module<torch.Tensor, torch.Tensor>) (optimizer: torch.optim.Optimizer)
                           (lossFn: CrossEntropyLoss) (input: torch.Tensor) (target: torch.Tensor) maxEpochs phase =

        //Learning rate scheduler (includes warmup and cosine decay)  
        let learningRateSchedule (warmupSteps: int) (totalSteps: int) (baseLr: float32) (step: int) : float32 =

            match step with
            | s when
                s < warmupSteps
                    ->
                    float32 s / float32 warmupSteps * baseLr
            | s when
                s < totalSteps
                    ->
                    let progress = float32 (s - warmupSteps) / float32 (totalSteps - warmupSteps)
                    baseLr * 0.5f * (1.0f + cos (MathF.PI * progress))
            | _ 
                    ->
                    0.0f       

        [0 .. maxEpochs - 1]
        |> List.iter
            (fun epoch 
                ->
                let counter = (+) epoch 1

                let baseLr = 5e-4f //5 * 10^-4  //adapt
                let warmupSteps = 10  //adapt

                // Set learning rate dynamically based on schedule
                let currentLr : float32 = learningRateSchedule warmupSteps maxEpochs baseLr epoch

                let paramGroups = optimizer.ParamGroups
                match paramGroups |> Seq.length > 0 with
                | true  ->
                        let paramGroup = paramGroups |> Seq.head 
                        //paramGroup.LearningRate <- float currentLr  //uncomment and use with a better PC than I possess
                        () //comment out if you use a better PC than me
                | false ->
                        failwith "No parameter groups found in optimizer"

                optimizer.zero_grad() // Clears old gradients from the previous step before the next backward pass 

                use output = model.forward input
                use loss = lossFn.forward(output.contiguous().view(-1L, int64 vocabSize), target.contiguous().view(-1L)) //computes gradients

                let perplexity = torch.exp(loss).item<float32>() 

                try
                    loss.backward() //computes loss
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0) |> ignore<float> //Gradient clipping limits the combined size (norm) of all parameter gradients to 1.0
                with
                | :? System.StackOverflowException 
                    as ex 
                    ->
                    printfn "StackOverflowException in %s, epoch %d: %s" phase counter (string ex.Message)
                | ex
                    ->
                    printfn "%s" (string ex.Message)

                //model change happens explicitly in optimizer.step() - changed model is the result of trainEpoch (this side effect is a nightmare for a functional programmer)   
                optimizer.step() |> ignore<torch.Tensor>   //update parameters based on gradients (Adam is actually run here)            
        
                match counter % 20 with
                | 0 -> printfn "%s Epoch %d, Loss: %.4f, Perplexity: %.4f" phase counter (loss.item<float32>()) perplexity
                | _ -> ()
            )                    

    //*************************************************************         
    // INFERENCE SECTION
    //*************************************************************    

    //Continuation passing style (collection functions do not work; tail-recursivity not possible to establish) 
    let rec private generateSeq (model: torch.nn.Module<torch.Tensor, torch.Tensor>) (inputSeq: torch.Tensor) (steps: int) 
                                (acc: int64 list) (cont: int64 list -> 'a) : 'a =

        let trimInput (input: torch.Tensor) =

            match input.shape.[1] > contextSize with
            | true  -> input.narrow(1, int64 (int input.shape.[1] - int contextSize), contextSize)
            | false -> input

        let sampleLogits (logits: torch.Tensor) =

            match strategy with
            | Top_k 
                ->
                let struct (probs, indices) = torch.topk(logits / temp, int (min topK (int64 vocabSize)), dim = 0)
                use indices = indices
                use probs = softmax(probs, dim = 0L)
                let idx = torch.multinomial(probs, 1).item<int64>()
                indices.[idx].item<int64>()

            | Top_p
                ->
                use scaledLogits : torch.Tensor = logits / temp
                use probs : torch.Tensor = softmax(scaledLogits, dim = 0L)
                let struct (sortedProbs: torch.Tensor, sortedIndices: torch.Tensor) = torch.sort(probs, dim = 0L, descending = true)
                use sortedProbs = sortedProbs
                use sortedIndices = sortedIndices
                use cumulativeProbs: torch.Tensor = torch.cumsum(sortedProbs, dim = 0L)
            
                let pTensor = torch.tensor(float32 p, device = cumulativeProbs.device) //C++ always gives a warning about ignoring complex numbers when applying dtype
                    (*
                    match cumulativeProbs.dtype with
                    | torch.ScalarType.Float32 -> torch.tensor(float32 p, dtype=torch.ScalarType.Float32, device = cumulativeProbs.device)
                    | torch.ScalarType.Float64 -> torch.tensor(p, dtype=torch.ScalarType.Float64, device = cumulativeProbs.device)
                    | _                        -> failwithf "Unsupported dtype for cumulativeProbs: %A" cumulativeProbs.dtype
                    *)
            
                use mask : torch.Tensor = torch.gt(cumulativeProbs, pTensor)
                use nonzero : torch.Tensor = torch.nonzero mask
            
                let cutoff : int64 =
                    match nonzero.shape.[0] with
                    | 0L -> sortedProbs.shape.[0]  // Use all tokens
                    | _  -> nonzero.[0].item<int64>()

                let cutoff = match cutoff with 0L -> sortedProbs.shape.[0] | _ -> cutoff
                
                use probsTopP = sortedProbs.narrow(0L, 0L, cutoff)
                use indicesTopP = sortedIndices.narrow(0L, 0L, cutoff)
                let probsTopPSum = probsTopP.sum().item<float32>()

                let idx =
                    match cutoff, probsTopPSum with
                    | 0L, _
                        -> 
                        torch.multinomial(probs, 1).item<int64>()

                    | _, 0.0f 

                        ->
                        torch.multinomial(probs, 1).item<int64>()

                    | _, _ 
                        ->
                        use probsRenormalised = probsTopP / probsTopP.sum()
                        torch.multinomial(probsRenormalised, 1).item<int64>()

                indicesTopP.[idx].item<int64>() 
                
            | Greedy 
                ->
                torch.argmax(logits, dim = 0).item<int64>()

            | S ->
                failwith "Unsupported sampling strategy"

        match steps >= maxSteps with
        | true  -> 
                List.rev >> cont <| acc  //CPS applied // Stop recursion if the maximum number of steps is reached
        | false ->
                use _ = torch.no_grad()
                use trimmedInput = trimInput inputSeq
                use logits = model.forward trimmedInput
                use lastLogit = logits.select(0, 0L).select(0, -1L)
    
                let nextToken = sampleLogits lastLogit

                match nextToken = eosTokenIdx || nextToken = padTokenIdx with
                | true  ->
                        List.rev >> cont <| nextToken :: acc  //CPS applied
                | false ->
                        let newInput = torch.cat([|inputSeq; torch.tensor([|nextToken|], device = inputSeq.device).unsqueeze(0L)|], dim = 1L)
                        generateSeq model newInput (steps + 1) (nextToken :: acc) cont

    let internal main () =

        //HELPERS
        use scope = torch.NewDisposeScope()

        let device = 
            match torch.cuda.is_available() with
            | true  -> torch.CUDA
            | false -> torch.CPU
        
        printfn "Using device: %A" <| (string device).ToUpper()

        // DATASET SIMULATION
        let dataset = TextData2.getSequences()

        // TOKENIZER
        let (inputData, targetData) = Tokenizer2.createInputTargetPairs dataset
        use input = torch.tensor(inputData, device = device)
        use target = torch.tensor(targetData, device = device)

        // GRADIENT-BASED PARAMETER* OPTIMIZATION (PRE-TRAINING) 
        // * weights, biases, potentially learned positional encodings, layer normalization parameters
        printfn "Starting pre-training..."

        use model : torch.nn.Module<torch.Tensor, torch.Tensor> = (new Transformer(int64 vocabSize, dModel, nHeads, numLayers, device)).``to``(device)
        use lossFn = new CrossEntropyLoss(ignore_index = padTokenIdx)
        use optimizer = torch.optim.Adam(model.parameters(), lr = learningRate) //Adam (Adaptive Moment Estimation) = gradient-based optimization algorithm //just configuration
        
        //Uncomment for pre-training
        model.train() //Setting the model for the pre-training mode
        
        //Uncomment for pre-training
        trainEpoch model optimizer lossFn input target epochs "Pre-training"

        // FINE-TUNING 
        printfn "Starting fine-tuning..."
        
        let (fineTuneInputData, fineTuneTargetData) = TextData2.getFineTuningCausalLMSequences ()
        use fineTuneInput = torch.tensor(fineTuneInputData, device = device)
        use fineTuneTarget = torch.tensor(fineTuneTargetData, device = device)
        use fineTuneOptimizer = torch.optim.Adam(model.parameters(), lr = learningRate)

        //Uncomment for fine-tuning
        model.train() //Setting the model for the fine-tuning mode

        //Uncomment for fine-tuning
        trainEpoch model fineTuneOptimizer lossFn fineTuneInput fineTuneTarget fineTuneEpochs "Fine-tuning"

        //Uncomment for saving weights and biases
        model.save("model4.pt") |> ignore<torch.nn.Module>
        model.load("model4.pt") |> ignore<torch.nn.Module>

        // INFERENCE
        printfn "Generating sequence after fine-tuning..."
               
        let promptContent = "What is the colour of the Sun? <sep>"
        let promptTokens = Tokenizer2.tokenize promptContent |> List.toArray

        model.``to``("cpu") |> ignore<torch.nn.Module<torch.Tensor, torch.Tensor>>

        model.eval() //Configures model to evaluation mode for sequence generation 

        use inputSeq = torch.tensor(promptTokens, device = torch.CPU).unsqueeze 0L
        
        //Generating the output sequence       
        let generated = generateSeq model inputSeq initialStep acc id
                       
        printf "Generated sequence (token IDs): "
        generated |> List.iter (printf "%d ")

        printfn "\n"
        
        printf "Generated sequence (words): "
      
        generated 
        |> List.iter
            (fun id 
                ->
                match id >= 0L && id < int64 vocabulary.Length with
                | true  -> printf "%s " (vocabulary |> List.item (int id))
                | false -> printf "<unk> "
            )

        printfn "\n"
               
        //*************************************************************         
        // DISPOSAL SECTION
        //*************************************************************
        model.Dispose()
        torch.CurrentDisposeScope.DisposeEverything() // see torch.NewDisposeScope() above
        System.GC.Collect()