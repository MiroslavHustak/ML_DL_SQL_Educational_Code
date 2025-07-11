namespace NeuralNetworks2 //237

open System
open System.Threading.Tasks

open TorchSharp
open TorchSharp.Modules

open type torch.nn
open type torch.nn.functional

open Settings2

//*******************************************************************
// GPT-2

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
// - Giles Thomas, Giles' blog, Writing an LLM from scratch (https://www.gilesthomas.com).
//*******************************************************************

module Transformer_TorchSharp2 =   

    //*************************************************************  
    // MODEL ARCHITECTURE DEFINITION SECTION
    // GPT-2-like architecture (decoder-only Transformer)
    // Does not include production-grade tokenizer
    //*************************************************************    

    type private TransformerDecoderLayer(dModel: int64, nHeads: int64, dropoutRate: float32) as self =

        inherit Module<torch.Tensor, torch.Tensor>("TransformerDecoderLayer")

        let qkvProjection = Linear(dModel, dModel * 3L)
        let outputProjection = Linear(dModel, dModel)
        let feedForward1 = Linear(dModel, dModel * 4L)
        let feedForward2 = Linear(dModel * 4L, dModel)
        let layerNorm1 = LayerNorm [|dModel|]
        let layerNorm2 = LayerNorm [|dModel|]
        let dropout = Dropout(float dropoutRate)

        do self.RegisterComponents()

        override _.forward x =

            match x.shape with
            | [|batch; seq; dmodel|] 
                when dmodel = dModel
                ->
                match dModel % nHeads <> 0L with
                | true  -> failwithf "%s (%d) must be divisible by %s (%d)" (nameof dModel) dModel (nameof nHeads) nHeads
                | false -> ()

                let headDim = dModel / nHeads
                let reshapedShape = [|batch; seq; 3L; nHeads; headDim|]

                use normedInput1 = layerNorm1.forward x //applied before the attention block (qkvProjection.forward) => pre-norm.
                use qkv = qkvProjection.forward normedInput1
                use qkvReshaped = qkv.view reshapedShape

                //use q = qkvReshaped.select(2, 0L).transpose(1, 2)
                //use k = qkvReshaped.select(2, 1L).transpose(1, 2)
                //use v = qkvReshaped.select(2, 2L).transpose(1, 2)
                
                let result = 
                    [
                        (fun () -> qkvReshaped.select(2, 0L).transpose(1, 2))
                        (fun () -> qkvReshaped.select(2, 1L).transpose(1, 2))
                        (fun () -> qkvReshaped.select(2, 2L).transpose(1, 2))
                    ]
                    |> List.Parallel.map (fun f -> f())

                use q = result |> List.head
                use k = result |> List.item 1
                use v = result |> List.last 
                
                (*
                let qTask = Task.Run (fun () -> qkvReshaped.select(2, 0L).transpose(1, 2))
                let kTask = Task.Run (fun () -> qkvReshaped.select(2, 1L).transpose(1, 2))
                let vTask = Task.Run (fun () -> qkvReshaped.select(2, 2L).transpose(1, 2))

                Task.WaitAll([| qTask :> Task; kTask :> Task; vTask :> Task |])
                
                use q = qTask.Result
                use k = kTask.Result
                use v = vTask.Result                        
                *)

                use attentionScores = torch.matmul(q, k.transpose(-2, -1)) / sqrt(float headDim)
                use mask = torch.triu(torch.ones([|seq; seq|], device = attentionScores.device), diagonal = 1L).to_type(torch.ScalarType.Bool)
                use maskedScores = attentionScores.masked_fill(mask.unsqueeze(0).unsqueeze(0), System.Single.NegativeInfinity)
                use attentionWeights = softmax(maskedScores, -1L) |> dropout.forward

                use contextVector =
                    torch.matmul(attentionWeights, v)
                    |> fun cv -> cv.transpose(1, 2)
                    |> fun cv -> cv.contiguous()
                    |> fun cv -> cv.view(batch, seq, dModel)
                 
                //two skip (residual) connections per decoder layer — one for attention, one for feed-forward  
                use attnOutput = outputProjection.forward contextVector
                use out1 = x + attnOutput //Skip (residual) connection 1

                use normedInput2 = layerNorm2.forward out1 //applied before the feed-forward block (feedForward1.forward) => pre-norm.
                use ff1 = feedForward1.forward normedInput2
                use activated = gelu ff1
                use ff2 = feedForward2.forward activated
                use ffOutput = dropout.forward ff2

                out1 + ffOutput //Skip (residual) connection 2

                //Post-Norm is rarely used in modern architectures except for legacy reasons

            | shapeArr 
                ->
                failwithf "Input tensor must have shape [batch; seq; %s] but got %A" (nameof dModel) shapeArr

    type private Transformer(vocabSize: int64, dModel: int64, nHeads: int64, numLayers: int, device: torch.Device) as self =

        inherit Module<torch.Tensor, torch.Tensor>("Transformer")
        
        // Sinusoidal non-learnable positional encodings
        let getPositionalEncodings (seqLen: int64) (dModel: int64) (device: torch.Device) : torch.Tensor =
        
            use position = torch.arange(seqLen, dtype = torch.float32, device = device).unsqueeze(1)
            use divTerm = torch.exp(torch.arange(0L, dModel, 2L, dtype = torch.float32, device = device) * (-Math.Log 10000.0 / float dModel))
                
            let encodings = torch.zeros([|seqLen; dModel|], device = device)  //use is not possible here
            encodings.index_copy_(1, torch.arange(0L, dModel, 2L, device = device), torch.sin(position * divTerm)) |> ignore<torch.Tensor>
            encodings.index_copy_(1, torch.arange(1L, dModel, 2L, device = device), torch.cos(position * divTerm)) |> ignore<torch.Tensor>
            encodings  

        let embedding = Embedding(vocabSize, dModel)
        let posEnc = getPositionalEncodings 128L dModel device  
        let dropout = Dropout(float dropoutRate)

        let decoderLayers =
            List.init numLayers (fun _ -> new TransformerDecoderLayer(dModel, nHeads, dropoutRate) :> torch.nn.Module<torch.Tensor, torch.Tensor>)
            |> List.toArray
            |> ModuleList<torch.nn.Module<torch.Tensor, torch.Tensor>>

        let outputLayer = Linear(dModel, vocabSize)
        let norm = LayerNorm [|dModel|]

        do outputLayer.weight <- embedding.weight
        do self.register_buffer("posEnc", posEnc)
        do self.RegisterComponents()

        override _.forward x =

            use emb = embedding.forward x
            let seqLen = x.shape.[1]
            use embWithPos = emb + posEnc.narrow(0L, 0L, seqLen)
            use embWithPosDropped = dropout.forward embWithPos

            //layer.forward executes all the operations defined in the TransformerDecoderLayer, including self-attention, feed-forward networks, residual connections, and normalization
            use decoderOut = decoderLayers |> Seq.fold (fun acc (layer: torch.nn.Module<torch.Tensor, torch.Tensor>) -> layer.forward acc) embWithPosDropped 
            use normOut = norm.forward decoderOut 

            //output is a tensor of shape [batchSize; seqLen; vocabSize] //values are logits (unnormalized scores) in float32 format
            //output is ready to compute a loss for training or predict the next token during inference
            outputLayer.forward(normOut).to_type(torch.ScalarType.Float32) 

    //*************************************************************   
    // For mathematicians:
    // PARAMETER ESTIMATION AND GRADIENT-BASED OPTIMISATION SECTION

    // For others:
    // MODEL TRAINING SECTION (USED FOR PRE-TRAINING AND FINE-TUNING)
    //*************************************************************

    let private trainEpoch (model: torch.nn.Module<torch.Tensor, torch.Tensor>) (optimizer: torch.optim.Optimizer)
                           (lossFn: CrossEntropyLoss) (input: torch.Tensor) (target: torch.Tensor) maxEpochs phase =

        [0 .. maxEpochs - 1]
        |> List.iter
            (fun epoch 
                ->
                let counter = epoch + 1

                optimizer.zero_grad() // Clears old gradients from the previous step before the next backward pass                

                use output = model.forward input
                use loss = lossFn.forward(output.contiguous().view(-1L, int64 vocabSize), target.contiguous().view(-1L))

                let perplexity = torch.exp(loss).item<float32>() 

                try
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0) |> ignore<float> // limits the combined size (norm) of all parameter gradients to 1.0
                with
                | :? System.StackOverflowException 
                    as ex 
                    ->
                    printfn "StackOverflowException in %s, epoch %d: %s" phase counter (string ex.Message)
                | ex
                    ->
                    printfn "%s" (string ex.Message)

                //model change happens explicitly in optimizer.step() - changed model is the result of trainEpoch (this side effect is a nightmare for a functional programmer)  
                optimizer.step() |> ignore<torch.Tensor>    //update parameters based on gradients (Adam is actually run here)          
        
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
                use scaledLogits: torch.Tensor = logits / temp
                use probs: torch.Tensor = softmax(scaledLogits, dim = 0L)
                
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
            
                use mask: torch.Tensor = torch.gt(cumulativeProbs, pTensor)
                use nonzero: torch.Tensor = torch.nonzero(mask)
            
                let cutoff: int64 =
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
        let (inputData, targetData) = Tokenizer22.createInputTargetPairs dataset
        use input = torch.tensor(inputData, device = device)
        use target = torch.tensor(targetData, device = device)

        // GRADIENT-BASED PARAMETER* OPTIMIZATION (PRE-TRAINING) 
        // * weights, biases, potentially learned positional encodings, layer normalization parameters
        printfn "Starting pre-training..."

        use model : torch.nn.Module<torch.Tensor, torch.Tensor> = (new Transformer(int64 vocabSize, dModel, nHeads, numLayers, device)).``to``(device)
        use lossFn = new CrossEntropyLoss(ignore_index = padTokenIdx)
        use optimizer = torch.optim.Adam(model.parameters(), lr = learningRate)
        
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
        //model.save("model2.pt") |> ignore<torch.nn.Module>
        //model.load("model2.pt") |> ignore<torch.nn.Module>

        // INFERENCE
        printfn "Generating sequence after fine-tuning..."
               
        let promptContent = prompt
        let promptTokens = Tokenizer22.tokenize promptContent |> List.toArray

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

        // DISPOSAL SECTION
        model.Dispose()
        torch.CurrentDisposeScope.DisposeEverything() // see torch.NewDisposeScope() above
        System.GC.Collect()