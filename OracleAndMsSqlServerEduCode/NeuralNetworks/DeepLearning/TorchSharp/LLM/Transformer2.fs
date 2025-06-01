namespace NeuralNetworks

open System

open TorchSharp
open TorchSharp.Modules

open type torch.nn
open type torch.nn.functional

open Settings2

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

module Transformer_TorchSharp2 =  //variant 2

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
                match dModel % nHeads with
                | 0L
                    ->
                    let headDim = dModel / nHeads
                    let reshapedShape = [|batch; seq; 3L; nHeads; headDim|]
                    
                    use qkv = qkvProjection.forward x
                    use qkvReshaped = qkv.view(reshapedShape)
                    
                    use q = qkvReshaped.select(2, 0L).transpose(1, 2)
                    use k = qkvReshaped.select(2, 1L).transpose(1, 2)
                    use v = qkvReshaped.select(2, 2L).transpose(1, 2)
                    
                    use attentionScores = torch.matmul(q, k.transpose(-2, -1)) / sqrt(float headDim)
                    use mask = torch.triu(torch.ones([|seq; seq|], device = attentionScores.device), diagonal = 1L).to_type(torch.ScalarType.Bool)
                    use maskedScores = attentionScores.masked_fill(mask.unsqueeze(0).unsqueeze(0), System.Single.NegativeInfinity)
                    use attentionWeights = softmax(maskedScores, -1L) |> dropout.forward
                    
                    let contextVector = torch.matmul(attentionWeights, v)
                    use contextVector =
                        contextVector
                        |> (fun cv -> cv.transpose(1, 2))
                        |> (fun cv -> cv.contiguous())
                        |> (fun cv -> cv.view(batch, seq, dModel))
                    
                    let out1 =
                        contextVector
                        |> outputProjection.forward
                        |> (fun output -> x + output)
                    
                    let out2 =
                        out1
                        |> layerNorm1.forward
                        |> feedForward1.forward
                        |> gelu
                        |> feedForward2.forward
                        |> dropout.forward
                    
                    out1 + out2
                    |> layerNorm2.forward

                | _ ->
                    failwithf "dModel (%d) must be divisible by nHeads (%d)" dModel nHeads

            | shapeArr
                ->
                failwithf "Input tensor must have shape [batch; seq; dModel] but got %A" shapeArr

    let private getPositionalEncodings (seqLen: int64) (dModel: int64) : torch.Tensor =

        let position = torch.arange(seqLen, dtype = torch.float32).unsqueeze(1)
        let divTerm = torch.exp(torch.arange(0L, dModel, 2L, dtype = torch.float32) * (-Math.Log 10000.0 / float dModel))
        
        let encodings = torch.zeros([|seqLen; dModel|])
        encodings.index_copy_(1, torch.arange(0L, dModel, 2L), torch.sin(position * divTerm)) |> ignore
        encodings.index_copy_(1, torch.arange(1L, dModel, 2L), torch.cos(position * divTerm)) |> ignore
        
        encodings

    type private Transformer(vocabSize: int64, dModel: int64, nHeads: int64, numLayers: int) as self =

        inherit Module<torch.Tensor, torch.Tensor>("Transformer")

        let embedding = Embedding(vocabSize, dModel)
        let posEnc = getPositionalEncodings 128L dModel
        let dropout = Dropout(float dropoutRate)

        let decoderLayers =
            List.init numLayers (fun _ -> new TransformerDecoderLayer(dModel, nHeads, dropoutRate) :> torch.nn.Module<torch.Tensor, torch.Tensor>)
            |> List.toArray
            |> ModuleList<torch.nn.Module<torch.Tensor, torch.Tensor>>
        
        let outputLayer = Linear(dModel, vocabSize)
        let norm = LayerNorm [|dModel|]

        do outputLayer.weight <- embedding.weight   
        do self.RegisterComponents()

        override _.forward x =

            let emb = embedding.forward x
            let seqLen = x.shape |> Array.item 1
            let embWithPos = emb + posEnc.narrow(0L, 0L, seqLen).``to``(x.device)
            let embWithPos = dropout.forward embWithPos
            
            let decoderOut = Seq.fold (fun acc (layer: torch.nn.Module<torch.Tensor, torch.Tensor>) -> layer.forward acc) embWithPos decoderLayers
            let normOut = norm.forward decoderOut
            
            outputLayer.forward(normOut).to_type(torch.ScalarType.Float32)

    let private trainEpoch (model: torch.nn.Module<torch.Tensor, torch.Tensor>) (optimizer: torch.optim.Optimizer)
                           (lossFn: CrossEntropyLoss) (input: torch.Tensor) (target: torch.Tensor) maxEpochs phase =

        [0 .. maxEpochs - 1]
        |> List.iter
            (fun epoch 
                ->
                let counter = epoch + 1

                optimizer.zero_grad()

                let output = model.forward input
                let loss = lossFn.forward(output.view(-1L, int64 vocabSize), target.view(-1L))

                let perplexity = torch.exp(loss).item<float32>() 

                try
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0) |> ignore<float>
                with
                | :? System.StackOverflowException 
                    as ex 
                        ->
                        printfn "StackOverflowException in %s, epoch %d: %s" phase counter (string ex.Message)
                | ex
                        ->
                        printfn "%s" (string ex.Message)

                optimizer.step() |> ignore                
        
                match counter % 20 with
                | 0 -> printfn "%s Epoch %d, Loss: %.4f, Perplexity: %.4f" phase counter (loss.item<float32>()) perplexity
                | _ -> ()

                loss.Dispose()
                output.Dispose()
        )                    

    let rec generateCPS (model: torch.nn.Module<torch.Tensor, torch.Tensor>) (inputSeq: torch.Tensor) (steps: int) 
                        (maxSteps: int) (acc: int64 list) (contextSize: int64) (temp: float32) (topK: int64) (strategy: string)
                        (cont: int64 list -> 'a) : 'a =

        let trimInput (input: torch.Tensor) =

            match input.shape.[1] > contextSize with
            | true  -> input.narrow(1, int64 (int input.shape.[1] - int contextSize), contextSize)
            | false -> input

        let sampleLogits (logits: torch.Tensor) =

            match strategy with
            | "top-k" 
                ->
                let struct (probs, indices) = torch.topk(logits / temp, int (min topK (int64 vocabSize)), dim = 0)
                let probs = softmax(probs, dim = 0L)
                let idx = torch.multinomial(probs, 1).item<int64>()
                indices.[idx].item<int64>()

            | "greedy"
                ->
                torch.argmax(logits, dim = 0).item<int64>()

            | s 
                ->
                failwithf "Unsupported sampling strategy: %s" s

        match steps >= maxSteps with
        | true  -> 
                List.rev >> cont <| acc
        | false ->
                use _ = torch.no_grad()

                let trimmedInput = trimInput inputSeq
                let logits: torch.Tensor = model.forward trimmedInput
                let lastLogit: torch.Tensor = logits.select(0, 0L).select(0, -1L)
        
                let nextToken = sampleLogits lastLogit

                match nextToken = eosTokenIdx || nextToken = padTokenIdx with
                | true  ->
                        List.rev >> cont <| nextToken :: acc
                | false ->
                        let newInput = torch.cat([|inputSeq; torch.tensor([|nextToken|], device = inputSeq.device).unsqueeze(0L)|], dim = 1L)
                        generateCPS model newInput (steps + 1) maxSteps (nextToken :: acc) contextSize temp topK strategy cont

    let internal main () =

        use scope = torch.NewDisposeScope()

        let device = 
            match torch.cuda.is_available() with
            | true  -> torch.CUDA
            | false -> torch.CPU
        
        printfn "Using device: %A" <| (string device).ToUpper()

        let dataset = TextData2.getSequences()
        let (inputData, targetData) = Tokenizer2.createInputTargetPairs dataset
        use input = torch.tensor(inputData, device = device)
        use target = torch.tensor(targetData, device = device)

        printfn "Starting pre-training..."

        use model : torch.nn.Module<torch.Tensor, torch.Tensor> = (new Transformer(int64 vocabSize, dModel, nHeads, numLayers)).``to``(device)
        use lossFn = new CrossEntropyLoss(ignore_index = padTokenIdx)
        use optimizer = torch.optim.Adam(model.parameters(), lr = learningRate)
        
        model.train()
        
        trainEpoch model optimizer lossFn input target epochs "Pre-training"

        printfn "Starting fine-tuning..."
        
        let (fineTuneInputData, fineTuneTargetData) = TextData2.getFineTuningCausalLMSequences ()
        use fineTuneInput = torch.tensor(fineTuneInputData, device = device)
        use fineTuneTarget = torch.tensor(fineTuneTargetData, device = device)
        use fineTuneOptimizer = torch.optim.Adam(model.parameters(), lr = learningRate)
        
        model.train()

        trainEpoch model fineTuneOptimizer lossFn fineTuneInput fineTuneTarget fineTuneEpochs "Fine-tuning"

        printfn "Generating sequence after fine-tuning..."
        
        model.``to``("cpu") |> ignore<torch.nn.Module<torch.Tensor, torch.Tensor>>
        model.eval()
        
        let question = "What is the colour of the sky? <sep>"
        let questionTokens = Tokenizer2.tokenize question |> Array.ofList
        use inputSeq = torch.tensor(questionTokens, device = torch.CPU).unsqueeze 0L
        
        let generated = generateCPS model inputSeq 0 64 [] contextSize temp topK strategy id
        
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
                | false -> printf "<UNK> "
            )

        printfn "\n"

        model.Dispose()

        torch.CurrentDisposeScope.DisposeEverything()

        System.GC.Collect()