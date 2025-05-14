namespace NeuralNetworks

open System
open TorchSharp
open TorchSharp.Modules
open type torch.nn
open type torch.nn.functional

// Educational code for understanding LLMs with TorchSharp

module Transformer_TorchSharp =

    // Vocabulary and hyperparameters
    let vocabulary = [|"The"; "Sun"; "is"; "yellow"; "black"; "sky"; "blue"; "<eos>"|]
    let vocabSize = vocabulary.Length
    let dModel = 64L
    let epochs = 20000
    let fineTuneEpochs = 2000
    let batch = 32L
    let fineTuneBatch = 10L
    let headDim = 32L
    let nHeads = 2L
    let numLayers = 2
    let dropoutRate = 0.1f
    let topK = 3L

    /// Generates positional encodings for token positions
    let getPositionalEncodings (seqLen: int64) (dModel: int64) : torch.Tensor =
        let position = torch.arange(seqLen, dtype=torch.float32).unsqueeze(1)
        let divTerm = torch.exp(torch.arange(0L, dModel, 2L, dtype=torch.float32) * -(Math.Log(10000.0) / float dModel))
        
        let encodings = torch.zeros([|seqLen; dModel|])
        encodings.index_copy_(1, torch.arange(0L, dModel, 2L), torch.sin(position * divTerm)) |> ignore
        encodings.index_copy_(1, torch.arange(1L, dModel, 2L), torch.cos(position * divTerm)) |> ignore
        encodings

    /// Transformer Decoder Layer with causal mask and multi-head self-attention
    type TransformerDecoderLayer(dModel: int64, nHeads: int64, dropoutRate: float32) as self =
        inherit Module<torch.Tensor, torch.Tensor>("TransformerDecoderLayer")
        
        let qkvProjection = Linear(dModel, dModel * 3L)
        let outputProjection = Linear(dModel, dModel)
        let feedForward1 = Linear(dModel, dModel * 4L)
        let feedForward2 = Linear(dModel * 4L, dModel) 
        let layerNorm1 = LayerNorm([|dModel|])
        let layerNorm2 = LayerNorm([|dModel|])
        let dropout = Dropout(float dropoutRate)

        do self.RegisterComponents()

        override _.forward(x) =
            let (batch, seq, _) = x.shape.[0], x.shape.[1], x.shape.[2]
            let reshapedShape = [|batch; seq; 3L; nHeads; headDim|]

            // Project and reshape input for multi-head attention
            use qkv = qkvProjection.forward(x)
            use qkvReshaped = qkv.view(reshapedShape)
            use q = qkvReshaped.select(2, 0L).transpose(1, 2)
            use k = qkvReshaped.select(2, 1L).transpose(1, 2)
            use v = qkvReshaped.select(2, 2L).transpose(1, 2)

            // Scaled Dot-Product Attention with causal masking
            use scores = torch.matmul(q, k.transpose(-2, -1)) / sqrt(float headDim)
            use mask = torch.triu(torch.ones([|seq; seq|], device=scores.device), diagonal=1L).to_type(torch.ScalarType.Bool)
            use maskedScores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), System.Single.NegativeInfinity)

            // Apply softmax and dropout
            use attentionWeights = softmax(maskedScores, -1L) |> dropout.forward
            use context = torch.matmul(attentionWeights, v)
            use context = context.transpose(1, 2).contiguous().view(batch, seq, dModel)

            // Apply final linear transformation and residual connection
            context
            |> outputProjection.forward
            |> fun output -> x + output
            |> layerNorm1.forward
            |> fun output ->
                output
                |> feedForward1.forward
                |> gelu
                |> feedForward2.forward
                |> fun ffOutput -> output + ffOutput
                |> layerNorm2.forward

    // MiniTransformer model with multiple decoder layers
    type Transformer(vocabSize: int64, dModel: int64, nHeads: int64, numLayers: int) as self =
        inherit Module<torch.Tensor, torch.Tensor>("Transformer")
        // Token embedding layer
        let embedding = Embedding(vocabSize, dModel)
        // Positional encoding (precomputed for max sequence length)
        let posEnc = getPositionalEncodings 10L dModel // Max sequence length 10
        // Dropout for regularization
        let dropout = Dropout(float dropoutRate)
        // Stack of transformer decoder layers
        let decoderLayers = ModuleList<torch.nn.Module<torch.Tensor, torch.Tensor>>([| for _ in 1..numLayers -> new TransformerDecoderLayer(dModel, nHeads, dropoutRate) :> torch.nn.Module<torch.Tensor, torch.Tensor> |])
        // Final linear output layer
        let outputLayer = Linear(dModel, vocabSize)
        // Final layer normalization
        let norm = LayerNorm([|dModel|])

        do self.RegisterComponents()

        override _.forward(x) =
            let emb = embedding.forward(x) // Convert token indices to embeddings: [batch, seq, dModel]
            let seqLen = x.shape.[1]
            // Add positional encodings (slice to match sequence length and move to input device)
            let embWithPos = emb + posEnc.narrow(0L, 0L, seqLen).``to``(x.device)
            // Apply dropout to embeddings
            let embWithPos = dropout.forward(embWithPos)
            // Process through multiple transformer decoder layers
            let dec = Seq.fold (fun x (layer: torch.nn.Module<torch.Tensor, torch.Tensor>) -> layer.forward(x)) embWithPos decoderLayers // [batch, seq, dModel]
            // Apply final layer normalization
            let normOut = norm.forward(dec)
            // Project to vocabulary size for logits
            outputLayer.forward(normOut).to_type(torch.ScalarType.Float32) // [batch, seq, vocabSize]

    // Training loop for pre-training or fine-tuning with perplexity evaluation
    let trainEpoch (model: torch.nn.Module<torch.Tensor, torch.Tensor>) (optimizer: torch.optim.Optimizer) (lossFn: CrossEntropyLoss) (input: torch.Tensor) (target: torch.Tensor) maxEpochs phase =
        [0..maxEpochs-1] |> List.iter (fun epoch ->
            optimizer.zero_grad() // Reset gradients
            use output = model.forward(input) // Forward pass: compute logits [batch, seq, vocabSize]
            // Compute cross-entropy loss
            use loss = lossFn.forward(output.view(-1L, vocabSize), target.view(-1L))
            // Compute perplexity
            let perplexity = torch.exp(loss).item<float32>()
            // Backpropagation with error handling
            try
                loss.backward() // Backpropagation: compute loss gradients
                // Clip gradients to stabilize backpropagation
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) |> ignore
            with
            | :? System.StackOverflowException as ex ->
                printfn "StackOverflowException in %s, epoch %d: %s" phase (epoch + 1) ex.Message
                Console.ReadLine() |> ignore
            optimizer.step() |> ignore // Update model weights
            //printfn "%s Epoch %d, Loss: %.4f, Perplexity: %.4f" phase (epoch + 1) (loss.item<float32>()) perplexity
            // Trigger garbage collection to free memory
            System.GC.Collect()
        )

    // Inference loop with top-k sampling
    let rec generate (model: torch.nn.Module<torch.Tensor, torch.Tensor>) (inputSeq: torch.Tensor) steps maxSteps acc =
        match steps with
        | s when s >= maxSteps -> acc
        | _ ->
            let _ = torch.no_grad() // Disable gradient computation for inference
            let logits: torch.Tensor = model.forward(inputSeq) // Compute logits: [1, seq, vocabSize]
            let lastLogit: torch.Tensor = logits.select(0, 0L).select(0, -1L) // Get logits for last token: [vocabSize]
            // Mask <eos> in first step to avoid premature termination
            let adjustedLogit: torch.Tensor = if steps = 0 then lastLogit.index_fill_(0, torch.tensor([|7L|], device=lastLogit.device), System.Single.NegativeInfinity) else lastLogit
            // Apply temperature scaling and top-k sampling
            let temp: float32 = 0.7f // Adjusted for balanced sampling
            let struct (probs: torch.Tensor, indices: torch.Tensor) = torch.topk(adjustedLogit / temp, int topK, dim=0) // Select top-k logits: [topK], [topK]
            let probs: torch.Tensor = softmax(probs, dim=0L) // Normalize to probabilities: [topK]
            let nextTokenIdx: int64 = torch.multinomial(probs, 1).item<int64>() // Sample from top-k: scalar
            let nextToken: int64 = indices.[nextTokenIdx].item<int64>() // Map back to original token index: scalar
            let newAcc: int64 list = nextToken :: acc // Accumulate token indices
            // Append new token to input sequence
            let newInput: torch.Tensor = torch.cat([|inputSeq; torch.tensor([|nextToken|], device=inputSeq.device).unsqueeze(0L)|], dim=1L) // [1, seq+1]
            generate model newInput (steps + 1) maxSteps newAcc

    // Main program
    let main () =
        // Select device (CPU or CUDA)
        let device = if torch.cuda.is_available() then torch.CUDA else torch.CPU
        printfn "Using device: %A" device

        // Initialize model, loss function, and optimizer
        use model = (new Transformer(int64 vocabSize, dModel, nHeads, numLayers)).``to``(device) // Create transformer and move to device
        use lossFn = new CrossEntropyLoss() // Cross-entropy loss for training
        use optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) // Adam optimizer for pre-training (lower LR)

        // Data preparation: pre-training input and target sequences (32 examples)
        let inputData = Array2D.init 32 3 (fun i k ->
            match (i, k) with
            // "The Sun is" -> "is yellow <eos>" (2600 examples instead of 20)
            | i, k when i < 2600 -> [|0L; 1L; 2L|].[k]
            // "The sky is" -> "is blue <eos>" (5 examples instead of 10)
            | i, k when i >= 26 && i < 31 -> [|0L; 5L; 2L|].[k]
            // "The Sun is" -> "is black <eos>" (1 example, kept the same)
            | i, k when i = 31 -> [|0L; 1L; 2L|].[k]
            | _ -> failwith "Invalid index")
        
        let targetData = Array2D.init 32 3 (fun i k ->
            match (i, k) with
            // "The Sun is" -> "is yellow <eos>" (26 examples instead of 20)
            | i, k when i < 26 -> [|2L; 3L; 7L|].[k]
            // "The sky is" -> "is blue <eos>" (5 examples instead of 10)
            | i, k when i >= 26 && i < 31 -> [|2L; 6L; 7L|].[k]
            // "The Sun is" -> "is black <eos>" (1 example, kept the same)
            | i, k when i = 31 -> [|2L; 4L; 7L|].[k]
            | _ -> failwith "Invalid index")        

        use input = torch.tensor(inputData, device=device) // [32, 3]
        use target = torch.tensor(targetData, device=device) // [32, 3]

        // Pre-training: train the model
        printfn "Starting pre-training..."
        model.train()
        trainEpoch model optimizer lossFn input target epochs "Pre-training"

        // Save the pre-trained model
        //torch.save(model.state_dict(), "transformer_model.pt")

        // Fine-tuning: prepare a small dataset to emphasize "the Sun is" -> "is yellow <eos>"
        let fineTuneInputData = Array2D.init 10 3 (fun i k ->
            [|0L; 1L; 2L|].[k]) // "the Sun is" for all 10 examples
        let fineTuneTargetData = Array2D.init 10 3 (fun i k ->
            [|2L; 3L; 7L|].[k]) // "is yellow <eos>" for all 10 examples
        use fineTuneInput = torch.tensor(fineTuneInputData, device=device) // [10, 3]
        use fineTuneTarget = torch.tensor(fineTuneTargetData, device=device) // [10, 3]

        // Fine-tuning: adjust model with a lower learning rate
        printfn "Starting fine-tuning..."
        use fineTuneOptimizer = torch.optim.Adam(model.parameters(), lr=0.0001) // Lower learning rate for fine-tuning
        model.train()
        trainEpoch model fineTuneOptimizer lossFn fineTuneInput fineTuneTarget fineTuneEpochs "Fine-tuning"

        // Save the fine-tuned model
        //torch.save(model.state_dict(), "transformer_model_finetuned.pt")

        // Inference: Generate sequence starting with "The Sun is" ([0, 1, 2])
        printfn "Generating sequence after fine-tuning..."
        model.eval()
        use inputSeq = torch.tensor([|0L; 1L; 2L|], device=device).unsqueeze(0L) // [1, 3]
        printf "Generated sequence (token IDs): "
        let generated = generate model inputSeq 0 2 [] |> List.rev // Generate 2 tokens
        generated |> List.iter (printf "%d ")
        printfn ""
        // Map token IDs to words
        printf "Generated sequence (words): "
        generated |> List.iter (fun id -> printf "%s " vocabulary.[int id])
        printfn ""