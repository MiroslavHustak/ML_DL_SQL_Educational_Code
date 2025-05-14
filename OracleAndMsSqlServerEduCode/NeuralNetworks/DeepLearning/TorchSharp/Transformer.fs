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
    let vocabSize = vocabulary.Length // 8
    let dModel = 64L // Embedding dimension (reduced)
    let epochs = 10000 // Number of pre-training epochs
    let fineTuneEpochs = 1000 // Number of fine-tuning epochs
    let batch = 32L // Batch size for pre-training
    let fineTuneBatch = 10L // Batch size for fine-tuning
    let headDim = 32L // Dimension per attention head (64 / 2)
    let nHeads = 2L // Number of attention heads (reduced)
    let numLayers = 2 // Number of transformer decoder layers
    let dropoutRate = 0.1f // Dropout rate for regularization
    let topK = 3L // Top-k sampling parameter

    // Positional encoding for token positions
    let getPositionalEncodings (seqLen: int64) (dModel: int64) : torch.Tensor =
        let pos = torch.arange(seqLen, dtype=torch.float32).unsqueeze(1) // [seqLen, 1]
        let divTerm = torch.exp(torch.arange(0L, dModel, 2L, dtype=torch.float32) * -(Math.Log(10000.0) / float dModel)) // [dModel/2]
        let pe = torch.zeros([|seqLen; dModel|]) // [seqLen, dModel]
        pe.index_copy_(1, torch.arange(0L, dModel, 2L), torch.sin(pos * divTerm)) |> ignore
        pe.index_copy_(1, torch.arange(1L, dModel, 2L), torch.cos(pos * divTerm)) |> ignore
        pe

    // Transformer Decoder Layer with causal mask and multi-head self-attention
    type TransformerDecoderLayer(dModel: int64, nHeads: int64, dropoutRate: float32) as self =
        inherit Module<torch.Tensor, torch.Tensor>("TransformerDecoderLayer")
        // Linear layers for projecting Query, Key, Value vectors
        let qkvProj = Linear(dModel, dModel * 3L)
        // Linear layer for attention output projection
        let outputProj = Linear(dModel, dModel)
        // Feed-forward network layers
        let ff1 = Linear(dModel, dModel * 4L)
        let ff2 = Linear(dModel * 4L, dModel)
        // Layer normalization for stabilizing training
        let norm1 = LayerNorm([|dModel|])
        let norm2 = LayerNorm([|dModel|])
        // Dropout for regularization
        let dropout = Dropout(float dropoutRate)

        do self.RegisterComponents()

        override _.forward(x) =
            let (batch, seq, _) = x.shape.[0], x.shape.[1], x.shape.[2]
            let newShape = [|batch; seq; 3L; nHeads; headDim|] // [batch, seq, 3, nHeads, headDim]

            use qkv = qkvProj.forward(x) // Project input to Query, Key, Value vectors
            use qkvReshaped = qkv.view(newShape) // Reshape to separate Q, K, V
            use q = qkvReshaped.select(2, 0L).transpose(1, 2) // [batch, nHeads, seq, headDim]
            use k = qkvReshaped.select(2, 1L).transpose(1, 2) // [batch, nHeads, seq, headDim]
            use v = qkvReshaped.select(2, 2L).transpose(1, 2) // [batch, nHeads, seq, headDim]
            use scores = torch.matmul(q, k.transpose(-2, -1)) / sqrt(float headDim) // Compute attention scores: [batch, nHeads, seq, seq]
            // Apply causal mask to hide future tokens
            use mask = torch.triu(torch.ones([|seq; seq|], device=scores.device), diagonal=1L).to_type(torch.ScalarType.Bool)
            use maskedScores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), System.Single.NegativeInfinity)
            // Normalize attention scores to weights using softmax
            use attentionWeights = softmax(maskedScores, -1L)
            // Apply dropout to attention weights
            use attentionWeights = dropout.forward(attentionWeights)
            // Compute context vectors: weighted sum of Value vectors
            use context = torch.matmul(attentionWeights, v) // [batch, nHeads, seq, headDim]
            // Reshape and project context vectors
            use context = context.transpose(1, 2).contiguous().view(batch, seq, dModel) // [batch, seq, dModel]
            context
            |> outputProj.forward // Linear projection of attention output
            |> fun output -> x + output // Residual connection
            |> norm1.forward // Layer normalization
            |> fun output ->
                output
                |> ff1.forward // First feed-forward layer
                |> gelu // GELU activation
                |> ff2.forward // Second feed-forward layer
                |> fun ffOutput -> output + ffOutput // Residual connection
                |> norm2.forward // Layer normalization

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
    let rec trainEpoch (model: torch.nn.Module<torch.Tensor, torch.Tensor>) (optimizer: torch.optim.Optimizer) (lossFn: CrossEntropyLoss) (input: torch.Tensor) (target: torch.Tensor) epoch maxEpochs phase =
        match epoch with
        | e when e >= maxEpochs -> ()
        | _ ->
            optimizer.zero_grad() // Reset gradients
            use output = model.forward(input) // Forward pass: compute logits [batch, seq, vocabSize]
            // Compute cross-entropy loss
            use loss = lossFn.forward(output.view(-1L, vocabSize), target.view(-1L))
            // Compute perplexity
            let perplexity = torch.exp(loss).item<float32>()
            // Backpropagation with error handling
            try
                loss.backward() // Backpropagation: compute loss gradients
            with
            | :? System.StackOverflowException as ex ->
                printfn "StackOverflowException in %s, epoch %d: %s" phase (epoch + 1) ex.Message
                // Trigger garbage collection to free memory
                System.GC.Collect()
                
            optimizer.step() |> ignore // Update model weights
            //printfn "%s Epoch %d, Loss: %.4f, Perplexity: %.4f" phase (epoch + 1) (loss.item<float32>()) perplexity
            // Trigger garbage collection to free memory
            System.GC.Collect()

            trainEpoch model optimizer lossFn input target (epoch + 1) maxEpochs phase

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
            // "the Sun is" -> "is yellow <eos>" (20 examples)
            | i, k when i < 20 -> [|0L; 1L; 2L|].[k]
            // "the sky is" -> "is blue <eos>" (10 examples)
            | i, k when i >= 20 && i < 30 -> [|0L; 5L; 2L|].[k]
            // "the Sun is" -> "is black <eos>" (1 example)
            | i, k when i = 30 -> [|0L; 1L; 2L|].[k]
            // "the sky is" -> "is black <eos>" (1 example)
            | i, k when i = 31 -> [|0L; 5L; 2L|].[k]
            | _ -> failwith "Invalid index")

        let targetData = Array2D.init 32 3 (fun i k ->
            match (i, k) with
            // "the Sun is" -> "is yellow <eos>" (20 examples)
            | i, k when i < 20 -> [|2L; 3L; 7L|].[k]
            // "the sky is" -> "is blue <eos>" (10 examples)
            | i, k when i >= 20 && i < 30 -> [|2L; 6L; 7L|].[k]
            // "the Sun is" -> "is black <eos>" (1 example)
            | i, k when i = 30 -> [|2L; 4L; 7L|].[k]
            // "the sky is" -> "is black <eos>" (1 example)
            | i, k when i = 31 -> [|2L; 4L; 7L|].[k]
            | _ -> failwith "Invalid index")

        use input = torch.tensor(inputData, device=device) // [32, 3]
        use target = torch.tensor(targetData, device=device) // [32, 3]

        // Pre-training: train the model
        printfn "Starting pre-training..."
        model.train()
        trainEpoch model optimizer lossFn input target 0 epochs "Pre-training"

        // Save the pre-trained model
        //torch.save(model.state_dict(), "transformer_model.pt")

        // Fine-tuning: prepare a small dataset to emphasize "the sky is" -> "is blue <eos>"
        let fineTuneInputData = Array2D.init 10 3 (fun i k ->
            [|0L; 5L; 2L|].[k]) // "the sky is" for all 10 examples
        let fineTuneTargetData = Array2D.init 10 3 (fun i k ->
            [|2L; 6L; 7L|].[k]) // "is blue <eos>" for all 10 examples
        use fineTuneInput = torch.tensor(fineTuneInputData, device=device) // [10, 3]
        use fineTuneTarget = torch.tensor(fineTuneTargetData, device=device) // [10, 3]

        // Fine-tuning: adjust model with a lower learning rate
        printfn "Starting fine-tuning..."
        use fineTuneOptimizer = torch.optim.Adam(model.parameters(), lr=0.0001) // Lower learning rate for fine-tuning
        model.train()
        trainEpoch model fineTuneOptimizer lossFn fineTuneInput fineTuneTarget 0 fineTuneEpochs "Fine-tuning"

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