namespace NeuralNetworks

open System

open TorchSharp
open TorchSharp.Modules

open type torch.nn
open type torch.nn.functional

open LoRA 
open Settings

module Transformer_TorchSharp4 =

    // Defines a custom RMSNorm module for layer normalization, used to stabilize training by normalizing activations.
    type private RMSNorm(normalizedShape: int64[], eps: float32) as self =
        inherit Module<torch.Tensor, torch.Tensor>("RMSNorm") // Inherits from TorchSharp's Module class, defining a custom layer.

        let weight = torch.nn.Parameter(torch.ones normalizedShape) // Creates a learnable weight parameter initialized to ones, matching the normalized shape.
        let eps = torch.tensor eps // Creates a small constant tensor (eps) to prevent division by zero during normalization.
        do self.RegisterComponents() // Registers the module's components (e.g., weight) with TorchSharp for parameter tracking.

        override _.forward (x: torch.Tensor) = // Defines the forward pass for RMSNorm.
            use norm = x.pow(torch.tensor(2.0f)).mean([|-1L|], keepdim = true).add(eps).sqrt() // Computes the root mean square (RMS) of the input: squares elements, takes mean over the last dimension, adds eps, and takes square root.
            x / norm * weight // Normalizes the input by dividing by RMS and scaling by the learnable weight.

    // Defines a transformer decoder layer, a building block of the transformer model, including attention and feed-forward components.
    type private TransformerDecoderLayer(dModel: int64, nHeads: int64, dropoutRate: float32, device: torch.Device, useLora: bool) as self =
        inherit Module<torch.Tensor, torch.Tensor>("TransformerDecoderLayer") // Inherits from Module class, defining a transformer decoder layer.

        let headDim = dModel / nHeads // Calculates the dimension of each attention head by dividing the model dimension by the number of heads.
        let kvHeads = nHeads / 4L |> max 1L // Sets the number of key/value heads (1/4 of query heads, minimum 1) for efficiency in multi-query attention.

        // Defines a helper function to create linear layers, optionally using LoRA (Low-Rank Adaptation) for efficient fine-tuning.
        let mkLinear (inF, outF) =
            match useLora with
            | true  -> new LoRALinear(inF, outF, rank = 4L, alpha = 32.0f, device = device) :> torch.nn.Module<torch.Tensor, torch.Tensor> // Creates a LoRA linear layer with rank 4 and scaling factor alpha.
            | false -> Linear(inF, outF) :> torch.nn.Module<torch.Tensor, torch.Tensor> // Creates a standard linear layer if LoRA is not used.

        let qProjection = mkLinear(dModel, dModel) // Creates the query projection layer, mapping input to query vectors for all heads.
        let kProjection = mkLinear(dModel, kvHeads * headDim) // Creates the key projection layer, mapping input to key vectors for key/value heads.
        let vProjection = mkLinear(dModel, kvHeads * headDim) // Creates the value projection layer, mapping input to value vectors for key/value heads.
        let outputProjection = mkLinear(dModel, dModel) // Creates the output projection layer, mapping attention output back to model dimension.
        let feedForward1 = Linear(dModel, dModel * 4L) // Creates the first linear layer of the feed-forward network, expanding to 4x model dimension.
        let feedForward2 = Linear(dModel * 4L, dModel) // Creates the second linear layer of the feed-forward network, projecting back to model dimension.
        let layerNorm1 = new RMSNorm([|dModel|], 1e-5f) // Creates the first RMSNorm layer, applied before attention.
        let layerNorm2 = new RMSNorm([|dModel|], 1e-5f) // Creates the second RMSNorm layer, applied before the feed-forward network.
        let dropout = Dropout(float dropoutRate) // Creates a dropout layer to prevent overfitting by randomly zeroing activations.

        do self.RegisterComponents() // Registers all components (projections, norms, dropout) with TorchSharp for parameter tracking.

        override _.forward x = // Defines the forward pass for the transformer decoder layer.
            // Defines a helper function to apply rotary positional embeddings (RoPE) to queries and keys for position-aware attention.
            let applyRotary (q: torch.Tensor) (k: torch.Tensor) : torch.Tensor * torch.Tensor =
                let lastDim = q.shape.[q.shape.Length - 1] // Gets the last dimension of the query tensor (head dimension).
                match lastDim % 2L <> 0L with true -> failwithf "The last dimension (%d) is not even, cannot apply rotary split." lastDim | false -> () // Ensures the head dimension is even for rotary split.
                let dim = lastDim / 2L // Splits the head dimension into two for rotary embeddings.
                use theta = torch.pow(10000.0f, torch.arange(0L, dim, device=q.device, dtype=torch.float32) / float32 dim) // Computes frequency scaling factors for RoPE.
                let seqLen = q.shape.[q.shape.Length - 2] // Gets the sequence length from the query tensor.
                use positionIds = torch.arange(seqLen, device=q.device, dtype=torch.float32).unsqueeze(-1) // Creates position indices for the sequence.
                use freqs = positionIds / theta // Computes position-dependent frequencies for RoPE.
                use sin = torch.sin freqs // Computes sine of frequencies for rotation.
                use cos = torch.cos freqs // Computes cosine of frequencies for rotation.

                // Defines a helper function to apply rotary embeddings by rotating pairs of dimensions.
                let reshapeForRotation (x: torch.Tensor) =
                    let split = x.split([|dim; dim|], -1L) // Splits the last dimension into two equal parts.
                    match split.Length <> 2 with
                    | true  -> failwithf "Split did not return two tensors. Split length: %d, dim: %A, x.shape: %A" split.Length dim x.shape | false -> () // Ensures the split produces exactly two tensors.
                    use a = split |> Array.head // Gets the first half of the split.
                    use b = split |> Array.last // Gets the second half of the split.
                    torch.cat([|(a * cos) - (b * sin); (a * sin) + (b * cos)|], dim = -1) // Applies rotary transformation: rotates pairs of dimensions using sine and cosine.

                reshapeForRotation q, reshapeForRotation k // Returns rotated query and key tensors.

            // Defines a helper function to compute ALiBi (Attention with Linear Biases) for positional bias in attention.
            let getAlibiBias (nHeads: int64) (seq: int64) (device: torch.Device) =
                use slopes = torch.linspace(1.0, 0.0, int nHeads, dtype = torch.float32, device = device).unsqueeze(-1).unsqueeze(-1) // Creates linearly decreasing slopes for each head.
                use bias = torch.arange(seq, device = device).unsqueeze(0).unsqueeze(0).float() // Creates position indices for the sequence.
                (*) slopes bias // Computes ALiBi bias by multiplying slopes with position indices.

            match x.shape with
            | [|batch; seq; dmodel|] when dmodel = dModel -> // Ensures the input shape is [batch, sequence_length, dModel].
                use normedInput = layerNorm1.forward x // Applies RMSNorm to the input before attention.

                // Computes query, key, and value projections, with optional CUDA optimization.
                let result =
                    match torch.cuda.is_available() with
                    | true -> // CUDA path for GPU acceleration.
                        let q = qProjection.forward(normedInput.view([|batch * seq; dModel|])).view([|batch; seq; nHeads; headDim|]).transpose(1, 2) // Projects input to queries, reshapes to [batch, nHeads, seq, headDim], and transposes for attention.
                        let k = kProjection.forward(normedInput.view([|batch * seq; dModel|])).view([|batch; seq; kvHeads; headDim|]).transpose(1, 2) // Projects input to keys, reshapes, and transposes.
                        let v = vProjection.forward(normedInput.view([|batch * seq; dModel|])).view([|batch; seq; kvHeads; headDim|]).transpose(1, 2) // Projects input to values, reshapes, and transposes.
                        let q = q.cuda() // Moves queries to GPU.
                        let k = k.cuda() // Moves keys to GPU.
                        let v = v.cuda() // Moves values to GPU.
                        let qkv = torch.cat([| q; k; v |], dim=2L) // Concatenates queries, keys, and values along the head dimension.
                        [
                            qkv.slice(1L, 0L, int64 nHeads, headDim) // Extracts queries: [batch, nHeads, seq, headDim].
                            qkv.slice(1L, int64 nHeads, int64 (nHeads + kvHeads), headDim) // Extracts keys: [batch, kvHeads, seq, headDim].
                            qkv.slice(1L, int64 (nHeads + kvHeads), int64 (nHeads + 2L * kvHeads), headDim) // Extracts values: [batch, kvHeads, seq, headDim].
                        ]
                    | false -> // CPU path using parallel execution.
                        [
                            (fun () -> qProjection.forward(normedInput.view([|batch * seq; dModel|])).view([|batch; seq; nHeads; headDim|]).transpose(1, 2)) // Projects and reshapes queries.
                            (fun () -> kProjection.forward(normedInput.view([|batch * seq; dModel|])).view([|batch; seq; kvHeads; headDim|]).transpose(1, 2)) // Projects and reshapes keys.
                            (fun () -> vProjection.forward(normedInput.view([|batch * seq; dModel|])).view([|batch; seq; kvHeads; headDim|]).transpose(1, 2)) // Projects and reshapes values.
                        ]
                        |> List.Parallel.map (fun f -> f()) // Executes projections in parallel on CPU.

                use q = result |> List.head // Extracts query tensor.
                use k = result |> List.item 1 // Extracts key tensor.
                use v = result |> List.last // Extracts value tensor.

                let qRoPE, kRoPE = applyRotary q k // Applies rotary positional embeddings to queries and keys.

                use qRoPE = qRoPE // Uses rotated queries.
                use kRoPE = kRoPE // Uses rotated keys.

                use attentionScores : torch.Tensor = torch.matmul(qRoPE, kRoPE.transpose(-2, -1)) / sqrt(float headDim) // Computes attention scores by matrix-multiplying queries with transposed keys and scaling.
                use alibiBias = getAlibiBias nHeads seq attentionScores.device // Computes ALiBi positional bias.
                use attentionScoresWithBias : torch.Tensor = (+) attentionScores alibiBias // Adds ALiBi bias to attention scores.
                use mask : torch.Tensor = torch.triu(torch.ones([|seq; seq|], device = attentionScores.device), diagonal = 1L).to_type(torch.ScalarType.Bool) // Creates a causal mask to prevent attending to future tokens.
                use maskedScores = attentionScoresWithBias.masked_fill(mask.unsqueeze(0).unsqueeze(0), System.Single.NegativeInfinity) // Applies causal mask by setting future token scores to negative infinity.
                use attentionWeights = softmax(maskedScores, -1L) |> dropout.forward // Applies softmax to get attention weights and applies dropout.

                let result =
                    [
                        (fun () -> torch.matmul(attentionWeights, v).transpose(1, 2).contiguous().view(batch, seq, dModel) |> outputProjection.forward) // Computes attention output, reshapes, and applies output projection.
                        (fun () -> normedInput |> layerNorm2.forward |> feedForward1.forward |> gelu |> feedForward2.forward |> dropout.forward) // Applies RMSNorm, feed-forward network (with GELU), and dropout.
                    ]
                    |> List.Parallel.map (fun f -> f()) // Executes attention and feed-forward computations in parallel.

                use attnResult = result |> List.head // Extracts attention output.
                use ffnResult = result |> List.last // Extracts feed-forward output.

                use residualScale = 1.0f / float32 (2 * numLayers) |> torch.tensor // Computes a residual scaling factor based on the number of layers.
                use out1 = x + attnResult * residualScale // Applies residual connection after attention, scaling to stabilize training.
                out1 + ffnResult * residualScale // Applies residual connection after feed-forward, scaling to stabilize training.

            | shapeArr -> failwithf "Invalid input shape: %A" shapeArr // Throws an error for invalid input shapes.

    // Defines the full transformer model, combining embedding, decoder layers, and output layers.
    type private Transformer(vocabSize: int64, dModel: int64, nHeads: int64, numLayers: int, device: torch.Device, useLora: bool) as self =
        inherit Module<torch.Tensor, torch.Tensor>("Transformer") // Inherits from Module class, defining the transformer.

        let embedding = Embedding(vocabSize, dModel) // Creates an embedding layer to map token IDs to dModel-dimensional vectors.
        let dropout = Dropout(float dropoutRate) // Creates a dropout layer for regularization.
        let decoderLayers =
            List.init numLayers (fun _ -> new TransformerDecoderLayer(dModel, nHeads, dropoutRate, device, useLora) :> torch.nn.Module<torch.Tensor, torch.Tensor>) // Initializes a list of transformer decoder layers.
            |> List.toArray
            |> ModuleList<torch.nn.Module<torch.Tensor, torch.Tensor>> // Wraps decoder layers in a ModuleList for TorchSharp.
        let outputLayer = Linear(dModel, vocabSize) // Creates the output linear layer to map to vocabulary size.
        let norm = new RMSNorm([|dModel|], 1e-5f) // Creates an RMSNorm layer for final normalization.

        do outputLayer.weight <- embedding.weight // Ties the output layer weights to the embedding weights for efficiency.
        do self.RegisterComponents() // Registers all components with TorchSharp.

        override _.forward x = // Defines the forward pass for the transformer.
            use emb = embedding.forward x // Embeds input token IDs into dModel-dimensional vectors.
            use embWithDropout = dropout.forward emb // Applies dropout to embeddings.
            use decoderOut = decoderLayers |> Seq.fold (fun acc (layer: torch.nn.Module<torch.Tensor, torch.Tensor>) -> layer.forward acc) embWithDropout // Passes embeddings through all decoder layers.
            use normOut = norm.forward decoderOut // Applies final RMSNorm.
            outputLayer.forward(normOut).to_type(torch.ScalarType.Float32) // Projects to vocabulary size and converts to float32 for output logits.

    // Defines the training function for one epoch, optimizing the model parameters.
    let private trainEpoch (model: torch.nn.Module<torch.Tensor, torch.Tensor>) (optimizer: torch.optim.Optimizer)
                           (lossFn: CrossEntropyLoss) (input: torch.Tensor) (target: torch.Tensor) maxEpochs phase =
        // Defines a learning rate schedule with warmup and cosine decay.
        let learningRateSchedule (warmupSteps: int) (totalSteps: int) (baseLr: float32) (step: int) : float32 =
            match step with
            | s when s < warmupSteps -> float32 s / float32 warmupSteps * baseLr // Linearly increases learning rate during warmup.
            | s when s < totalSteps -> let progress = float32 (s - warmupSteps) / float32 (totalSteps - warmupSteps) // Computes progress for cosine decay.
                                       baseLr * 0.5f * (1.0f + cos (MathF.PI * progress)) // Applies cosine decay to reduce learning rate.
            | _ -> 0.0f // Sets learning rate to 0 after total steps.

        [0 .. maxEpochs - 1] |> List.iter (fun epoch -> // Iterates over epochs.
            let counter = (+) epoch 1 // Increments epoch counter for display.
            let baseLr = 5e-4f // Sets base learning rate (5 * 10^-4).
            let warmupSteps = 10 // Sets warmup steps for learning rate schedule.

            let paramGroups = optimizer.ParamGroups // Gets optimizer parameter groups.
            match paramGroups |> Seq.length > 0 with
            | true -> let paramGroup = paramGroups |> Seq.head // Gets the first parameter group.
                      //paramGroup.LearningRate <- float currentLr // Uncomment to set learning rate dynamically (disabled due to hardware limitations).
                      () // Placeholder for disabled learning rate update.
            | false -> failwith "No parameter groups found in optimizer" // Throws an error if no parameter groups exist.

            optimizer.zero_grad() // Clears gradients from the previous step.
            use output = model.forward input // Computes model output (logits) for the input.
            use loss = lossFn.forward(output.contiguous().view(-1L, int64 vocabSize), target.contiguous().view(-1L)) // Computes cross-entropy loss between flattened logits and targets.
            let perplexity = torch.exp(loss).item<float32>() // Computes perplexity (exponential of loss) as a performance metric.

            try
                loss.backward() // Computes gradients of the loss with respect to model parameters.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0) |> ignore<float> // Clips gradients to a maximum norm of 1.0 to prevent exploding gradients.
            with
            | :? System.StackOverflowException as ex -> printfn "StackOverflowException in %s, epoch %d: %s" phase counter (string ex.Message) // Handles stack overflow errors.
            | ex -> printfn "%s" (string ex.Message) // Handles other exceptions.

            optimizer.step() |> ignore<torch.Tensor> // Updates model parameters using Adam optimizer based on computed gradients.
            match counter % 20 with
            | 0 -> printfn "%s Epoch %d, Loss: %.4f, Perplexity: %.4f" phase counter (loss.item<float32>()) perplexity // Prints loss and perplexity every 20 epochs.
            | _ -> () // No action for other epochs.
        )

    // Defines a function to generate a sequence of tokens during inference.
    let rec private generateSeq (model: torch.nn.Module<torch.Tensor, torch.Tensor>) (inputSeq: torch.Tensor) (steps: int)
                                (acc: int64 list) (cont: int64 list -> 'a) : 'a =
        // Trims the input sequence to the maximum context size.
        let trimInput (input: torch.Tensor) =
            match input.shape.[1] > contextSize with
            | true -> input.narrow(1, int64 (int input.shape.[1] - int contextSize), contextSize) // Trims sequence to context size if too long.
            | false -> input // Returns input unchanged if within context size.

        // Samples a token from the output logits using a specified strategy (Top-k, Top-p, or Greedy).
        let sampleLogits (logits: torch.Tensor) =
            match strategy with
            | Top_k -> // Top-k sampling strategy.
                let struct (probs, indices) = torch.topk(logits / temp, int (min topK (int64 vocabSize)), dim = 0) // Selects top-k logits and indices, scaled by temperature.
                use indices = indices // Holds indices of top-k tokens.
                use probs = softmax(probs, dim = 0L) // Applies softmax to top-k logits to get probabilities.
                let idx = torch.multinomial(probs, 1).item<int64>() // Samples an index from the probability distribution.
                indices.[idx].item<int64>() // Returns the sampled token ID.
            | Top_p -> // Top-p (nucleus) sampling strategy.
                use scaledLogits : torch.Tensor = logits / temp // Scales logits by temperature.
                use probs : torch.Tensor = softmax(scaledLogits, dim = 0L) // Applies softmax to get probabilities.
                let struct (sortedProbs: torch.Tensor, sortedIndices: torch.Tensor) = torch.sort(probs, dim = 0L, descending = true) // Sorts probabilities and indices in descending order.
                use sortedProbs = sortedProbs // Holds sorted probabilities.
                use sortedIndices = sortedIndices // Holds sorted indices.
                use cumulativeProbs: torch.Tensor = torch.cumsum(sortedProbs, dim = 0L) // Computes cumulative probabilities.
                use pTensor = torch.tensor(float32 p, device = cumulativeProbs.device) // Creates a tensor for the top-p threshold.
                use mask : torch.Tensor = torch.gt(cumulativeProbs, pTensor) // Creates a mask for probabilities above the threshold.
                use nonzero : torch.Tensor = torch.nonzero mask // Finds indices where cumulative probabilities exceed the threshold.
                let cutoff : int64 =
                    match nonzero.shape.[0] with
                    | 0L -> sortedProbs.shape.[0] // Uses all tokens if no probabilities exceed the threshold.
                    | _ -> nonzero.[0].item<int64>() // Sets cutoff to the first index exceeding the threshold.
                let cutoff = match cutoff with 0L -> sortedProbs.shape.[0] | _ -> cutoff // Ensures cutoff is non-zero.
                use probsTopP = sortedProbs.narrow(0L, 0L, cutoff) // Selects top-p probabilities.
                use indicesTopP = sortedIndices.narrow(0L, 0L, cutoff) // Selects top-p indices.
                let probsTopPSum = probsTopP.sum().item<float32>() // Computes sum of top-p probabilities.
                let idx =
                    match cutoff, probsTopPSum with
                    | 0L, _ -> torch.multinomial(probs, 1).item<int64>() // Falls back to full distribution sampling if cutoff is 0.
                    | _, 0.0f -> torch.multinomial(probs, 1).item<int64>() // Falls back to full distribution if top-p sum is 0.
                    | _, _ -> use probsRenormalised = probsTopP / probsTopP.sum() // Renormalizes top-p probabilities.
                              torch.multinomial(probsRenormalised, 1).item<int64>() // Samples from renormalized top-p distribution.
                indicesTopP.[idx].item<int64>() // Returns the sampled token ID.
            | Greedy -> torch.argmax(logits, dim = 0).item<int64>() // Greedy sampling: selects the token with the highest logit.
            | S -> failwith "Unsupported sampling strategy" // Throws an error for unsupported strategies.

        match steps >= maxSteps with
        | true -> List.rev >> cont <| acc // Stops recursion and returns accumulated tokens if maximum steps are reached.
        | false ->
            use _ = torch.no_grad() // Disables gradient computation for inference.
            use trimmedInput = trimInput inputSeq // Trims input sequence to context size.
            use logits = model.forward trimmedInput // Computes model output logits.
            use lastLogit = logits.select(0, 0L).select(0, -1L) // Selects logits for the last token in the sequence.
            let nextToken = sampleLogits lastLogit // Samples the next token ID.
            match nextToken = eosTokenIdx || nextToken = padTokenIdx with
            | true -> List.rev >> cont <| nextToken :: acc // Stops if end-of-sequence or padding token is sampled.
            | false -> let newInput = torch.cat([|inputSeq; torch.tensor([|nextToken|], device = inputSeq.device).unsqueeze(0L)|], dim = 1L) // Appends sampled token to input sequence.
                       generateSeq model newInput (steps + 1) (nextToken :: acc) cont // Recursively generates the next token.

    // Main function to orchestrate pre-training, fine-tuning, and inference.
    let internal main () =
        use scope = torch.NewDisposeScope() // Creates a scope for automatic tensor disposal.
        let device = match torch.cuda.is_available() with | true -> torch.CUDA | false -> torch.CPU // Selects GPU (CUDA) or CPU based on availability.
        printfn "Using device: %A" <| (string device).ToUpper() // Prints the selected device.

        let dataset = TextData2.getSequences() // Retrieves the dataset for training.
        let (inputData, targetData) = Tokenizer2.createInputTargetPairs dataset // Tokenizes dataset into input-target pairs.
        use input = torch.tensor(inputData, device = device) // Converts input data to a tensor on the selected device.
        use target = torch.tensor(targetData, device = device) // Converts target data to a tensor on the selected device.

        printfn "Starting pre-training..." // Indicates the start of pre-training.
        let useLora = false // Disables LoRA for pre-training (recommended for full model training).
        use model : torch.nn.Module<torch.Tensor, torch.Tensor> = (new Transformer(int64 vocabSize, dModel, nHeads, numLayers, device, useLora)).``to``(device) // Initializes and moves the transformer model to the device.
        use lossFn = new CrossEntropyLoss(ignore_index = padTokenIdx) // Creates a cross-entropy loss function, ignoring padding tokens.
        use optimizer = torch.optim.Adam(model.parameters(), lr = learningRate) // Initializes Adam optimizer with model parameters and learning rate.
        model.train() // Sets the model to training mode.
        trainEpoch model optimizer lossFn input target epochs "Pre-training" // Runs pre-training for the specified number of epochs.
        model.save("model4.pt") |> ignore<torch.nn.Module> // Saves the pre-trained model to a file.

        printfn "Starting fine-tuning..." // Indicates the start of fine-tuning.
        model.load("model4.pt", strict = false) |> ignore<torch.nn.Module> // Loads the pre-trained model, allowing non-strict loading for flexibility.
        let freezeBaseWeights (model: torch.nn.Module) = // Defines a function to freeze non-LoRA weights during fine-tuning.
            model.named_parameters()
            |> Seq.iter (fun struct (name, param) ->
                match name.EndsWith(".A") || name.EndsWith(".B") with
                | true -> () // Keeps LoRA parameters trainable.
                | false -> param.requires_grad <- false // Freezes non-LoRA parameters.
            )
        match useLora with | true -> freezeBaseWeights model | false -> () // Freezes base weights if LoRA is enabled.
        let (fineTuneInputData, fineTuneTargetData) = TextData2.getFineTuningCausalLMSequences () // Retrieves fine-tuning dataset.
        use fineTuneInput = torch.tensor(fineTuneInputData, device = device) // Converts fine-tuning input to a tensor.
        use fineTuneTarget = torch.tensor(fineTuneTargetData, device = device) // Converts fine-tuning target to a tensor.
        use fineTuneOptimizer = torch.optim.Adam(model.parameters(), lr = learningRate) // Initializes Adam optimizer for fine-tuning.
        model.train() // Sets the model to training mode for fine-tuning.
        trainEpoch model fineTuneOptimizer lossFn fineTuneInput fineTuneTarget fineTuneEpochs "Fine-tuning" // Runs fine-tuning for the specified epochs.

        let gradNorm = model.parameters() |> Seq.filter (fun p1 -> p1.requires_grad) |> Seq.map (fun p2 -> p2.grad.norm()) |> Seq.reduce (fun acc norm -> acc + norm) // Computes the total gradient norm for trainable parameters.
        printfn "Gradient norm: %.4f" (gradNorm.item<float32>()) // Prints the gradient norm.
        model.save("model4.pt") |> ignore<torch.nn.Module> // Saves the fine-tuned model.

        printfn "Generating sequence after fine-tuning..." // Indicates the start of inference.
        model.load("model4.pt") |> ignore<torch.nn.Module> // Loads the fine-tuned model.
        let promptContent = "What is the colour of the sky? <sep>" // Defines the prompt for sequence generation.
        let promptTokens = Tokenizer2.tokenize promptContent |> List.toArray // Tokenizes the prompt into token IDs.
        model.``to``("cpu") |> ignore<torch.nn.Module<torch.Tensor, torch.Tensor>> // Moves the model to CPU for inference.
        model.eval() // Sets the model to evaluation mode.
        use inputSeq = torch.tensor(promptTokens, device = torch.CPU).unsqueeze 0L // Converts prompt tokens to a tensor with batch dimension.
        let generated = generateSeq model inputSeq initialStep acc id // Generates a sequence starting from the prompt.
        printf "Generated sequence (token IDs): "
        generated |> List.iter (printf "%d ") // Prints generated token IDs.
        printf "Generated sequence (words): "
        generated |> List.iter (fun id ->
            match id >= 0L && id < int64 vocabulary.Length with
            | true -> printf "%s " (vocabulary |> List.item (int id)) // Converts token IDs to words using the vocabulary.
            | false -> printf "<unk> ") // Prints <unk> for unknown tokens.