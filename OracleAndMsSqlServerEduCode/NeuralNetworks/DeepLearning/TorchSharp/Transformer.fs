namespace NeuralNetworks

open System
open TorchSharp
open TorchSharp.Modules
open type torch.nn
open type torch.nn.functional

module Transformer_TorchSharp =

    // Vocabulary and hyperparameters
    let vocabulary = [|"The"; "Sun"; "is"; "yellow"; "black"; "sky"; "blue"; "<eos>"|]
    let vocabSize = vocabulary.Length // 8
    let dModel = 128L
    let epochs = 10000
    let batch = 64L
    let headDim = 32L // 128 / 4
    let nHeads = 4L

    // Transformer Decoder Layer with causal mask
    type TransformerDecoderLayer(dModel: int64, nHeads: int64) as self =
        inherit Module<TorchSharp.torch.Tensor, TorchSharp.torch.Tensor>("TransformerDecoderLayer")
        let qkvProj = Linear(dModel, dModel * 3L)
        let outputProj = Linear(dModel, dModel)
        let ff1 = Linear(dModel, dModel * 4L)
        let ff2 = Linear(dModel * 4L, dModel)
        let norm1 = LayerNorm([|dModel|])
        let norm2 = LayerNorm([|dModel|])

        do self.RegisterComponents()

        override _.forward(x) =
            let (batch, seq, _) = x.shape.[0], x.shape.[1], x.shape.[2]
            let newShape = [|batch; seq; 3L; nHeads; headDim|] // [batch, seq, 3, nHeads, headDim]
            x
            |> qkvProj.forward
            |> fun (t: TorchSharp.torch.Tensor) -> (* printfn "Tensor shape before view: %A, dtype: %A" t.shape t.dtype;*) t.view(newShape)
            |> fun (qkv: TorchSharp.torch.Tensor) ->
                use q: TorchSharp.torch.Tensor = qkv.select(2, 0L) // [batch, seq, nHeads, headDim]
                use k: TorchSharp.torch.Tensor = qkv.select(2, 1L) // [batch, seq, nHeads, headDim]
                use v: TorchSharp.torch.Tensor = qkv.select(2, 2L) // [batch, seq, nHeads, headDim]
                (q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)) // [batch, nHeads, seq, headDim]
            |> fun (q: TorchSharp.torch.Tensor, k: TorchSharp.torch.Tensor, v: TorchSharp.torch.Tensor) ->
                use scores: TorchSharp.torch.Tensor = TorchSharp.torch.matmul(q, k.transpose(-2, -1)) / sqrt(float headDim) // [batch, nHeads, seq, seq]
                // Add causal mask
                use mask = TorchSharp.torch.triu(TorchSharp.torch.ones([|seq; seq|], device=scores.device), diagonal=1L).to_type(TorchSharp.torch.ScalarType.Bool)
                use maskedScores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), System.Single.NegativeInfinity)
                use context: TorchSharp.torch.Tensor =
                    maskedScores
                    |> fun (t: TorchSharp.torch.Tensor) -> torch.nn.functional.softmax(t, -1L) // [batch, nHeads, seq, seq]
                    |> fun (t: TorchSharp.torch.Tensor) -> TorchSharp.torch.matmul(t, v) // [batch, nHeads, seq, headDim]
                    |> fun (t: TorchSharp.torch.Tensor) -> t.transpose(1, 2).contiguous().view(batch, seq, dModel) // [batch, seq, dModel]
                context
                |> outputProj.forward
                |> fun (output: TorchSharp.torch.Tensor) -> norm1.forward(x + output) // Residual + LayerNorm
                |> fun (output: TorchSharp.torch.Tensor) ->
                    output
                    |> ff1.forward
                    |> relu
                    |> ff2.forward
                    |> fun (ffOutput: TorchSharp.torch.Tensor) -> norm2.forward(output + ffOutput) // Residual + LayerNorm

    // Transformer Model
    type Transformer(vocabSize: int64, dModel: int64, nHeads: int64) as self =
        inherit Module<TorchSharp.torch.Tensor, TorchSharp.torch.Tensor>("Transformer")
        let embedding = Embedding(vocabSize, dModel)
        let decoderLayer = new TransformerDecoderLayer(dModel, nHeads)
        let outputLayer = Linear(dModel, vocabSize)

        do self.RegisterComponents()

        override _.forward(x) =
            let emb = embedding.forward(x) // [batch, seq, dModel]
            //printfn "Embedding output shape: %A, dtype: %A" emb.shape emb.dtype
            let dec = decoderLayer.forward(emb) // [batch, seq, dModel]
            //printfn "Decoder output shape: %A, dtype: %A" dec.shape dec.dtype
            let out = outputLayer.forward(dec).to_type(TorchSharp.torch.ScalarType.Float32) // [batch, seq, vocabSize]
            //printfn "Output layer shape: %A, dtype: %A" out.shape out.dtype
            out

    // Recursive training loop
    let rec trainEpoch (model: torch.nn.Module<TorchSharp.torch.Tensor, TorchSharp.torch.Tensor>) (optimizer: torch.optim.Optimizer) (lossFn: TorchSharp.Modules.CrossEntropyLoss) (input: TorchSharp.torch.Tensor) (target: TorchSharp.torch.Tensor) epoch maxEpochs =
        match epoch with
        | e when e >= maxEpochs -> ()
        | _ ->
            optimizer.zero_grad()
            let output: TorchSharp.torch.Tensor = model.forward(input) // [batch, seq, vocabSize]
            //printfn "Train output shape: %A, dtype: %A" output.shape output.dtype
            //printfn "Target shape: %A, dtype: %A" target.shape target.dtype
            use loss: TorchSharp.torch.Tensor = output.view(-1L, vocabSize) |> (fun logits -> lossFn.forward(logits, target.view(-1L)))
            loss.backward()
            optimizer.step() |> ignore
            //printfn "Epoch %d, Loss: %.4f" (epoch + 1) (loss.item<float32>())
            trainEpoch model optimizer lossFn input target (epoch + 1) maxEpochs

    // Recursive inference loop with <eos> masking
    let rec generate (model: torch.nn.Module<TorchSharp.torch.Tensor, TorchSharp.torch.Tensor>) (inputSeq: TorchSharp.torch.Tensor) steps maxSteps acc =
        match steps with
        | s when s >= maxSteps -> acc
        | _ ->
            let _ = TorchSharp.torch.no_grad()
            let logits: TorchSharp.torch.Tensor = model.forward(inputSeq) // [1, seq, vocabSize]
            //printfn "Generate logits shape: %A, dtype: %A" logits.shape logits.dtype
            let lastLogit: TorchSharp.torch.Tensor = logits.select(0, 0L).select(0, -1L) // [vocabSize]
            //printfn "Raw logits: %A" (lastLogit.cpu().data<float32>().ToArray())
            // Mask <eos> (index 7) in first step
            let adjustedLogit = if steps = 0 then lastLogit.index_fill_(0, TorchSharp.torch.tensor([|7L|], device=lastLogit.device), System.Single.NegativeInfinity) else lastLogit
            let temp = 0.05f // Lower temperature for sharper probabilities
            let probs = torch.nn.functional.softmax(adjustedLogit / temp, dim=0L)
            //printfn "Logit probabilities: %A" (probs.cpu().data<float32>().ToArray())
            let nextToken = torch.multinomial(probs, 1).item<int64>()
            // let nextToken = probs.argmax().item<int64>() // Optional: Test with argmax
            let newAcc = nextToken :: acc
            let newInput: TorchSharp.torch.Tensor = TorchSharp.torch.cat([|inputSeq; TorchSharp.torch.tensor([|nextToken|], device=inputSeq.device).unsqueeze(0L)|], dim=1L)
            generate model newInput (steps + 1) maxSteps newAcc

    // Main program
    let main () =    
        // Device selection with pattern matching
        let device: TorchSharp.torch.Device =
            match TorchSharp.torch.cuda.is_available() with
            | true -> TorchSharp.torch.CUDA
            | false -> TorchSharp.torch.CPU
        //printfn "Using device: %A" device

        // Initialize model, loss, and optimizer
        use model = (new Transformer(int64 vocabSize, dModel, nHeads)).``to``(device)
        use lossFn = new TorchSharp.Modules.CrossEntropyLoss()
        use optimizer = TorchSharp.torch.optim.Adam(model.parameters(), lr=0.0005) // Lower learning rate

        let inputData = Array2D.init 64 3 (fun i k ->
            match (i, k) with
            // "the Sun is" -> "is yellow <eos>" (30 examples)
            | 0, 0 -> 0L  | 0, 1 -> 1L  | 0, 2 -> 2L   // the Sun is
            | 1, 0 -> 0L  | 1, 1 -> 1L  | 1, 2 -> 2L   // the Sun is
            | 2, 0 -> 0L  | 2, 1 -> 1L  | 2, 2 -> 2L   // the Sun is
            | 3, 0 -> 0L  | 3, 1 -> 1L  | 3, 2 -> 2L   // the Sun is
            | 4, 0 -> 0L  | 4, 1 -> 1L  | 4, 2 -> 2L   // the Sun is
            | 5, 0 -> 0L  | 5, 1 -> 1L  | 5, 2 -> 2L   // the Sun is
            | 6, 0 -> 0L  | 6, 1 -> 1L  | 6, 2 -> 2L   // the Sun is
            | 7, 0 -> 0L  | 7, 1 -> 1L  | 7, 2 -> 2L   // the Sun is
            | 8, 0 -> 0L  | 8, 1 -> 1L  | 8, 2 -> 2L   // the Sun is
            | 9, 0 -> 0L  | 9, 1 -> 1L  | 9, 2 -> 2L   // the Sun is
            | 10, 0 -> 0L | 10, 1 -> 1L | 10, 2 -> 2L  // the Sun is
            | 11, 0 -> 0L | 11, 1 -> 1L | 11, 2 -> 2L  // the Sun is
            | 12, 0 -> 0L | 12, 1 -> 1L | 12, 2 -> 2L  // the Sun is
            | 13, 0 -> 0L | 13, 1 -> 1L | 13, 2 -> 2L  // the Sun is
            | 14, 0 -> 0L | 14, 1 -> 1L | 14, 2 -> 2L  // the Sun is
            | 15, 0 -> 0L | 15, 1 -> 1L | 15, 2 -> 2L  // the Sun is
            | 16, 0 -> 0L | 16, 1 -> 1L | 16, 2 -> 2L  // the Sun is
            | 17, 0 -> 0L | 17, 1 -> 1L | 17, 2 -> 2L  // the Sun is
            | 18, 0 -> 0L | 18, 1 -> 1L | 18, 2 -> 2L  // the Sun is
            | 19, 0 -> 0L | 19, 1 -> 1L | 19, 2 -> 2L  // the Sun is
            | 20, 0 -> 0L | 20, 1 -> 1L | 20, 2 -> 2L  // the Sun is
            | 21, 0 -> 0L | 21, 1 -> 1L | 21, 2 -> 2L  // the Sun is
            | 22, 0 -> 0L | 22, 1 -> 1L | 22, 2 -> 2L  // the Sun is
            | 23, 0 -> 0L | 23, 1 -> 1L | 23, 2 -> 2L  // the Sun is
            | 24, 0 -> 0L | 24, 1 -> 1L | 24, 2 -> 2L  // the Sun is
            | 25, 0 -> 0L | 25, 1 -> 1L | 25, 2 -> 2L  // the Sun is
            | 26, 0 -> 0L | 26, 1 -> 1L | 26, 2 -> 2L  // the Sun is
            | 27, 0 -> 0L | 27, 1 -> 1L | 27, 2 -> 2L  // the Sun is
            | 28, 0 -> 0L | 28, 1 -> 1L | 28, 2 -> 2L  // the Sun is
            | 29, 0 -> 0L | 29, 1 -> 1L | 29, 2 -> 2L  // the Sun is
            // "the sky is" -> "is blue <eos>" (20 examples)
            | 30, 0 -> 0L | 30, 1 -> 5L | 30, 2 -> 2L  // the sky is
            | 31, 0 -> 0L | 31, 1 -> 5L | 31, 2 -> 2L  // the sky is
            | 32, 0 -> 0L | 32, 1 -> 5L | 32, 2 -> 2L  // the sky is
            | 33, 0 -> 0L | 33, 1 -> 5L | 33, 2 -> 2L  // the sky is
            | 34, 0 -> 0L | 34, 1 -> 5L | 34, 2 -> 2L  // the sky is
            | 35, 0 -> 0L | 35, 1 -> 5L | 35, 2 -> 2L  // the sky is
            | 36, 0 -> 0L | 36, 1 -> 5L | 36, 2 -> 2L  // the sky is
            | 37, 0 -> 0L | 37, 1 -> 5L | 37, 2 -> 2L  // the sky is
            | 38, 0 -> 0L | 38, 1 -> 5L | 38, 2 -> 2L  // the sky is
            | 39, 0 -> 0L | 39, 1 -> 5L | 39, 2 -> 2L  // the sky is
            | 40, 0 -> 0L | 40, 1 -> 5L | 40, 2 -> 2L  // the sky is
            | 41, 0 -> 0L | 41, 1 -> 5L | 41, 2 -> 2L  // the sky is
            | 42, 0 -> 0L | 42, 1 -> 5L | 42, 2 -> 2L  // the sky is
            | 43, 0 -> 0L | 43, 1 -> 5L | 43, 2 -> 2L  // the sky is
            | 44, 0 -> 0L | 44, 1 -> 5L | 44, 2 -> 2L  // the sky is
            | 45, 0 -> 0L | 45, 1 -> 5L | 45, 2 -> 2L  // the sky is
            | 46, 0 -> 0L | 46, 1 -> 5L | 46, 2 -> 2L  // the sky is
            | 47, 0 -> 0L | 47, 1 -> 5L | 47, 2 -> 2L  // the sky is
            | 48, 0 -> 0L | 48, 1 -> 5L | 48, 2 -> 2L  // the sky is
            | 49, 0 -> 0L | 49, 1 -> 5L | 49, 2 -> 2L  // the sky is
            // "the Sun is" -> "is black <eos>" (4 examples)
            | 50, 0 -> 0L | 50, 1 -> 1L | 50, 2 -> 2L  // the Sun is
            | 51, 0 -> 0L | 51, 1 -> 1L | 51, 2 -> 2L  // the Sun is
            | 52, 0 -> 0L | 52, 1 -> 1L | 52, 2 -> 2L  // the Sun is
            | 53, 0 -> 0L | 53, 1 -> 1L | 53, 2 -> 2L  // the Sun is
            // "the sky is" -> "is black <eos>" (4 examples)
            | 54, 0 -> 0L | 54, 1 -> 5L | 54, 2 -> 2L  // the sky is
            | 55, 0 -> 0L | 55, 1 -> 5L | 55, 2 -> 2L  // the sky is
            | 56, 0 -> 0L | 56, 1 -> 5L | 56, 2 -> 2L  // the sky is
            | 57, 0 -> 0L | 57, 1 -> 5L | 57, 2 -> 2L  // the sky is
            // Creative variations (6 examples)
            | 58, 0 -> 0L | 58, 1 -> 5L | 58, 2 -> 2L  // the sky is -> is yellow <eos>
            | 59, 0 -> 0L | 59, 1 -> 5L | 59, 2 -> 2L  // the Sun is -> is blue <eos>
            | 60, 0 -> 0L | 60, 1 -> 5L | 60, 2 -> 2L  // the sky is -> is yellow <eos>
            | 61, 0 -> 0L | 61, 1 -> 5L | 61, 2 -> 2L  // the Sun is -> is blue <eos>
            | 62, 0 -> 0L | 62, 1 -> 5L | 62, 2 -> 2L  // the sky is -> is blue <eos>
            | 63, 0 -> 0L | 63, 1 -> 5L | 63, 2 -> 2L  // the Sun is -> is yellow <eos>
            | _ -> failwith "Invalid index")
        
        let targetData = Array2D.init 64 3 (fun i k ->
            match (i, k) with
            // "the Sun is" -> "is yellow <eos>" (30 examples)
            | 0, 0 -> 2L  | 0, 1 -> 3L  | 0, 2 -> 7L   // is yellow <eos>
            | 1, 0 -> 2L  | 1, 1 -> 3L  | 1, 2 -> 7L   // is yellow <eos>
            | 2, 0 -> 2L  | 2, 1 -> 3L  | 2, 2 -> 7L   // is yellow <eos>
            | 3, 0 -> 2L  | 3, 1 -> 3L  | 3, 2 -> 7L   // is yellow <eos>
            | 4, 0 -> 2L  | 4, 1 -> 3L  | 4, 2 -> 7L   // is yellow <eos>
            | 5, 0 -> 2L  | 5, 1 -> 3L  | 5, 2 -> 7L   // is yellow <eos>
            | 6, 0 -> 2L  | 6, 1 -> 3L  | 6, 2 -> 7L   // is yellow <eos>
            | 7, 0 -> 2L  | 7, 1 -> 3L  | 7, 2 -> 7L   // is yellow <eos>
            | 8, 0 -> 2L  | 8, 1 -> 3L  | 8, 2 -> 7L   // is yellow <eos>
            | 9, 0 -> 2L  | 9, 1 -> 3L  | 9, 2 -> 7L   // is yellow <eos>
            | 10, 0 -> 2L | 10, 1 -> 3L | 10, 2 -> 7L  // is yellow <eos>
            | 11, 0 -> 2L | 11, 1 -> 3L | 11, 2 -> 7L  // is yellow <eos>
            | 12, 0 -> 2L | 12, 1 -> 3L | 12, 2 -> 7L  // is yellow <eos>
            | 13, 0 -> 2L | 13, 1 -> 3L | 13, 2 -> 7L  // is yellow <eos>
            | 14, 0 -> 2L | 14, 1 -> 3L | 14, 2 -> 7L  // is yellow <eos>
            | 15, 0 -> 2L | 15, 1 -> 3L | 15, 2 -> 7L  // is yellow <eos>
            | 16, 0 -> 2L | 16, 1 -> 3L | 16, 2 -> 7L  // is yellow <eos>
            | 17, 0 -> 2L | 17, 1 -> 3L | 17, 2 -> 7L  // is yellow <eos>
            | 18, 0 -> 2L | 18, 1 -> 3L | 18, 2 -> 7L  // is yellow <eos>
            | 19, 0 -> 2L | 19, 1 -> 3L | 19, 2 -> 7L  // is yellow <eos>
            | 20, 0 -> 2L | 20, 1 -> 3L | 20, 2 -> 7L  // is yellow <eos>
            | 21, 0 -> 2L | 21, 1 -> 3L | 21, 2 -> 7L  // is yellow <eos>
            | 22, 0 -> 2L | 22, 1 -> 3L | 22, 2 -> 7L  // is yellow <eos>
            | 23, 0 -> 2L | 23, 1 -> 3L | 23, 2 -> 7L  // is yellow <eos>
            | 24, 0 -> 2L | 24, 1 -> 3L | 24, 2 -> 7L  // is yellow <eos>
            | 25, 0 -> 2L | 25, 1 -> 3L | 25, 2 -> 7L  // is yellow <eos>
            | 26, 0 -> 2L | 26, 1 -> 3L | 26, 2 -> 7L  // is yellow <eos>
            | 27, 0 -> 2L | 27, 1 -> 3L | 27, 2 -> 7L  // is yellow <eos>
            | 28, 0 -> 2L | 28, 1 -> 3L | 28, 2 -> 7L  // is yellow <eos>
            | 29, 0 -> 2L | 29, 1 -> 3L | 29, 2 -> 7L  // is yellow <eos>
            // "the sky is" -> "is blue <eos>" (20 examples)
            | 30, 0 -> 2L | 30, 1 -> 6L | 30, 2 -> 7L  // is blue <eos>
            | 31, 0 -> 2L | 31, 1 -> 6L | 31, 2 -> 7L  // is blue <eos>
            | 32, 0 -> 2L | 32, 1 -> 6L | 32, 2 -> 7L  // is blue <eos>
            | 33, 0 -> 2L | 33, 1 -> 6L | 33, 2 -> 7L  // is blue <eos>
            | 34, 0 -> 2L | 34, 1 -> 6L | 34, 2 -> 7L  // is blue <eos>
            | 35, 0 -> 2L | 35, 1 -> 6L | 35, 2 -> 7L  // is blue <eos>
            | 36, 0 -> 2L | 36, 1 -> 6L | 36, 2 -> 7L  // is blue <eos>
            | 37, 0 -> 2L | 37, 1 -> 6L | 37, 2 -> 7L  // is blue <eos>
            | 38, 0 -> 2L | 38, 1 -> 6L | 38, 2 -> 7L  // is blue <eos>
            | 39, 0 -> 2L | 39, 1 -> 6L | 39, 2 -> 7L  // is blue <eos>
            | 40, 0 -> 2L | 40, 1 -> 6L | 40, 2 -> 7L  // is blue <eos>
            | 41, 0 -> 2L | 41, 1 -> 6L | 41, 2 -> 7L  // is blue <eos>
            | 42, 0 -> 2L | 42, 1 -> 6L | 42, 2 -> 7L  // is blue <eos>
            | 43, 0 -> 2L | 43, 1 -> 6L | 43, 2 -> 7L  // is blue <eos>
            | 44, 0 -> 2L | 44, 1 -> 6L | 44, 2 -> 7L  // is blue <eos>
            | 45, 0 -> 2L | 45, 1 -> 6L | 45, 2 -> 7L  // is blue <eos>
            | 46, 0 -> 2L | 46, 1 -> 6L | 46, 2 -> 7L  // is blue <eos>
            | 47, 0 -> 2L | 47, 1 -> 6L | 47, 2 -> 7L  // is blue <eos>
            | 48, 0 -> 2L | 48, 1 -> 6L | 48, 2 -> 7L  // is blue <eos>
            | 49, 0 -> 2L | 49, 1 -> 6L | 49, 2 -> 7L  // is blue <eos>
            // "the Sun is" -> "is black <eos>" (4 examples)
            | 50, 0 -> 2L | 50, 1 -> 4L | 50, 2 -> 7L  // is black <eos>
            | 51, 0 -> 2L | 51, 1 -> 4L | 51, 2 -> 7L  // is black <eos>
            | 52, 0 -> 2L | 52, 1 -> 4L | 52, 2 -> 7L  // is black <eos>
            | 53, 0 -> 2L | 53, 1 -> 4L | 53, 2 -> 7L  // is black <eos>
            // "the sky is" -> "is black <eos>" (4 examples)
            | 54, 0 -> 2L | 54, 1 -> 4L | 54, 2 -> 7L  // is black <eos>
            | 55, 0 -> 2L | 55, 1 -> 4L | 55, 2 -> 7L  // is black <eos>
            | 56, 0 -> 2L | 56, 1 -> 4L | 56, 2 -> 7L  // is black <eos>
            | 57, 0 -> 2L | 57, 1 -> 4L | 57, 2 -> 7L  // is black <eos>
            // Creative variations (6 examples)
            | 58, 0 -> 2L | 58, 1 -> 3L | 58, 2 -> 7L  // is yellow <eos>
            | 59, 0 -> 2L | 59, 1 -> 3L | 59, 2 -> 7L  // is yellow <eos>
            | 60, 0 -> 2L | 60, 1 -> 3L | 60, 2 -> 7L  // is yellow <eos>
            | 61, 0 -> 2L | 61, 1 -> 3L | 61, 2 -> 7L  // is yellow <eos>
            | 62, 0 -> 2L | 62, 1 -> 3L | 62, 2 -> 7L  // is yellow <eos>
            | 63, 0 -> 2L | 63, 1 -> 3L | 63, 2 -> 7L  // is yellow <eos>
            | _ -> failwith "Invalid index")

        use input = TorchSharp.torch.tensor(inputData, device=device) // [4, 3]
        use target = TorchSharp.torch.tensor(targetData, device=device) // [4, 3]

        // Train model
        model.train()
        trainEpoch model optimizer lossFn input target 0 epochs
               
        // Inference: Start with "The Sun is" -> [0, 1, 2]
        model.eval()

        //tady se specialne ptam, co je za "The Sun is" [|0L; 1L; 2L|]
        //let inputSeq: TorchSharp.torch.Tensor = TorchSharp.torch.tensor([|0L; 1L; 2L|], device=device).unsqueeze(0L) // [1, 3]
        //You could test other sequences, e.g., "The sky is" ([0, 5, 2]), to get blue <eos>       
        use inputSeq: TorchSharp.torch.Tensor = TorchSharp.torch.tensor([|0L; 1L; 2L|], device=device).unsqueeze(0L) 
        printf "Generated sequence (token IDs): "
        let generated = generate model inputSeq 0 2 [] |> List.rev // Generate 2 tokens (yellow, <eos>)
        generated |> List.iter (printf "%d ")
        printfn ""

        // Map token IDs to words
        printf "Generated sequence (words): "
        generated |> List.iter (fun id -> printf "%s " vocabulary.[int id])
        printfn ""


