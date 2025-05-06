namespace NeuralNetworks

open System
open TorchSharp
open TorchSharp.Modules
open type torch.nn
open type torch.nn.functional

module Transformer_TorchSharp =

    // Vocabulary and hyperparameters
    let vocabulary = [|"the"; "Sun"; "is"; "yellow"; "black"; "sky"; "blue"; "<eos>"|]
    let vocabSize = vocabulary.Length // 8
    let dModel = 128L
    let epochs = 15
    let batch = 4L
    let seq = 3L
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
            |> fun (t: TorchSharp.torch.Tensor) -> printfn "Tensor shape before view: %A, dtype: %A" t.shape t.dtype; t.view(newShape)
            |> fun (qkv: TorchSharp.torch.Tensor) ->
                let q: TorchSharp.torch.Tensor = qkv.select(2, 0L) // [batch, seq, nHeads, headDim]
                let k: TorchSharp.torch.Tensor = qkv.select(2, 1L) // [batch, seq, nHeads, headDim]
                let v: TorchSharp.torch.Tensor = qkv.select(2, 2L) // [batch, seq, nHeads, headDim]
                (q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)) // [batch, nHeads, seq, headDim]
            |> fun (q: TorchSharp.torch.Tensor, k: TorchSharp.torch.Tensor, v: TorchSharp.torch.Tensor) ->
                let scores: TorchSharp.torch.Tensor = TorchSharp.torch.matmul(q, k.transpose(-2, -1)) / sqrt(float headDim) // [batch, nHeads, seq, seq]
                // Add causal mask
                let mask = TorchSharp.torch.triu(TorchSharp.torch.ones([|seq; seq|], device=scores.device), diagonal=1L).to_type(TorchSharp.torch.ScalarType.Bool)
                let maskedScores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), System.Single.NegativeInfinity)
                let context: TorchSharp.torch.Tensor =
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
            printfn "Embedding output shape: %A, dtype: %A" emb.shape emb.dtype
            let dec = decoderLayer.forward(emb) // [batch, seq, dModel]
            printfn "Decoder output shape: %A, dtype: %A" dec.shape dec.dtype
            let out = outputLayer.forward(dec).to_type(TorchSharp.torch.ScalarType.Float32) // [batch, seq, vocabSize]
            printfn "Output layer shape: %A, dtype: %A" out.shape out.dtype
            out

    // Recursive training loop
    let rec trainEpoch (model: torch.nn.Module<TorchSharp.torch.Tensor, TorchSharp.torch.Tensor>) (optimizer: torch.optim.Optimizer) (lossFn: TorchSharp.Modules.CrossEntropyLoss) (input: TorchSharp.torch.Tensor) (target: TorchSharp.torch.Tensor) epoch maxEpochs =
        match epoch with
        | e when e >= maxEpochs -> ()
        | _ ->
            optimizer.zero_grad()
            let output: TorchSharp.torch.Tensor = model.forward(input) // [batch, seq, vocabSize]
            printfn "Train output shape: %A, dtype: %A" output.shape output.dtype
            printfn "Target shape: %A, dtype: %A" target.shape target.dtype
            let loss: TorchSharp.torch.Tensor = output.view(-1L, vocabSize) |> (fun logits -> lossFn.forward(logits, target.view(-1L)))
            loss.backward()
            optimizer.step() |> ignore
            printfn "Epoch %d, Loss: %.4f" (epoch + 1) (loss.item<float32>())
            trainEpoch model optimizer lossFn input target (epoch + 1) maxEpochs

    // Recursive inference loop with <eos> masking
    let rec generate (model: torch.nn.Module<TorchSharp.torch.Tensor, TorchSharp.torch.Tensor>) (inputSeq: TorchSharp.torch.Tensor) steps maxSteps acc =
        match steps with
        | s when s >= maxSteps -> acc
        | _ ->
            use _ = TorchSharp.torch.no_grad()
            let logits: TorchSharp.torch.Tensor = model.forward(inputSeq) // [1, seq, vocabSize]
            printfn "Generate logits shape: %A, dtype: %A" logits.shape logits.dtype
            let lastLogit: TorchSharp.torch.Tensor = logits.select(0, 0L).select(0, -1L) // [vocabSize]
            printfn "Raw logits: %A" (lastLogit.cpu().data<float32>().ToArray())
            // Mask <eos> (index 7) in first step
            let adjustedLogit = if steps = 0 then lastLogit.index_fill_(0, TorchSharp.torch.tensor([|7L|], device=lastLogit.device), System.Single.NegativeInfinity) else lastLogit
            let temp = 0.5f // Lower temperature for sharper probabilities
            let probs = torch.nn.functional.softmax(adjustedLogit / temp, dim=0L)
            printfn "Logit probabilities: %A" (probs.cpu().data<float32>().ToArray())
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
        printfn "Using device: %A" device

        // Initialize model, loss, and optimizer
        let model = (new Transformer(int64 vocabSize, dModel, nHeads)).``to``(device)
        let lossFn = new TorchSharp.Modules.CrossEntropyLoss()
        let optimizer = TorchSharp.torch.optim.Adam(model.parameters(), lr=0.0005) // Lower learning rate

        // Training data: 4 examples
        let inputData = Array2D.init 4 3 (fun i k ->
            match (i, k) with
            | 0, 0 -> 0L // the
            | 0, 1 -> 1L // Sun
            | 0, 2 -> 2L // is
            | 1, 0 -> 0L // the
            | 1, 1 -> 5L // sky
            | 1, 2 -> 2L // is
            | 2, 0 -> 0L // the
            | 2, 1 -> 1L // Sun
            | 2, 2 -> 2L // is
            | 3, 0 -> 0L // the
            | 3, 1 -> 5L // sky
            | 3, 2 -> 2L // is
            | _ -> failwith "Invalid index")
        let input = TorchSharp.torch.tensor(inputData, device=device) // [4, 3]

        let targetData = Array2D.init 4 3 (fun i k ->
            match (i, k) with
            | 0, 0 -> 2L // is
            | 0, 1 -> 3L // yellow
            | 0, 2 -> 7L // <eos>
            | 1, 0 -> 2L // is
            | 1, 1 -> 6L // blue
            | 1, 2 -> 7L // <eos>
            | 2, 0 -> 2L // is
            | 2, 1 -> 3L // yellow
            | 2, 2 -> 7L // <eos>
            | 3, 0 -> 2L // is
            | 3, 1 -> 6L // blue
            | 3, 2 -> 7L // <eos>
            | _ -> failwith "Invalid index")
        let target = TorchSharp.torch.tensor(targetData, device=device) // [4, 3]

        // Train model
        model.train()
        trainEpoch model optimizer lossFn input target 0 epochs

        // Inference: Start with "The Sun is" -> [0, 1, 2]
        model.eval()

        //tady se specialne ptam, co je za "The Sun is" [|0L; 1L; 2L|]
        //let inputSeq: TorchSharp.torch.Tensor = TorchSharp.torch.tensor([|0L; 1L; 2L|], device=device).unsqueeze(0L) // [1, 3]
        //You could test other sequences, e.g., "the sky is" ([0, 5, 2]), to get blue <eos>       
        let inputSeq: TorchSharp.torch.Tensor = TorchSharp.torch.tensor([|0L; 1L; 2L|], device=device).unsqueeze(0L) 
        printf "Generated sequence (token IDs): "
        let generated = generate model inputSeq 0 2 [] |> List.rev // Generate 2 tokens (yellow, <eos>)
        generated |> List.iter (printf "%d ")
        printfn ""

        // Map token IDs to words
        printf "Generated sequence (words): "
        generated |> List.iter (fun id -> printf "%s " vocabulary.[int id])
        printfn ""

  