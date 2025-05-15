namespace NeuralNetworks

open System
open TorchSharp
open TorchSharp.Modules

open type torch.nn
open type torch.nn.functional

module Transformer_TorchSharp =

    // Vocabulary and hyperparameters
    // Remark: Defines the vocabulary for tokenization, mapping words to indices (e.g., "The" = 0, "Sun" = 1, ..., "<eos>" = 7).
    let private vocabulary = [|"The"; "Sun"; "is"; "yellow"; "black"; "sky"; "blue"; "<eos>"|]   //pouze pro indexizaci
    let private mv = vocabulary |> Array.mapi (fun i _ -> i) //tady zbytecne, ale aby to formalne bylo jako u realnych modelu
    let private vocabSize = vocabulary.Length
    
    let [<Literal>] private dModel = 64L //embeddings of size 64
    let [<Literal>] private epochs = 20000
    let [<Literal>] private fineTuneEpochs = 2000
    let private batch = 32L
    let [<Literal>] private fineTuneBatch = 10L
    let [<Literal>] private headDim = 32L
    let [<Literal>] private nHeads = 2L
    let [<Literal>] private numLayers = 2
    let [<Literal>] private dropoutRate = 0.1f
    let [<Literal>] private topK = 3L

    /// Generates positional encodings for token positions
    let private getPositionalEncodings (seqLen: int64) (dModel: int64) : torch.Tensor =

        let position = torch.arange(seqLen, dtype=torch.float32).unsqueeze(1)
        let divTerm = torch.exp(torch.arange(0L, dModel, 2L, dtype=torch.float32) * -(Math.Log(10000.0) / float dModel))
        
        let encodings = torch.zeros([|seqLen; dModel|])
        encodings.index_copy_(1, torch.arange(0L, dModel, 2L), torch.sin(position * divTerm)) |> ignore
        encodings.index_copy_(1, torch.arange(1L, dModel, 2L), torch.cos(position * divTerm)) |> ignore
        encodings
        // Remark: Positional encodings are added to token embeddings to incorporate sequence order, complementing the tokenization process.

    /// Transformer Decoder Layer with causal mask and multi-head self-attention
    type private TransformerDecoderLayer(dModel: int64, nHeads: int64, dropoutRate: float32) as self =

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

            let (batch, seq, _) = x.shape |> Array.head, x.shape |> Array.item 1, x.shape |> Array.item 2
            let reshapedShape = [|batch; seq; 3L; nHeads; headDim|]
            use qkv = qkvProjection.forward(x)
            use qkvReshaped = qkv.view(reshapedShape)
            use q = qkvReshaped.select(2, 0L).transpose(1, 2)
            use k = qkvReshaped.select(2, 1L).transpose(1, 2)
            use v = qkvReshaped.select(2, 2L).transpose(1, 2)
            use scores = torch.matmul(q, k.transpose(-2, -1)) / sqrt(float headDim)
            use mask = torch.triu(torch.ones([|seq; seq|], device=scores.device), diagonal=1L).to_type(torch.ScalarType.Bool)
            use maskedScores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), System.Single.NegativeInfinity)
            use attentionWeights = softmax(maskedScores, -1L) |> dropout.forward
            use context = torch.matmul(attentionWeights, v)
            use context = context.transpose(1, 2).contiguous().view(batch, seq, dModel)

            context
            |> outputProjection.forward
            |> fun output -> x + output
            |> layerNorm1.forward
            |> fun output 
                ->
                output
                |> feedForward1.forward
                |> gelu
                |> feedForward2.forward
                |> fun ffOutput -> output + ffOutput
                |> layerNorm2.forward

    // MiniTransformer model with multiple decoder layers
    type private Transformer(vocabSize: int64, dModel: int64, nHeads: int64, numLayers: int) as self =

        inherit Module<torch.Tensor, torch.Tensor>("Transformer")

        //EMBEDDING AND ENCODING TOKEN POSITIONS
        //instantiate an embedding layer 
        let embedding = Embedding(vocabSize, dModel)  //jen priprava, zatim jsou tam jen nahodna cisla
        // Remark: The Embedding layer converts token indices (from tokenization) into dense embedding vectors of size dModel (64).       
        
        let posEnc = getPositionalEncodings 10L dModel   //jen priprava, zatim jsou tam jen nahodna cisla
        // Remark: Positional encodings are added to token embeddings to preserve sequence order, complementing tokenization.

        //tohle neni embedding, ale musi to tady byt quli registrace v self.RegisterComponents(), ktera musi byt tam, kde je 
        let dropout = Dropout(float dropoutRate)
        let decoderLayers = ModuleList<torch.nn.Module<torch.Tensor, torch.Tensor>>(List.init numLayers (fun _ -> new TransformerDecoderLayer(dModel, nHeads, dropoutRate) :> torch.nn.Module<torch.Tensor, torch.Tensor>) |> List.toArray)
        let outputLayer = Linear(dModel, vocabSize)
        let norm = LayerNorm([|dModel|])

        do self.RegisterComponents() //registruje Embedding, DropoutLinear, LayerNorm, takze vyse uvedene parameterless fce nemuzeme dat do logickeho poradi, tj. do forward(x)

        override _.forward(x) =

            let emb = embedding.forward(x) // Convert token indices to embeddings: [batch, seq, dModel]
            // Remark: This line performs the conversion of tokenized input (integer indices) to embedding vectors using the Embedding layer.
            let seqLen = x.shape |> Array.item 1

            //absolute positional embeddings    
            let embWithPos = emb + posEnc.narrow(0L, 0L, seqLen).``to``(x.device) // Add positional encodings
            // Remark: Adds positional encodings to token embeddings to incorporate sequence position information.
            
            //The addition (emb + posEnc) combines token embeddings with positional information, 
            //enabling the Transformer to distinguish token positions (e.g., "Sun" in position 1 vs. position 2)            

            //viz take attention head - dropout zabranuje overfitting (coz moje manualni tokenizace urcite dela) natalie.clanweb.eu //nahodne vynulovani nekterych neuronu
            //pouze pri training, ne pro inference (tam musi byt droput vynulovany
            let embWithPos = dropout.forward(embWithPos)

            //sbaleni vice hidden layers??? TODO zjistit
            let dec = Seq.fold (fun x (layer: torch.nn.Module<torch.Tensor, torch.Tensor>) -> layer.forward(x)) embWithPos decoderLayers

            let normOut = norm.forward(dec)  //zjistit, jestli je to final normalization layer

            outputLayer.forward(normOut).to_type(torch.ScalarType.Float32) // [batch, seq, vocabSize]

    // Training loop for pre-training or fine-tuning with perplexity evaluation
    let private trainEpoch (model: torch.nn.Module<torch.Tensor, torch.Tensor>) (optimizer: torch.optim.Optimizer) (lossFn: CrossEntropyLoss) (input: torch.Tensor) (target: torch.Tensor) maxEpochs phase =
        [ 0 .. maxEpochs - 1 ]
        |> List.iter
            (fun epoch
                ->
                optimizer.zero_grad()
                use output = model.forward(input)
                use loss = lossFn.forward(output.view(-1L, vocabSize), target.view(-1L))
                let perplexity = torch.exp(loss).item<float32>()
                try
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) |> ignore
                with
                | :? System.StackOverflowException as ex
                    ->
                    printfn "StackOverflowException in %s, epoch %d: %s" phase (epoch + 1) (string ex.Message) 
                    Console.ReadLine() |> ignore
                | ex 
                    -> 
                    printfn "%s" (string ex.Message) 
                optimizer.step() |> ignore
                let counter = (+) epoch 1
                match counter % 100 = 0 && counter > 100 with
                | true  -> printfn "%s Epoch %d, Loss: %.4f, Perplexity: %.4f" phase counter (loss.item<float32>()) perplexity
                | false -> ()
                System.GC.Collect()
            )

    // Inference loop with top-k sampling
    let rec private generate (model: torch.nn.Module<torch.Tensor, torch.Tensor>) (inputSeq: torch.Tensor) steps maxSteps acc =

        match steps with
        | s when s >= maxSteps -> acc
        | _ ->
            let _ = torch.no_grad()
            let logits: torch.Tensor = model.forward(inputSeq)
            let lastLogit: torch.Tensor = logits.select(0, 0L).select(0, -1L)
            let adjustedLogit: torch.Tensor =
                match (=) steps 0 with
                | true  -> lastLogit.index_fill_(0, torch.tensor([|7L|], device=lastLogit.device), System.Single.NegativeInfinity)
                | false -> lastLogit
            let temp: float32 = 0.7f
            let struct (probs: torch.Tensor, indices: torch.Tensor) = torch.topk(adjustedLogit / temp, int topK, dim=0)
            let probs: torch.Tensor = softmax(probs, dim=0L)
            let nextTokenIdx: int64 = torch.multinomial(probs, 1).item<int64>()
            let nextToken: int64 = indices.[nextTokenIdx].item<int64>()
            let newAcc: int64 list = nextToken :: acc
            let newInput: torch.Tensor = torch.cat([|inputSeq; torch.tensor([|nextToken|], device=inputSeq.device).unsqueeze(0L)|], dim=1L)
            generate model newInput (steps + 1) maxSteps newAcc

    let internal main () =

        let device = match torch.cuda.is_available() with true -> torch.CUDA | false -> torch.CPU
        printfn "Using device: %A" <| (string device).ToUpper()

        // TOKENIZATION

        //1) Built-in TorchSharp tokenizer  - does not exist :-(
        //2) External tokenizers such as HuggingFace (Python interop is needed)
        //3) Manual tokenization - for learning purposes only
        //4) Custom tokenizer - for learning purposes only

        // MANUAL TOKENIZATION
        (*
        In your provided F# code for the Transformer model, the inputData and targetData are manually created as pre-tokenized sequences of indices (e.g., [0, 1, 2]
        for "The Sun is" and [2, 3, 7] for "is yellow <eos>"), and the target data is manually shifted by one token to align with the autoregressive training objective 
        (predicting the next token). In a real-world scenario, a tokenizer would handle the process of splitting raw text into tokens, mapping them to indices in a vocabulary,
        and preparing input-target pairs, including the shifting for autoregressive models.        
        *)

        // Data preparation: pre-training input and target sequences (320 examples)
        let inputData = //A sequence of tokens up to a certain point (e.g., [0, 1, 2] for "The Sun is").    
            Array2D.init 320 3
                (fun i k 
                    ->
                    match (i, k) with
                    // "The Sun is" -> "is yellow <eos>" (310 examples)
                    | i, k when i < 310 -> [|0L; 1L; 2L|] |> Array.item k
                    // "The sky is" -> "is blue <eos>" (99 examples)
                    | i, k when i >= 310 && i <= 318 -> [|0L; 5L; 2L|] |> Array.item k
                    // "The Sun is" -> "is black <eos>" (1 example)
                    | i, k when i = 319 -> [|0L; 1L; 2L|] |> Array.item k
                    | _ -> failwith "Invalid index"
                )
        // Remark: Data preparation: Creates input sequences as token indices (e.g., "The Sun is" = [0, 1, 2]) based on the vocabulary.
        // This implicitly assumes tokenization has occurred, mapping words to their indices in the vocabulary array.  
        
        let targetData =  //The next token(s) in the sequence (e.g., [2, 3, 7] for "is yellow <eos>"). //tj. jakoby posunute o jeden token (nebot take hledame next token)
            Array2D.init 320 3
                (fun i k
                    ->
                    match (i, k) with
                    // "The Sun is" -> "is yellow <eos>" (310 examples)
                    | i, k when i < 310 -> [|2L; 3L; 7L|] |> Array.item k
                    // "The sky is" -> "is blue <eos>" (99 examples)
                    | i, k when i >= 310 && i <= 318 -> [|2L; 6L; 7L|] |> Array.item k
                    // "The Sun is" -> "is black <eos>" (1 example)
                    | i, k when i = 319 -> [|2L; 4L; 7L|] |> Array.item k
                    | _ -> failwith "Invalid index"
                )
        // Remark: Data preparation: Creates target sequences as token indices (e.g., "is yellow <eos>" = [2, 3, 7]).
        // This represents the expected output tokens, assuming tokenization of the target text.

         //The input and target sequences are typically shifted by one token, as the model predicts the next token for each position.

        // CUSTOM-MADE TOKENIZER
        let dataset = TextData.getSequences()
        // Remark: Uses TextData module to simulate scraped text (300 instances of "The Sun is yellow", 1 instance of "The Sun is black").
        let (inputData, targetData) = Tokenizer.createInputTargetPairs dataset
        
        //Diky shadowing nyni pouzivame inputData a targetData z custom-made tokenizatoru 
        use input = torch.tensor(inputData, device=device) // [32, 3]
        // Remark: Converts tokenized input data (indices) into a tensor for model processing.
        use target = torch.tensor(targetData, device=device) // [32, 3]
        // Remark: Converts tokenized target data (indices) into a tensor for training.

        printfn "Starting pre-training..."        
        
        //Obsahuje embedding, positioning, atd. .....
        use model = (new Transformer(int64 vocabSize, dModel, nHeads, numLayers)).``to``(device)

        use lossFn = new CrossEntropyLoss()
        use optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        model.train()

        trainEpoch model optimizer lossFn input target epochs "Pre-training"
        
        // Fine-tuning: prepare a small dataset to emphasize "the Sun is" -> "is yellow <eos>"
        let fineTuneInputData = Array2D.init 10 3 (fun i k -> [|0L; 1L; 2L|] |> Array.item k) // "the Sun is" for all 10 examples
        // Remark: Data preparation: Creates fine-tuning input sequences as token indices ("the Sun is" = [0, 1, 2]).
        let fineTuneTargetData = Array2D.init 10 3 (fun i k -> [|2L; 3L; 7L|] |> Array.item k) // "is yellow <eos>" for all 10 examples
        // Remark: Data preparation: Creates fine-tuning target sequences as token indices ("is yellow <eos>" = [2, 3, 7]).
        
        use fineTuneInput = torch.tensor(fineTuneInputData, device=device) // [10, 3]
        // Remark: Converts tokenized fine-tuning input data into a tensor.
        use fineTuneTarget = torch.tensor(fineTuneTargetData, device=device) // [10, 3]
        // Remark: Converts tokenized fine-tuning target data into a tensor.

        printfn "Starting fine-tuning..."
        use fineTuneOptimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        model.train()
        
        trainEpoch model fineTuneOptimizer lossFn fineTuneInput fineTuneTarget fineTuneEpochs "Fine-tuning"

        printfn "Generating sequence after fine-tuning..."

        //INFERENCE
        model.eval()
        use inputSeq = torch.tensor([|0L; 1L; 2L|], device=device).unsqueeze(0L) // [1, 3]
        // Remark: Data preparation: Creates an input sequence for inference ("The Sun is" = [0, 1, 2]) as token indices.
        printf "Generated sequence (token IDs): "
        let generated = generate model inputSeq 0 2 [] |> List.rev // Generate 2 tokens
        generated |> List.iter (printf "%d ")
        printfn "\n"
        printf "Generated sequence (words): "
        generated |> List.iter (fun id -> printf "%s " (vocabulary |> Array.item (int id)))
        // Remark: Converts generated token indices back to words using the vocabulary, reversing the tokenization process for human-readable output.
        printfn "\n"