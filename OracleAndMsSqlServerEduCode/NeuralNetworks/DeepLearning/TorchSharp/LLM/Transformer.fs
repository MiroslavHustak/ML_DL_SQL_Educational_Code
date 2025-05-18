namespace NeuralNetworks

open System

open TorchSharp
open TorchSharp.Modules

open type torch.nn
open type torch.nn.functional

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

    //TODO!!! zrobit record a presunout do dat
    // Vocabulary and hyperparameters
    // Remark: Defines the vocabulary for tokenization, mapping words to indices (e.g., "The" = 0, "Sun" = 1, ..., "<eos>" = 7).
    let private vocabulary = [|"The"; "Sun"; "is"; "yellow"; "black"; "sky"; "blue"; "<eos>"|]   //pouze pro indexizaci
    let private mv = vocabulary |> Array.mapi (fun i _ -> i) //tady zbytecne, ale aby to formalne bylo jako u realnych modelu
    let private vocabSize = vocabulary.Length
    
    //TODO!!! zrobit record a presunout do dat
    let [<Literal>] private dModel = 72L //embeddings of size 72
    let [<Literal>] private epochs = 14000
    let [<Literal>] private fineTuneEpochs = 2000 //max new tokens
    let private batch = 32L
    let [<Literal>] private fineTuneBatch = 10L
    let [<Literal>] private nHeads = 12L  //AttentionHeadCount v prednasce
    let [<Literal>] private numLayers = 2
    let [<Literal>] private dropoutRate = 0.1f
    let [<Literal>] private topK = 3L
    let [<Literal>] private contextSize = 1024

    //This function generates sinusoidal non-learnable positional encodings for token positions
    let private getPositionalEncodings (seqLen: int64) (dModel: int64) : torch.Tensor =

        let position = torch.arange(seqLen, dtype = torch.float32).unsqueeze(1)
        //.unsqueeze(1) adds a new dimension at index 1 a misto rozmeru [|5|] zrobi [|5,  1|], viz take educational code
        
        let divTerm = torch.exp(torch.arange(0L, dModel, 2L, dtype = torch.float32) * (- Math.Log 10000.0 / float dModel))
        //Each element of the matrix is independently transformed as e^x
        
        let encodings = torch.zeros([|seqLen; dModel|])
        encodings.index_copy_(1, torch.arange(0L, dModel, 2L), torch.sin(position * divTerm)) |> ignore
        encodings.index_copy_(1, torch.arange(1L, dModel, 2L), torch.cos(position * divTerm)) |> ignore
        encodings
        // Remark: Positional encodings are added to token embeddings to incorporate sequence order, complementing the tokenization process.

    // Transformer Decoder Layer with causal mask and multi-head self-attention
    // cca to odpovida pojmu Transformer Block v prednasce Tomáše Hercega
    type private TransformerDecoderLayer(dModel: int64, nHeads: int64, dropoutRate: float32) as self =

        inherit Module<torch.Tensor, torch.Tensor>("TransformerDecoderLayer")
        
        // Step 1: Input Embedding (assumed to be handled before this layer)
        // The input `x` is expected to be embedded tokens of shape [batch, seq, dModel].
        
        let qkvProjection = Linear (dModel, dModel * 3L) // Step 2: Projects input to Q, K, V vectors simultaneously. //qkvProjection is a single linear layer that transforms the input tensor x into concatenated Q, K, and V
        let outputProjection = Linear (dModel, dModel)  // Step 9: Final linear transformation after multi-head attention.
        let feedForward1 = Linear (dModel, dModel * 4L) // Feed-forward network (not part of attention flow, but part of transformer layer).
        let feedForward2 = Linear (dModel * 4L, dModel) // Feed-forward network second layer.
        //The factor of 4 in the F# code (and C# code in the lecture) sets the hidden dimension of the FFN to 4×dmodel, a standard convention in Transformer models to increase capacity. It’s hardcoded for simplicity but could be made configurable.
                
        let layerNorm1 = LayerNorm [|dModel|]        // Normalization after attention (standard in transformers).
        let layerNorm2 = LayerNorm [|dModel|]        // Normalization after feed-forward network.
        let dropout = Dropout (float dropoutRate)      // Step 6: Dropout for regularization, applied to attention weights.

        //Quli registrace Linear, Dropout, LayerNorm v self.RegisterComponents() vyse uvedene parameterless fce nemuzeme dat do logickeho poradi, tj. do forward x
    
        do self.RegisterComponents()
    
        override _.forward x =

            // Input `x` is [batch, seq, dModel], where seq is sequence length, dModel is embedding dimension.
            let (batch, seq, _) = x.shape |> Array.head, x.shape |> Array.item 1, x.shape |> Array.item 2
            
            // Improvement: Ensure dModel is divisible by nHeads to avoid invalid headDim.
            match dModel % nHeads <> 0L with
            | true  -> failwithf "dModel (%d) must be divisible by nHeads (%d) to compute headDim." dModel nHeads
            | false -> ()

            let headDim = dModel / nHeads // Compute dimension per head for multi-head attention.

            //Tady se to lisi od prednasky, moje verze je udajne vypocetne ucinnejsi
            //The qkvReshaped variable in the F# code is used because it employs a single Linear layer to compute Q, K, and V together, requiring an intermediate reshaping step to split the concatenated tensor into three parts for multi-head attention. 
            let reshapedShape = [|batch; seq; 3L; nHeads; headDim|] // Shape for splitting Q, K, V across heads.
    
            // Step 2: Projecting into Q, K, V vectors.
            use qkv = qkvProjection.forward x // Applies linear transformation to get combined Q, K, V: [batch, seq, dModel*3].
    
            // Step 8 (partial): Reshape for multi-head attention.
            use qkvReshaped = qkv.view reshapedShape // Reshapes to [batch, seq, 3, nHeads, headDim] for splitting Q, K, V.
    
            // Step 2 (continued): Extract Q, K, V for each head.
            use q = qkvReshaped.select(2, 0L).transpose(1, 2) // Q: [batch, nHeads, seq, headDim].
            use k = qkvReshaped.select(2, 1L).transpose(1, 2) // K: [batch, nHeads, seq, headDim].
            use v = qkvReshaped.select(2, 2L).transpose(1, 2) // V: [batch, nHeads, seq, headDim].
    
            // Step 3: Computing attention scores.
            use attentionScores = torch.matmul(q, k.transpose(-2, -1)) / sqrt(float headDim) // QK^T / sqrt(d_k): [batch, nHeads, seq, seq].
    
            // Step 4: Causal masking for auto-regressive tasks. // This prevents attending to future tokens during training.
            use mask = torch.triu(torch.ones([|seq; seq|], device=attentionScores.device), diagonal=1L).to_type(torch.ScalarType.Bool) // Creates upper triangular mask.
            use maskedScores = attentionScores.masked_fill(mask.unsqueeze(0).unsqueeze(0), System.Single.NegativeInfinity) // Applies mask to prevent attending to future tokens.
            //vsimni si nahrazeni nul za minus nekonecno
    
            // Step 5: Applying softmax for normalization.
            // Step 6: Applying dropout to attention weights.
            use attentionWeights = softmax(maskedScores, -1L) |> dropout.forward // Softmax over last dimension, then apply dropout: [batch, nHeads, seq, seq].
    
            // Step 7: Calculating context vectors. //tj. co je ve vete dulezite
            use contextVector = torch.matmul(attentionWeights, v) // Attention weights * V: [batch, nHeads, seq, headDim].
    
            // Step 8: Concatenate multi-head outputs and reshape. //tech 12 "sloupcu" je zase slouceno dohromady, 
            use contextVector = contextVector.transpose(1, 2).contiguous().view(batch, seq, dModel) // Transpose to [batch, seq, nHeads, headDim], then reshape to [batch, seq, dModel].
    
            // Step 9: Final linear transformation.
            // NORMALIZACE A AKTIVACNI FUNKCE
            contextVector
            |> outputProjection.forward // Applies final linear layer: [batch, seq, dModel].
            |> fun output -> x + output // Residual connection: adds input to output.
            |> layerNorm1.forward // Applies layer normalization after attention.
            |> fun output
                ->
                output
                |> feedForward1.forward // Feed-forward network (first layer).
                |> gelu // Applies GELU activation.
                |> feedForward2.forward // Feed-forward network (second layer).
                |> dropout.forward 
                |> fun ffOutput -> output + ffOutput // Residual connection for feed-forward.
                |> layerNorm2.forward // Final layer normalization.  //Normalize after feed-forward
            
            //*********************************************************************************
            // Viz poznámka 1 (Notes.txt) - rozdíly oproti přednášce Tomáše Hercega
            //*********************************************************************************

    //nazev Transformer zde neodpovida pojmu TRANSFORMER BLOCK ci TRANSFORMER z prednasky             
    type private Transformer(vocabSize: int64, dModel: int64, nHeads: int64, numLayers: int) as self =

        inherit Module<torch.Tensor, torch.Tensor>("Transformer")

        //EMBEDDING AND ENCODING TOKEN POSITIONS
        //instantiate an embedding layer 
        let embedding = Embedding (vocabSize, dModel)  //jen priprava, zatim jsou tam jen nahodna cisla
        // Remark: The Embedding layer converts token indices (from tokenization) into dense embedding vectors of size dModel (64).       
        
        let positionEmbedding = Embedding (1024L, dModel)
        let posEnc = getPositionalEncodings 1024L dModel   //jen priprava, zatim jsou tam jen nahodna cisla
        // Remark: Positional encodings are added to token embeddings to preserve sequence order, complementing tokenization.

        //tohle neni primo embedding, ale musi to tady byt quli registrace v self.RegisterComponents(), ktera musi byt tam, kde je 
        let dropout = Dropout (float dropoutRate)
        
        let decoderLayers = 
            let createLayer _ = new TransformerDecoderLayer(dModel, nHeads, dropoutRate) :> torch.nn.Module<torch.Tensor, torch.Tensor>
            let decoderLayersList = List.init numLayers createLayer
            let decoderLayersArray = List.toArray decoderLayersList        
            ModuleList<torch.nn.Module<torch.Tensor, torch.Tensor>>(decoderLayersArray)
        
        let outputLayer = Linear (dModel, vocabSize)
        let norm = LayerNorm [|dModel|]

        do self.RegisterComponents() //registruje Embedding, DropoutLinear, LayerNorm, takze vyse uvedene parameterless fce nemuzeme dat do logickeho poradi, tj. do forward x

        override _.forward x =

            let emb = embedding.forward x // Convert token indices to embeddings: [batch, seq, dModel]
            // Remark: This line performs the conversion of tokenized input (integer indices) to embedding vectors using the Embedding layer.
            let seqLen = x.shape |> Array.item 1

            //*********************************************************************************
            // Viz poznámka 2 (Notes.txt) - rozdíly oproti přednášce Tomáše Hercega
            //*********************************************************************************

            //absolute positional embeddings    
            let embWithPos = emb + posEnc.narrow(0L, 0L, seqLen).``to``(x.device) // Add positional encodings //soucet tensoru
            // Remark: Adds positional encodings to token embeddings to incorporate sequence position information.
            
            //The addition (emb + posEnc) combines token embeddings with positional information, 
            //enabling the Transformer to distinguish token positions (e.g., "Sun" in position 1 vs. position 2)            

            //viz take attention head - dropout zabranuje overfitting (coz moje manualni tokenizace urcite dela) natalie.clanweb.eu //nahodne vynulovani nekterych neuronu
            //pouze pri training, ne pro inference (tam musi byt droput vynulovany)
            let embWithPos = dropout.forward embWithPos

            //applying a sequence of transformer decoder layers to the input embWithPos     
            //multiple transformer layers are stacked (e.g., numLayers in F# or config.LayersCount in C#). This stacking is similar to having multiple hidden layers in a deep neural network, where each layer learns increasingly complex representations.
            let decoderLayersOutput = Seq.fold (fun x (layer: torch.nn.Module<torch.Tensor, torch.Tensor>) -> layer.forward(x)) embWithPos decoderLayers
            //The decoderLayers in F# and transformersBlock in C# represent a stack of transformer layers, which collectively act as the hidden layers of the transformer model.

            //layer normalization on the output of the transformer layers            
            let normOut = norm.forward decoderLayersOutput           
            
            outputLayer.forward(normOut).to_type(torch.ScalarType.Float32) // [batch, seq, vocabSize]

    // Training loop for pre-training or fine-tuning with perplexity evaluation
    let private trainEpoch (model: torch.nn.Module<torch.Tensor, torch.Tensor>) (optimizer: torch.optim.Optimizer) (lossFn: CrossEntropyLoss) (input: torch.Tensor) (target: torch.Tensor) maxEpochs phase =
       
        [ 0 .. maxEpochs - 1 ]
        |> List.iter
            (fun epoch
                ->
                optimizer.zero_grad()  //Resets the gradients of all parameters managed by the optimizer to zero.
                
                use output = model.forward input
                use loss = lossFn.forward(output.view(-1L, vocabSize), target.view(-1L))
                let perplexity = torch.exp(loss).item<float32>()
                
                //**************************************************
                // Tady se v pozadi meni stav modelu
                try
                    loss.backward() //meni model
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) |> ignore
                with
                | :? System.StackOverflowException as ex
                    ->
                    printfn "StackOverflowException in %s, epoch %d: %s" phase (epoch + 1) (string ex.Message) 
                    Console.ReadLine() |> ignore
                | ex 
                    -> 
                    printfn "%s" (string ex.Message) 
               
                optimizer.step() |> ignore  //meni model
                //**************************************************
                
                let counter = (+) epoch 1
                match counter % 100 = 0 && counter > 100 with
                | true  -> printfn "%s Epoch %d, Loss: %.4f, Perplexity: %.4f" phase counter (loss.item<float32>()) perplexity
                | false -> ()

                System.GC.Collect()
            )
        
    // INFERENCE
    // Inference loop with top-k sampling and early stopping

    //*********************************************************************************
    // Viz poznámka 3 (Notes.txt) - kód v C# odpovídající metodě GenerateText v přednášce Tomáše Hercega a mé funkci generate v F#
    //*********************************************************************************

    let rec [<TailCall>] private generate (model: torch.nn.Module<torch.Tensor, torch.Tensor>) (inputSeq: torch.Tensor) steps maxSteps acc contextSize (temp: float32) (topK: int64) (strategy: string) =
    // The generate function contains multinomial sampling, temperature, and stop token handling 
        
        let trimInput (input: torch.Tensor) (contextSize: int) =

            match input.shape.[1] > contextSize with
            | true  -> input.narrow(1, -contextSize, contextSize) // Keep only the last contextSize tokens
            | false -> input
        // Remark: Ensures the input sequence does not exceed the model's context size, preventing memory issues and compatibility with positional encodings.

        // Helper function to sample the next token from logits
        // Viz také přednáška od Tomáše Hercega
        let sampleLogits (logits: torch.Tensor) (temp: float32) (topK: int64) (strategy: string) (vocabSize: int64) =

            // Ensure topK does not exceed vocabulary size to prevent invalid indexing
            let effectiveTopK = min topK vocabSize
            
            match effectiveTopK <= 0L with
            | true  -> failwith "topK must be positive and not exceed vocabulary size"
            | false -> ()
            
            match strategy with
            | "top-k" 
                ->
                // Select top-k logits and their indices
                let struct (probs, indices) = torch.topk(logits / temp, int effectiveTopK, dim=0)
                let probs = softmax(probs, dim=0L)
                // Remark: Multinomial sampling converts top-k probabilities into a single token index, introducing controlled randomness in token selection.

                //If probs = tensor([0.1, 0.7, 0.1, 0.1]), then idx will be 1 most of the time (because 0.7 probability), but it can also be 0, 2, or 3 occasionally.
                let idx = torch.multinomial(probs, 1).item<int64>() //It randomly picks one category index from probs
                indices.[idx].item<int64>()
            | "greedy" 
                ->
                // Select the token with the highest logit
                torch.argmax(logits, dim=0).item<int64>()
            | _ 
                ->
                failwith $"Unsupported sampling strategy: {strategy}"
        // Remark: Temperature is applied by scaling logits (logits / temp) before top-k sampling or greedy selection, controlling the randomness of the output (lower temp = less random, higher temp = more random).

        match steps with
        | s
            when s >= maxSteps
            ->
            acc // Stop recursion if the maximum number of steps is reached
        | _ 
            ->
            // Components:
            let _ = torch.no_grad() // Disable gradient computation to save memory and computation during inference
            
            let adjustedLogit: torch.Tensor =
                let trimmedInput = trimInput inputSeq contextSize  // Trim the input sequence to the model's context size to prevent exceeding the maximum sequence length
                let logits: torch.Tensor = model.forward trimmedInput // Run the forward pass to get the model's predictions (logits) for the input sequence; shape: [batch=1, seqLen, vocabSize]
                let lastLogit: torch.Tensor = logits.select(0, 0L).select(0, -1L) // Extract the logits for the last token in the sequence; shape: [vocabSize]
                
                match steps with
                | 0 -> lastLogit.index_fill_(0, torch.tensor([|7L|], device=lastLogit.device), System.Single.NegativeInfinity) // On the first step, set the logit for the <eos> token (ID 7) to negative infinity to prevent generating <eos> as the first token
                | _ -> lastLogit // Use the logits as-is for subsequent steps
            
            // Remark: Stop token handling prevents <eos> (ID 7) from being generated as the first token, ensuring meaningful sequence continuation.
            let nextToken: int64 = sampleLogits adjustedLogit temp topK strategy (int64 vocabSize) // Sample the next token using the specified strategy
            let newAcc: int64 list = nextToken :: acc // Accumulate the generated token by prepending it to the accumulator list
            // Remark: Stop token handling also checks for <eos> (ID 7) to stop generation early, mimicking natural sequence termination.
            
            match nextToken with
            | 7L 
                ->
                newAcc // Stop if <eos> is generated
            | _ ->
                let newInput: torch.Tensor = torch.cat([|inputSeq; torch.tensor([|nextToken|], device=inputSeq.device).unsqueeze(0L)|], dim=1L) // Append the new token to the input sequence for the next iteration; shape: [batch=1, seqLen+1]
                generate model newInput (steps + 1) maxSteps newAcc contextSize temp topK strategy // Recursively call generate with the updated input, step count, and accumulator    
    
    let internal main () =     

        let device = match torch.cuda.is_available() with true -> torch.CUDA | false -> torch.CPU
        printfn "Using device: %A" <| (string device).ToUpper()

        // TOKENIZATION        
        // External tokenizers such as HuggingFace (Python interop is needed) shall be normally used for code used in production.

        // MANUAL TOKENIZATION
       
        // TODO presunout do dat
        // Data preparation: pre-training input and target sequences (320 examples)
        let inputData = //A sequence of tokens up to a certain point (e.g., [0, 1, 2] for "The Sun is").    
            Array2D.init 320 3
                (fun i k 
                    ->
                    match (i, k) with
                    | i, k when i < 310 -> [|0L; 1L; 2L|] |> Array.item k
                    | i, k when i >= 310 && i <= 318 -> [|0L; 5L; 2L|] |> Array.item k
                    | i, k when i = 319 -> [|0L; 1L; 2L|] |> Array.item k
                    | _ -> failwith "Invalid index"
                )
         //TODO presunout do dat   
        // Remark: Data preparation: Creates input sequences as token indices (e.g., "The Sun is" = [0, 1, 2]) based on the vocabulary.
        // This implicitly assumes tokenization has occurred, mapping words to their indices in the vocabulary array.  
        
        let targetData =  //The next token(s) in the sequence (e.g., [2, 3, 7] for "is yellow <eos>"). //tj. jakoby posunute o jeden token (nebot take hledame next token)
            Array2D.init 320 3
                (fun i k
                    ->
                    match (i, k) with
                    | i, k when i < 310 -> [|2L; 3L; 7L|] |> Array.item k
                    | i, k when i >= 310 && i <= 318 -> [|2L; 6L; 7L|] |> Array.item k
                    | i, k when i = 319 -> [|2L; 4L; 7L|] |> Array.item k
                    | _ -> failwith "Invalid index"
                )
        // Remark: Data preparation: Creates target sequences as token indices (e.g., "is yellow <eos>" = [2, 3, 7]).
        // This represents the expected output tokens, assuming tokenization of the target text.

         //The input and target sequences are typically shifted by one token, as the model predicts the next token for each position.

        // CUSTOM-MADE TOKENIZER or TIKTOK TOKENIZER
        let dataset = TextData.getSequences()
        // Remark: Uses TextData module to simulate scraped text (300 instances of "The Sun is yellow", 1 instance of "The Sun is black").
        
        //Diky shadowing nyni pouzivame inputData a targetData bud z custom-made tokenizatoru nebo z TikTokTokenizeru
        let (inputData, targetData) = Tokenizer.createInputTargetPairs dataset 
        //let (inputData, targetData) = TikTokTokenizer.createInputTargetPairs dataset //TikTokTokenizer //nelze quli male delky     
        
        use input = torch.tensor(inputData, device=device) // [32, 3]
        // Remark: Converts tokenized input data (indices) into a tensor for model processing.
        use target = torch.tensor(targetData, device=device) // [32, 3]
        // Remark: Converts tokenized target data (indices) into a tensor for training.

        printfn "Starting pre-training..."        
              
        use model = (new Transformer(int64 vocabSize, dModel, nHeads, numLayers)).``to``(device)

        use lossFn = new CrossEntropyLoss() //viz přednáška Tomáše Hercega
        use optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) //Adam is the optimizer that updates the model’s parameters (weights) to minimize the loss computed by CrossEntropyLoss
        //Adam (Adaptive Moment Estimation)       

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
        model.eval()  // When model.eval() is called, the Dropout module’s forward method does not apply dropout (i.e., it effectively sets the dropout rate to zero for that pass).
        //This ensures that the Dropout modules in both the TransformerDecoderLayer (applied to attention weights) and the Transformer (applied to embeddings with positional encodings) do not drop any activations, making the model’s output deterministic during inference.
                    
        use inputSeq = torch.tensor([|0L; 1L; 2L|], device=device).unsqueeze 0L // [1, 3]
        // Remark: Data preparation: Creates an input sequence for inference ("The Sun is" = [0, 1, 2]) as token indices.
        
        printf "Generated sequence (token IDs): "
        let generated = generate model inputSeq 0 2 [] contextSize 0.7f 3L "top-k" |> List.rev // Generate 2 tokens with temp=0.7, topK=3, top-k sampling
        generated |> List.iter (printf "%d ")
        printfn "\n"
        
        printf "Generated sequence (words): "
        generated |> List.iter (fun id -> printf "%s " (vocabulary |> Array.item (int id)))
        // Remark: Converts generated token indices back to words using the vocabulary, reversing the tokenization process for human-readable output.
        printfn "\n"