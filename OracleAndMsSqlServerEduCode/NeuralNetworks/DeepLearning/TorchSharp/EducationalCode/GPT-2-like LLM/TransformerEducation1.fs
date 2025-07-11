namespace NeuralNetworks2

open System

open TorchSharp
open TorchSharp.Modules

open type torch.nn
open type torch.nn.functional

module Transformer_TorchSharpEducation =

    // Vocabulary and hyperparameters
    // Remark: Defines the vocabulary for tokenization, mapping words to indices (e.g., "The" = 0, "Sun" = 1, ..., "<eos>" = 7).
    let private vocabulary = [ "The"; "Sun"; "is"; "yellow"; "black"; "sky"; "blue"; "<eos>" ]   //pouze pro indexizaci
    let private mv = vocabulary |> List.mapi (fun i _ -> i) //tady zbytecne, ale aby to formalne bylo jako u realnych modelu
    let private vocabSize = vocabulary |> List.length
    
    //TODO!!! zrobit record a presunout do dat
    let [<Literal>] private dModel = 72L //embeddings of size 72
    let [<Literal>] private epochs = 20000
    let [<Literal>] private fineTuneEpochs = 2000 //max new tokens
    let private batch = 32L
    let [<Literal>] private fineTuneBatch = 10L
    let [<Literal>] private nHeads = 12L  //AttentionHeadCount v prednasce, taky jich ma 12
    let [<Literal>] private numLayers = 2
    let [<Literal>] private dropoutRate = 0.1f
    let [<Literal>] internal temp = 1.0f
    (*
    Temperature Scaling: A method to control randomness in text generation.
    Lower temperature (e.g. 0.2) → more predictable, focused output.
    Higher temperature (e.g. 1.0+) → more random and creative output.
    *)
    let [<Literal>] private topK = 3L //Limits the model to choosing from only the top k most likely next words.
    let [<Literal>] private contextSize = 1024
    let [<Literal>] private learningRate = 0.001    

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

    type private TransformerDecoderLayer(dModel: int64, nHeads: int64, dropoutRate: float32) as self =

        //MULTI-HEAD ATTENTION

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
        let dropout = Dropout (float dropoutRate)    // Step 6: Dropout for regularization, applied to attention weights.

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
            //transpose je pro prohozeni heads, aby se dostaly "dopredu" - prohozeni dimenzi 1 a 2
            use q = qkvReshaped.select(2, 0L).transpose(1, 2) // Q: [batch, nHeads, seq, headDim]. //pro kazdy token najde, co je "nejdulezitejsi"
            use k = qkvReshaped.select(2, 1L).transpose(1, 2) // K: [batch, nHeads, seq, headDim]. //pro kazdy token najde svoji vlastni dulezitost
            use v = qkvReshaped.select(2, 2L).transpose(1, 2) // V: [batch, nHeads, seq, headDim].
    
            // Step 3: Computing attention scores. //torch.matmul contains scalar multiplications  
            //tam,kde je skalarni soucin nejvyssi, tak ty tokeny mohou spolu souviset
            //transpose abych srovnal rozmery matic a mohl je nasobit
            use attentionScores = torch.matmul(q, k.transpose(-2, -1)) / sqrt(float headDim) // QK^T / sqrt(d_k): [batch, nHeads, seq, seq].
    
            // Step 4: Causal masking for auto-regressive tasks. // This prevents attending to future tokens during training.
            use mask = torch.triu(torch.ones([|seq; seq|], device=attentionScores.device), diagonal=1L).to_type(torch.ScalarType.Bool) // Creates upper triangular mask.
            use maskedScores = attentionScores.masked_fill(mask.unsqueeze(0).unsqueeze(0), System.Single.NegativeInfinity) // Applies mask to prevent attending to future tokens.
            //slozitejsi nez v prednasce 
            //torch.ones vytvori tensor vyplneny 1. 
            //torch.triu extracts the upper triangular part of the matrix, starting above the main diagonal. Elements in the upper triangle remain 1, while elements on or below the diagonal are set to 0.
            //.to(dtype=torch.bool) converts the matrix to a boolean type, where 1 becomes True (future token and should not be attended to) and 0 becomes False (can be attended to (current or past token)
            //maskedScores zajisti nahradue True za minus nekonecno    
            //nize uvedeny softmax z minus nekonecno zrobi nuly.
   
            // Step 5: Applying softmax for normalization. - kdyz vyhodime cca pulku matice (ci tenzoru), musi se opet srovnat zbyvajici hodnoty (normalizace), z -nekonecno budou nuly, jinde tak, aby soucet prvku byl 1
            // Step 6: Applying dropout to attention weights. - opet jeste neco vylifruju
            use attentionWeights = softmax(maskedScores, -1L) |> dropout.forward // Softmax over last dimension, then apply dropout: [batch, nHeads, seq, seq].
    
            // Step 7: Calculating context vectors. //tj. co je ve vete dulezite
            //tady konecne pouzijeme v - value vektor
            use contextVector = torch.matmul(attentionWeights, v) // Attention weights * V: [batch, nHeads, seq, headDim]. 
    
            // Step 8: Concatenate multi-head outputs and reshape. //tech 12 heads (jakoby 12 "myslenek") je zase slouceno dohromady 
            use contextVector = contextVector.transpose(1, 2).contiguous().view(batch, seq, dModel) // Transpose to [batch, seq, nHeads, headDim], then reshape to [batch, seq, dModel].
    
            // Step 9: Final linear transformation.   
            //contextualized embeddings (output of layerNorm2.forward in the code)             
            contextVector
            |> outputProjection.forward // Applies final linear layer: [batch, seq, dModel].
            |> fun output -> x + output // Residual connection - to je ta shortcuts v prednasce,  adds input to attention output, urychli backpropagation
            |> layerNorm1.forward // Applies layer normalization after attention. //NORMALIZACE (aby nebyly problemy se stabilitou)
            |> fun output
                ->
                output
                |> feedForward1.forward // Feed-forward network (first layer).
                |> gelu // Applies GELU activation. //AKTIVACNI FUNKCE
                |> feedForward2.forward // Feed-forward network (second layer).
                |> dropout.forward 
                |> fun ffOutput -> output + ffOutput // Residual connection for FFN - to je ta shortcuts v prednasce, urychli backpropagation
                |> layerNorm2.forward // Final layer normalization.  //Normalize after feed-forward

            (*
            Transformer decoder layers (or blocks) consist of a multi-head attention mechanism followed by a feed-forward neural network (FFN), 
            with residual connections and layer normalization, as seen in your TransformerDecoderLayer class. 
            These are often called "hidden layers" in the context of neural networks because they process intermediate representations between the input and output.   
            
            feedForward1 and feedForward2 represent the two linear layers that make up the position-wise feed-forward neural network (FFN) within each transformer decoder layer.
            
            The model has numLayers = 2, meaning it has two transformer decoder layers, each containing:
            A multi-head attention mechanism (with qkvProjection and outputProjection).
            
            A feed-forward neural network (FFN) with feedForward1 and feedForward2.
            
            Key Point: Each decoder layer has its own pair of feedForward1 and feedForward2 instances. So, with numLayers = 2, you actually have:
            Two feedForward1 layers (one per decoder layer, each with weights [288, 72] and bias [288]).
            
            Two feedForward2 layers (one per decoder layer, each with weights [72, 288] and bias [72]).
            
            The number of FFN layers (two per decoder layer) is fixed by the transformer architecture, not by the number of decoder layers. 
            Even if you had 1, 12, or 96 decoder layers (like GPT-2 or GPT-3), each would still have exactly one FFN with two linear layers.
            *)  
            
            //*********************************************************************************
            // Viz poznámka 1 (Notes.txt) - rozdíly oproti přednášce Tomáše Hercega
            //*********************************************************************************

    type private Transformer(vocabSize: int64, dModel: int64, nHeads: int64, numLayers: int) as self =

        inherit Module<torch.Tensor, torch.Tensor>("Transformer")

        //EMBEDDING AND ENCODING TOKEN POSITIONS
        //instantiate an embedding layer 
        let embedding = Embedding (vocabSize, dModel)  //jen priprava, zatim jsou tam jen nahodna cisla //v prednasce 50257 x 768, ja mam pouze 8 x 72 
        // Remark: The Embedding layer converts token indices (from tokenization) into dense embedding vectors of size dModel (72).       
        
        let posEnc = getPositionalEncodings 1024L dModel   //jen priprava, zatim jsou tam jen nahodna cisla
        // Remark: Positional encodings are added to token embeddings to preserve sequence order, complementing tokenization.

        //tohle neni primo embedding, ale musi to tady byt quli registrace v self.RegisterComponents(), ktera musi byt tam, kde je 
        let dropout = Dropout (float dropoutRate)
        
        let decoderLayers = 
            let createLayer _ = new TransformerDecoderLayer(dModel, nHeads, dropoutRate) :> torch.nn.Module<torch.Tensor, torch.Tensor>
            let decoderLayersList = List.init numLayers createLayer
            let decoderLayersArray = List.toArray decoderLayersList        
            ModuleList<torch.nn.Module<torch.Tensor, torch.Tensor>>(decoderLayersArray)
        
        let outputLayer = Linear (dModel, vocabSize) //Initialization of W
        (*
        Example initial W:        
        w_0 ("The")    = [0.01, -0.02, 0.03, 0.01]
        w_1 ("Sun")    = [-0.01, 0.02, -0.01, -0.03]
        w_2 ("is")     = [0.02, -0.01, 0.01, 0.02]
        w_3 ("yellow") = [-0.03, 0.01, -0.02, 0.01]
        w_4 ("<eos>")  = [0.00, 0.02, -0.01, -0.02]   
        *)
        
        let norm = LayerNorm [|dModel|]

        do self.RegisterComponents() //registruje Embedding, DropoutLinear, LayerNorm, takze vyse uvedene parameterless fce nemuzeme dat do logickeho poradi, tj. do forward x

        override _.forward x =

            let emb = embedding.forward x // Convert token indices to embeddings: [batch, seq, dModel]
            // Remark: This line performs the conversion of tokenized input (integer indices) to embedding vectors using the Embedding layer.
            (*
            x: This is the input tensor containing token IDs, typically with shape [batch, seq], where:
            batch is the batch size (number of sequences being processed).
            
            seq is the sequence length (number of tokens in each sequence).
            
            For example, if x contains your token IDs [40134, 2052, 133, 389, 12] for a single sequence, the shape might be [1, 5] (1 batch, 5 tokens).  
            
            Token ID 40134 → Row 40134 of the embedding matrix → A vector of size embedding_dim (e.g., [0.12, -0.34, ..., 0.56]) - hodnoty vektoru dale menene pomoci position encodings, dropout, normalisation.           
            The embedding vector itself (e.g., [0.12, -0.34, ..., 0.56]) is the representation of the token, with 768 values that encode its semantic or syntactic properties.
            These values are learned during training (or pre-training, in the case of models like BERT) to capture the token’s semantic (meaning-related) and syntactic (grammar-related) properties.
            The values are coordinates in a high-dimensional space
            *)
           
            let seqLen = x.shape |> Array.item 1

            //*********************************************************************************
            // Viz poznámka 2 (Notes.txt) - rozdíly oproti přednášce Tomáše Hercega
            //*********************************************************************************

            //absolute positional embeddings
            //mam to slozitejsi, nez v prednasce
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

            //layer normalization on the output of the transformer layers //uplne posledni embeddings  (not trained yet)         
            let normOut = norm.forward decoderLayersOutput           
            
            //output is a tensor of shape [batchSize; seqLen; vocabSize] //values are logits (unnormalized scores) in float32 format
            //output is ready to compute a loss for training or predict the next token during inference.
            outputLayer.forward(normOut).to_type(torch.ScalarType.Float32) // [batch, seq, vocabSize]

    // Training loop for pre-training or fine-tuning with perplexity evaluation
    let private trainEpoch (model: torch.nn.Module<torch.Tensor, torch.Tensor>) (optimizer: torch.optim.Optimizer) (lossFn: CrossEntropyLoss) (input: torch.Tensor) (target: torch.Tensor) maxEpochs phase =
       
        [ 0 .. maxEpochs - 1 ]
        |> List.iter
            (fun epoch
                ->
                optimizer.zero_grad()  //Resets the gradients of all parameters managed by the optimizer to zero.
                
                // Forward pass: Compute logits
                // - Uses W and b from outputLayer, learned from previous epochs
                use output = model.forward input
                // Compute loss: Compare logits to target tokens
                //You must compute the loss first, because without the loss tensor, backward() has nothing to differentiate.
                use loss = lossFn.forward(output.view(-1L, vocabSize), target.view(-1L))
                
                (*
                Training Loop:
                The loop (for epoch in 0 .. epochs - 1) updates (not calculates from scratch) the embedding weights and W using gradient descent. It starts with random floats and refines them to predict correct next tokens (e.g., “yellow” after “is”).

                Loss Function:
                The loss function (CrossEntropyLoss) compares the logits to target token IDs (e.g., ID 3 for “yellow”), not previous embeddings or weights. It penalizes incorrect predictions, guiding updates to W and embeddings.

                Minimal Loss:
                As the loss decreases over epochs, W and embeddings align so that logits = emb @ W^T + b gives the highest logit for the correct next token (e.g., logits[3] = 3.0 for “yellow”), selected via argmax.
                *)

                let perplexity = torch.exp(loss).item<float32>()
                
                //**************************************************
                // Tady se v pozadi meni stav modelu
                try
                    //The lecturer’s mention of "shortcuts" likely refers to residual connections (also called skip connections). 
                    //These connections speed up and stabilize backpropagation by allowing gradients to flow more directly through the network, addressing issues like vanishing or exploding gradients. 
                    // Backward pass: Compute gradients for W, b, and embedding weights
                    loss.backward() //meni model // //This triggers backpropagation, computing gradients through the entire model, including the residual connections.
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) |> ignore // limits the combined size (norm) of all parameter gradients to 1.0, preventing exploding gradients
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

    //soucast inference
    let rec [<TailCall>] private generate (model: torch.nn.Module<torch.Tensor, torch.Tensor>) (inputSeq: torch.Tensor) steps maxSteps acc contextSize (temp: float32) (topK: int64) (strategy: string) =
    // The generate function contains multinomial sampling, temperature, and stop token handling 
        
        let trimInput (input: torch.Tensor) (contextSize: int) =

            match input.shape.[1] > contextSize with
            | true  -> input.narrow(1, -contextSize, contextSize) // Keep only the last contextSize tokens
            | false -> input
        // Remark: Ensures the input sequence does not exceed the model's context size, preventing memory issues and compatibility with positional encodings.
        (*
        The trimInput function ensures that the input sequence passed to your model does not exceed the model's context window size (contextSize).
        Most transformer models (like GPT, LLMs, etc.) have a fixed context window—for example, 1024 or 2048 tokens.
        If you give the model more tokens than it was trained to handle, it will either fail, raise an error, or simply ignore the extra tokens.
        During sequence generation, your input can grow (as you append predicted tokens).
        To avoid errors and keep only the most recent context, you "trim" the input.
        *)

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
                // Select top-k logits and their indices // existuje jeste Top-p sampling
                let struct (probs, indices) = torch.topk(logits / temp, int effectiveTopK, dim=0) //temp is applied here //controls the randomness of the model's output. Lower temperatures lead to more predictable, deterministic outputs, while higher temperatures encourage more diverse, creative, and sometimes random results. 
                let probs = softmax(probs, dim=0L) //softargmax - converts a tuple of K real numbers into a probability distribution of K possible outcomes na dimenzi tenzoru 0
                // Remark: Multinomial sampling converts top-k probabilities into a single token index, introducing controlled randomness in token selection.

                //If probs = tensor([0.1, 0.7, 0.1, 0.1]) (sum = 1), then idx will be index 1 most of the time (because 0.7 probability), but it can also be 0, 2, or 3 occasionally because 10% is still not so small probability.
                let idx = torch.multinomial(probs, 1).item<int64>() //It picks one category index from probs based on probability (so it is random to some extent)
                indices.[idx].item<int64>()
            | "greedy" 
                ->
                // Select the token with the highest logit
                torch.argmax(logits, dim=0).item<int64>()  
                //arguments of the maxima //returns the parameter for which the function result is maximal, here it returns the index (as int64) where the logits tensor is maximized
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
            
            // Forward pass: Compute logits
            // - Uses learned W and b to transform embedding to logits
            //Logits are the raw scores output by the final linear layer of a model, typically before applying a probability-normalizing function like softmax (for multi-class classification) or sigmoid (for binary classification)
            let adjustedLogit: torch.Tensor =
                let trimmedInput = trimInput inputSeq contextSize  // Trim the input sequence to the model's context size to prevent exceeding the maximum sequence length
                let logits: torch.Tensor = model.forward trimmedInput // Run the forward pass to get the model's predictions (logits) for the input sequence; shape: [batch=1, seqLen, vocabSize]
                let lastLogit: torch.Tensor = logits.select(0, 0L).select(0, -1L) // Extract the logits for the last token in the sequence; shape: [vocabSize]
                //if the last layer is a linear transformation, the output for an input ( x ) is computed as z=Wx+bz = Wx + bz = Wx + b, where ( W ) is the weight matrix, ( b ) is the bias, and ( z ) is the vector of logits.
                //The term "logits" originates from the logistic function (sigmoid), where the input to the sigmoid (softmax by analogy) is called the logit.
                                
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
        
        (*
        // bare minimum
        match steps with
        | s when s >= maxSteps ->
            acc // Stop after maxSteps
        | _ ->
            let _ = torch.no_grad() // Disable gradients for inference
            let logits = model.forward inputSeq // Shape: [1, seqLen, vocabSize]
            let lastLogit = logits.select(0, 0L).select(0, -1L) // Logits for last token: [8]
            let nextToken = torch.argmax(lastLogit, dim=0).item<int64>() // Greedy: Pick highest logit
            let newAcc = nextToken :: acc // Accumulate token
            let newInput = torch.cat([|inputSeq; torch.tensor([|nextToken|], device=inputSeq.device).unsqueeze(0L)|], dim=1L) // Append new token
            generate model newInput (steps + 1) maxSteps newAcc
        *)   

    
    let internal main () =     
        
        //CUDA® is a parallel computing platform and programming model developed by NVIDIA for general computing on graphical processing units (GPUs).
        let device = match torch.cuda.is_available() with true -> torch.CUDA | false -> torch.CPU
        printfn "Using device: %A" <| (string device).ToUpper()

        //input processing pipeline (tokenization, embeddings, positional encodings, normalization, and dropout)         

        // TOKENIZATION        
        // External tokenizers such as HuggingFace (Python interop is needed) shall be normally used for code used in production.

        // MANUAL TOKENIZATION
       
        // TODO presunout do dat
        // Data preparation: pre-training input and target sequences (320 examples)
        // V obrovskem mnozstvi textu nachazime oboji - jak vstupni text, tak aji vysledny text (tj. vlastne nic noveho LLM nevytvori)
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
        
        let targetData =  //The next token(s) in the sequence (e.g., [2, 3, 7] for "is yellow <eos>"). //jakoby vysledek (jak by mel vysledek vypadat), tj. jakoby posunute o jeden token (nebot take hledame next token)
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
        let (inputData, targetData) = Tokenizer21.createInputTargetPairs dataset 
        //let (inputData, targetData) = TikTokTokenizer.createInputTargetPairs dataset //TikTokTokenizer //nelze quli male delky     
        
        use input = torch.tensor(inputData, device = device) // [320, 3]
        // Remark: Converts tokenized input data (indices) into a tensor for model processing.
        use target = torch.tensor(targetData, device = device) // [320, 3]
        // Remark: Converts tokenized target data (indices) into a tensor for training.

        //parameters optimized via gradient-based methods are weights, biases, potentially learned positional encodings, and layer normalization parameters γ and β. These collectively form the parameter set θ, adjusted to minimize the loss function during pre-training

        printfn "Starting pre-training..."        
              
        use model = (new Transformer(int64 vocabSize, dModel, nHeads, numLayers)).``to``(device) //embedding and positional embeddings jsou az tady (na rozdil od prednasky)

        use lossFn = new CrossEntropyLoss() //viz přednáška Tomáše Hercega
        use optimizer = torch.optim.Adam(model.parameters(), lr = learningRate) //Adam is the optimizer that updates the model’s parameters (weights) to minimize the loss computed by CrossEntropyLoss
        //Adam (Adaptive Moment Estimation) = gradient-based optimization algorithm       

        model.train()

        trainEpoch model optimizer lossFn input target epochs "Pre-training"
        
        // Fine-tuning: prepare a small dataset to emphasize "the Sun is" -> "is yellow <eos>"
        let fineTuneInputData = Array2D.init 10 3 (fun i k -> [|0L; 1L; 2L|] |> Array.item k) // "the Sun is" for all 10 examples
        // Remark: Data preparation: Creates fine-tuning input sequences as token indices ("the Sun is" = [0, 1, 2]).
        let fineTuneTargetData = Array2D.init 10 3 (fun i k -> [|2L; 3L; 7L|] |> Array.item k) // "is yellow <eos>" for all 10 examples
        // Remark: Data preparation: Creates fine-tuning target sequences as token indices ("is yellow <eos>" = [2, 3, 7]).
        
        use fineTuneInput = torch.tensor(fineTuneInputData, device = device) // [10, 3]
        // Remark: Converts tokenized fine-tuning input data into a tensor.
        use fineTuneTarget = torch.tensor(fineTuneTargetData, device = device) // [10, 3]
        // Remark: Converts tokenized fine-tuning target data into a tensor.

        printfn "Starting fine-tuning..."
        use fineTuneOptimizer = torch.optim.Adam(model.parameters(), lr = learningRate)
        model.train()
        
        trainEpoch model fineTuneOptimizer lossFn fineTuneInput fineTuneTarget fineTuneEpochs "Fine-tuning"

        printfn "Generating sequence after fine-tuning..."
        
        //INFERENCE
        model.eval()  // When model.eval() is called, the Dropout module’s forward method does not apply dropout (i.e., it effectively sets the dropout rate to zero for that pass).
        //This ensures that the Dropout modules in both the TransformerDecoderLayer (applied to attention weights) and the Transformer (applied to embeddings with positional encodings) do not drop any activations, making the model’s output deterministic during inference.
                    
        use inputSeq = torch.tensor([|0L; 1L; 2L|], device = device).unsqueeze 0L // [1, 3]
        // Remark: Data preparation: Creates an input sequence for inference ("The Sun is" = [0, 1, 2]) as token indices.
        
        printf "Generated sequence (token IDs): "
        let generated = generate model inputSeq 0 2 [] contextSize 0.7f topK "top-k" |> List.rev // Generate 2 tokens with temp=0.7, topK=3, top-k sampling
        //let generated = generate model inputSeq 0 2 [] contextSize 0.7f topK "greedy" |> List.rev 
        generated |> List.iter (printf "%d ")
        printfn "\n"
        
        printf "Generated sequence (words): "
        generated |> List.iter (fun id -> printf "%s " (vocabulary |> List.item (int id)))
        // Remark: Converts generated token indices back to words using the vocabulary, reversing the tokenization process for human-readable output.
        printfn "\n"

    (*
    //Simplified code for educational purposes
        
    // Simple Transformer model
    type Transformer(vocabSize: int64, dModel: int64) as self =
        inherit Module<torch.Tensor, torch.Tensor>("Transformer")
        
        // Embedding layer: Maps token IDs to dModel-dimensional vectors
        // - Initialized with random floats when the Transformer is created
        // - Weights (vocabSize x dModel, [5, 4]) are learned during training
        let embedding = Embedding(vocabSize, dModel)
        
        // Output layer: Maps dModel-dimensional embeddings to vocabSize logits
        // - W: Weight matrix, shape [vocabSize, dModel] = [5, 4], initialized with random floats
        // - b: Bias vector, shape [vocabSize] = [5], initialized with zeros or small floats
        // - W and b are learnable parameters, updated during training
        let outputLayer = Linear(dModel, vocabSize)
        
        do self.RegisterComponents()
        
        override _.forward x =
            // Convert token IDs to embeddings: [batch, seq] -> [batch, seq, dModel]
            // - Uses embedding layer’s weights (randomly initialized, then learned)
            let emb = embedding.forward x // Shape: [1, seq, 4]
            // Directly pass to output layer (no transformer layers for simplicity)
            // - Applies logits = emb @ W^T + b
            // - W^T: [4, 5], emb: [1, seq, 4] -> output: [1, seq, 5]
            outputLayer.forward emb // Shape: [1, seq, vocabSize]
    
    module Trainer =
        // Minimal training loop with basic gradient descent
        let train (model: torch.nn.Module<torch.Tensor, torch.Tensor>) (input: torch.Tensor) (target: torch.Tensor) =
            // Use simple gradient descent (no Adam, no optimizations)
            let learningRate = 0.01
            let lossFn = torch.nn.CrossEntropyLoss()
            
            for epoch in 0 .. epochs - 1 do
                // Zero out gradients
                model.zero_grad()
                // Forward pass: Compute logits
                // - Uses W and b from outputLayer, learned from previous epochs
                use output = model.forward input // Shape: [1, seq, vocabSize]
                // Compute loss: Compare logits to target tokens
                use loss = lossFn.forward(output.view(-1L, int64 vocabSize), target.view(-1L))
                // Backward pass: Compute gradients for W, b, and embedding weights
                loss.backward()
                // Update weights with gradient descent
                // - W and b are updated: W -= learningRate * gradient(W)
                for param in model.parameters() do
                    if param.grad().IsSome then
                        param.data().sub_(param.grad().Value.mul(scalar learningRate)) |> ignore
                // Print loss every 20 epochs
                if epoch % 20 = 0 then
                    printfn "Epoch %d, Loss: %.4f" epoch (loss.item<float32>())
    
    // Inference: Predict next token
    let generate (model: torch.nn.Module<torch.Tensor, torch.Tensor>) (inputSeq: torch.Tensor) =
        let _ = torch.no_grad() // Disable gradients for inference
        // Forward pass: Compute logits
        // - Uses learned W and b to transform embedding to logits
        let logits = model.forward inputSeq // Shape: [1, seq, 5]
        // Select logits for last token
        let lastLogit = logits.select(0, 0L).select(0, -1L) // Shape: [5]
        // Greedy sampling: Pick token with highest logit
        torch.argmax(lastLogit, dim=0).item<int64>()
    
    let main () =
        let device = torch.CPU // Use CPU for simplicity
        printfn "Using device: %A" device
        
        // Prepare data
        let (inputData, targetData) = Tokenizer.createInputTargetPairs dataset
        use input = torch.tensor(inputData, device=device) // Shape: [1, 4]
        use target = torch.tensor(targetData, device=device) // Shape: [1, 4]
        
        // Create model
        // - Initializes W ([5, 4]) and b ([5]) with random floats
        use model = new Transformer(int64 vocabSize, dModel).``to``(device)
        
        // Train model
        // - Updates W, b, and embedding weights to learn "The Sun is yellow"
        printfn "Training..."
        model.train()
        Trainer.train model input target
        
        // Inference: Predict next token after "The Sun is"
        printfn "Generating..."
        model.eval()
        use inputSeq = torch.tensor([|0L; 1L; 2L|], device=device).unsqueeze(0) // [1, 3] for "The Sun is"
        let nextToken = generate model inputSeq
        printfn "Predicted token ID: %d (%s)" nextToken vocabulary.[int nextToken]
    
    main()
    
    *)