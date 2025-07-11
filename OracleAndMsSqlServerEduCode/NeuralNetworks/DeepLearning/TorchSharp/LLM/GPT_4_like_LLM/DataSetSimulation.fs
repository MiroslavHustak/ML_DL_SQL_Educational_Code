namespace NeuralNetworks

//*********************************************
// Texts simulating dataset for model training  
//*********************************************

module TextData2 =

    type QAPair = string * string

    /// Returns a shuffled list of Q&A sequences for pre-training (as single strings)
    let internal getSequences () : string list =

        let qaPairs =
            [
                "What is the colour of the Sun? <sep> The colour of the Sun is yellow."
                "What is the colour of the sky? <sep> The colour of the sky is blue."
                "What colour is the Sun? <sep> yellow."
                "What colour is the sky? <sep> blue."
                "Is the Sun black? <sep> No."
                "Is the sky yellow? <sep> No."
                "Is the Sun yellow? <sep> Yes."
                "Is the sky orange? <sep> No."
                "Is the sky blue? <sep> Yes."
                "The sky is not green."
                "The Sun is not green."
                "Orange is not green."
                "Is the colour green blue? <sep> No."
                "Orange is not blue."
                "Orange is not yellow."
            ]

        let nExamplesPerPair = 16
        let qa = List.collect (fun s -> List.replicate nExamplesPerPair s) qaPairs
        let all = qa
        let seed = 42

        let shuffleList (input: 'a list) (seed: int) : 'a list =

            let rnd = System.Random seed

            input
            |> List.map (fun x -> rnd.Next(), x)
            |> List.sortBy fst
            |> List.map snd

        shuffleList all seed

    /// Builds (input, target) arrays for decoder-only causal LM fine-tuning.
    /// Pattern: [prompt tokens] <sep> [answer tokens] <eos>
    /// Input: all tokens except last; Target: all tokens except first.
    /// Masking: Use pad token for prompt and <sep> in target so loss is not computed there.
    let internal getFineTuningCausalLMSequences lora : int64[,] * int64[,] =
          
        let qaPairs : QAPair list =
            [
                "What is the colour of the Sun?",     "The colour of the Sun is yellow."
                "What is the colour of the sky?",     "The colour of the sky is blue."
                "What colour is the Sun?",            "yellow."
                "What colour is the sky?",            "blue."
                "Is the Sun black?",                  "No."
                "Is the sky yellow?",                 "No."
                "Is the Sun yellow?",                 "Yes."
                "Is the sky blue?",                   "Yes."
                "What is the colour of the Sun?", "<sep> The colour of the Sun is yellow."
                "What is the colour of the sky?", "<sep> The colour of the sky is blue."
                "What colour is the Sun?", "<sep> yellow."
                "What colour is the sky?", "<sep> blue."
                "Is the Sun black?", "<sep> No."
                "Is the sky yellow?", "<sep> No."
                "Is the Sun yellow?", "<sep> Yes."
                "Is the sky blue?", "<sep> Yes."
            ]
        
        let qaPairsLoRA : QAPair list =
            [
                "What is the colour of the Sun?",     "The colour of the Sun is yellow."
                "What is the colour of the sky?",     "The colour of the sky is blue."
                "What colour is the Sun?",            "yellow."
                "What colour is the sky?",            "blue."
                "Is the Sun orange?",                 "No."
                "Is the sky orange?",                 "No."
                "Is the Sun yellow?",                 "Yes."
                "Is the sky blue?",                   "Yes."
            ]

        let qaPairs = 
            match lora with
            | true  -> qaPairs
            | false -> qaPairsLoRA

        let nExamplesPerPair = 32

        let pad = Tokenizer2.wordToIndex.["<pad>"]
        let eos = Tokenizer2.eosTokenIdx
        let sep = Tokenizer2.wordToIndex.["<sep>"]

        // Build all full sequences for fine-tuning: Q + <sep> + A + <eos>
        let fullSeqs =
            qaPairs
            |> List.collect 
                (fun (q, a)
                    ->
                    let qTokens = Tokenizer2.tokenize q
                    let aTokens = Tokenizer2.tokenize a
                    let sequence = qTokens @ [sep] @ aTokens @ [eos]
                    List.replicate nExamplesPerPair sequence
                )

        let maxLen = //24
            fullSeqs
            |> List.map List.length
            |> List.max
            |> (+) 0

        // Pad with <pad>
        let paddedSeqs =
            fullSeqs
            |> List.map 
                (fun seq 
                    ->
                    match seq.Length >= maxLen with
                    | true  -> seq
                    | false -> seq @ List.replicate (maxLen - seq.Length) pad
            )

        // Classic LM shifting for input/target
        let inputs =
            paddedSeqs
            |> List.map (fun seq -> seq |> List.take (maxLen - 1))

        let targets =
            paddedSeqs
            |> List.map (fun seq -> seq |> List.skip 1)

        // Mask prompt+<sep> in targets by setting to pad
        let maskPrompt (q: string) =
            let qTokens = Tokenizer2.tokenize q
            // +1 for <sep>
            List.length qTokens + 1

        let masks =
            qaPairs
            |> List.collect
                (fun (q, _) 
                    ->
                    let promptLen = maskPrompt q
                    List.replicate nExamplesPerPair promptLen
            )

        let maskedTargets =
            List.zip targets masks
            |> List.map 
                (fun (tgt, maskLen)
                    ->
                    tgt
                    |> List.mapi 
                        (fun i tok 
                            ->
                            match i < maskLen with
                            | true  -> pad
                            | false -> tok
                        ) 
                )

        let nSamples = inputs.Length
        let seqLen = maxLen - 1

        let inputArr = Array2D.init nSamples seqLen (fun i j -> inputs.[i].[j])
        let targetArr = Array2D.init nSamples seqLen (fun i j -> maskedTargets.[i].[j])

        inputArr, targetArr