namespace NeuralNetworks

open System
open TorchSharp

module Tokenizer =

    // Define the vocabulary
    let private vocabulary = [ "The"; "Sun"; "is"; "yellow"; "black"; "sky"; "blue"; "<eos>" ]

    // Tokenize a single text sequence into indices, appending <eos>
    let private tokenize (text: string) : int64 list =

        // indexuje slovni zasobu
        let wordToIndex = 
            vocabulary 
            |> Seq.mapi (fun i word -> (word, int64 i))
            |> Map.ofSeq

        text.Split(' ') //indexuje vstupni text a prida <eos>
        |> List.ofArray
        |> List.map 
            (fun word 
                -> 
                match wordToIndex |> Map.tryFind word with
                | Some idx -> idx
                | None     -> failwith <| sprintf "Unknown word: %s" word
            )
        |> fun tokens -> List.append tokens [7L] // Append <eos> token (index 7)

    // Create input-target pairs for a list of text sequences (immutable)
    let createInputTargetPairs (sequences: string list) : (int64[,] * int64[,]) =

        let numSequences = sequences |> List.length
        let seqLength = (tokenize (sequences |> List.head)) |> List.length // Assume all sequences have same length

        // Tokenize all sequences into a list of token arrays
        let tokenizedSequences = sequences |> List.map tokenize

        // Create input data as a list of lists (one list per sequence)
        let inputs =
            tokenizedSequences
            |> List.map id // Input is the full sequence

        // Create target data as a list of lists (shifted by one, pad with 0L)
        let targets =
            tokenizedSequences
            |> List.map 
                (fun tokens 
                    ->
                    List.init seqLength 
                        (fun k 
                            ->
                            match k < seqLength - 1 with 
                            | true  -> tokens |> List.item (k + 1) // Shifted token
                            | false -> 0L // Pad last position with 0L
                        )
            )

        // Convert the lists of lists into 2D arrays
        let inputData =
            Array2D.init numSequences seqLength (fun i j -> inputs.[i].[j])            

        let targetData =
            Array2D.init numSequences seqLength (fun i j -> targets.[i].[j])

        (inputData, targetData)

    // Convert indices back to words for inference output
    let detokenize (indices: int64 list) : string list =

        indices 
        |> List.map 
            (fun idx 
                -> 
                match idx with
                | idx 
                    when idx >= 0L && idx < int64 vocabulary.Length 
                        -> vocabulary |> List.item (int idx)
                | _ 
                        -> failwith <| sprintf "Invalid token index: %i" idx
            )

module TikTokTokenizer =

    open TiktokenSharp

    // NEBUDE FUNGOVAT, nebot mam vocabSize = 8, coz je malo

    // Uses the cl100k_base encoding from Tiktoken, which has a large vocabulary (typically ~100,000 tokens).    
    // Uses tikToken.Encode to convert text into token IDs, which are typically in the range [0, ~100,000) based on the cl100k_base vocabulary.    
    // Appends a custom <eos> token with ID 100000L, which is far outside the model’s vocabSize = 8.   

    // Tokenize a single text sequence into indices, appending <eos>
    let tokenize (text: string) : int64 list =

        // Initialize the TikToken encoder with a specific encoding (e.g., for gpt-3.5-turbo)
        let tikToken = TikToken.GetEncoding "cl100k_base"

        // Define a custom <eos> token ID
        let eosTokenId = 100000L  // Custom ID for <eos>

        let tokenIds = tikToken.Encode text |> Seq.map int64 |> List.ofSeq
        List.append tokenIds [eosTokenId]      

    // Create input-target pairs for a list of text sequences (immutable)
    let createInputTargetPairs (sequences: string list) : (int64[,] * int64[,]) =
           
        let numSequences = sequences |> List.length
        let seqLength = (tokenize (sequences |> List.head)) |> List.length // Assume all sequences have same length

        // Tokenize all sequences into a list of token arrays
        let tokenizedSequences = sequences |> List.map tokenize

        // Create input data as a list of arrays (one array per sequence)
        let inputs =
            tokenizedSequences
            |> List.map id // Input is the full sequence

        // Create target data as a list of arrays (shifted by one, pad with 0L)
        let targets =
            tokenizedSequences
            |> List.map
                (fun tokens 
                    ->
                    List.init seqLength 
                        (fun k
                            ->                   
                            match k < seqLength - 1 with 
                            | true  -> tokens |> List.item (k + 1) // Shifted token
                            | false -> 0L // Pad last position with 0L
                        )
                )

        // Convert the lists of arrays into 2D arrays
        let inputData =
            Array2D.init numSequences seqLength (fun i j -> inputs.[i].[j])

        let targetData =
            Array2D.init numSequences seqLength (fun i j -> targets.[i].[j])

        (inputData, targetData)