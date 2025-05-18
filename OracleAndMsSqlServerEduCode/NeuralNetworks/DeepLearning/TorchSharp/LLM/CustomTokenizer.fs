namespace NeuralNetworks

open System
open TorchSharp

module TextData =
    // Simulates a dataset scraped from the internet
    let getSequences () : string list =
        (List.init 300 (fun _ -> "The Sun is yellow") @ List.init 80 (fun _ -> "The Sun is black")) @ List.init 100 (fun _ -> "The sky is blue")

module Tokenizer =

    // Define the vocabulary
    let private vocabulary = [|"The"; "Sun"; "is"; "yellow"; "black"; "sky"; "blue"; "<eos>"|]

    // Tokenize a single text sequence into indices, appending <eos>
    let tokenize (text: string) : int64[] =

        // Create a dictionary to map words to indices
        let wordToIndex = 
            vocabulary 
            |> Array.mapi (fun i word -> (word, int64 i))
            |> Map.ofArray

        text.Split(' ')
        |> Array.map 
            (fun word 
                -> 
                match wordToIndex.TryFind word with
                | Some idx -> idx
                | None -> failwith $"Unknown word: {word}"
            )
        |> fun tokens -> Array.append tokens [|7L|] // Append <eos> token (index 7)

    // Create input-target pairs for a list of text sequences (immutable)
    let createInputTargetPairs (sequences: string list) : (int64[,] * int64[,]) =

        let numSequences = sequences.Length
        let seqLength = (tokenize (sequences.[0])).Length // Assume all sequences have same length

        // Tokenize all sequences into a list of token arrays
        let tokenizedSequences = sequences |> List.map tokenize

        // Create input data as a list of arrays (one array per sequence)
        let inputs =
            tokenizedSequences
            |> List.map (fun tokens -> tokens) // Input is the full sequence

        // Create target data as a list of arrays (shifted by one, pad with 0L)
        let targets =
            tokenizedSequences
            |> List.map 
                (fun tokens 
                    ->
                    Array.init seqLength 
                        (fun k 
                            ->
                            match k < seqLength - 1 with 
                            | true  -> tokens.[k + 1] // Shifted token
                            | false -> 0L // Pad last position with 0L
                        )
            )

        // Convert the lists of arrays into 2D arrays
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
                | idx when idx >= 0L && idx < int64 vocabulary.Length -> vocabulary.[int idx]
                | _ -> failwith $"Invalid token index: {idx}"
            )

module TikTokTokenizer =

    open TiktokenSharp

    // NEBUDE FUNGOVAT, nebot mam vocabSize = 8, coz je malo

    // Uses the cl100k_base encoding from Tiktoken, which has a large vocabulary (typically ~100,000 tokens).    
    // Uses tikToken.Encode to convert text into token IDs, which are typically in the range [0, ~100,000) based on the cl100k_base vocabulary.    
    // Appends a custom <eos> token with ID 100000L, which is far outside the model’s vocabSize = 8.   

    // Tokenize a single text sequence into indices, appending <eos>
    let tokenize (text: string) : int64[] =

        // Initialize the TikToken encoder with a specific encoding (e.g., for gpt-3.5-turbo)
        let tikToken = TikToken.GetEncoding "cl100k_base"

        // Define a custom <eos> token ID
        let eosTokenId = 100000L  // Custom ID for <eos>

        let tokenIds = tikToken.Encode text |> Seq.map int64 |> Array.ofSeq
        Array.append tokenIds [|eosTokenId|]      

    // Create input-target pairs for a list of text sequences (immutable)
    let createInputTargetPairs (sequences: string list) : (int64[,] * int64[,]) =

        let numSequences = sequences.Length
        let seqLength = (tokenize sequences.[0]).Length // Assume all sequences have same length

        // Tokenize all sequences into a list of token arrays
        let tokenizedSequences = sequences |> List.map tokenize

        // Create input data as a list of arrays (one array per sequence)
        let inputs =
            tokenizedSequences
            |> List.map (fun tokens -> tokens) // Input is the full sequence

        // Create target data as a list of arrays (shifted by one, pad with 0L)
        let targets =
            tokenizedSequences
            |> List.map
                (fun tokens 
                    ->
                    Array.init seqLength 
                        (fun k
                            ->                   
                            match k < seqLength - 1 with 
                            | true  -> tokens.[k + 1] // Shifted token
                            | false -> 0L // Pad last position with 0L
                        )
                )

        // Convert the lists of arrays into 2D arrays
        let inputData =
            Array2D.init numSequences seqLength (fun i j -> inputs.[i].[j])

        let targetData =
            Array2D.init numSequences seqLength (fun i j -> targets.[i].[j])

        (inputData, targetData)