namespace NeuralNetworks2

open TiktokenSharp
open System.Text.RegularExpressions

module Tokenizer21 =
   
    // Precompute the word-to-index map (functional, not rebuilt per call)
    let private wordToIndex =
        Settings2.vocabulary
        |> Seq.mapi (fun i word -> (word, int64 i))
        |> Map.ofSeq

    // Tokenize a single text sequence into indices, appending <eos>
    let private tokenize (text: string) : int64 list =

        text.Split(' ')
        |> Array.toList
        |> List.map
            (fun word 
                ->
                match Map.tryFind word wordToIndex with
                | Some idx -> idx
                | None     -> failwithf "Unknown word: %s" word
            )
        |> fun tokens -> tokens @ [Settings2.eosTokenIdx]

    // Pad sequence to target length (for batching)
    let private padTo (padIdx: int64) (len: int) (seq: int64 list) =
        
        match seq.Length >= len with
        | true  -> seq |> Seq.take len |> Seq.toList
        | false -> seq @ List.replicate (len - List.length seq) padIdx

    // Create input/target arrays, padding with <pad> token
    let internal createInputTargetPairs (sequences: string list) : (int64[,] * int64[,]) =

        let tokenized = sequences |> List.map tokenize
        let seqLength = tokenized |> List.map List.length |> List.max
        let numSequences = tokenized.Length

        // Inputs are tokens except the last, targets are tokens except the first (classic LM setup)
        let inputs =
            tokenized
            |> List.map (padTo Settings2.padTokenIdx seqLength)

        let targets =
            tokenized
            |> List.map 
                (fun seq 
                    ->
                    let shifted = (seq |> List.tail) @ [Settings2.padTokenIdx]
                    padTo Settings2.padTokenIdx seqLength shifted
                )

        let inputArr = Array2D.init numSequences seqLength (fun i j -> inputs.[i].[j])
        let targetArr = Array2D.init numSequences seqLength (fun i j -> targets.[i].[j])
        inputArr, targetArr

    // Convert indices back to words for inference output
    let internal detokenize (indices: int64 list) : string list =

        indices
        |> List.map 
            (fun idx
                ->
                match idx with
                | _ when idx >= 0L && idx < int64 Settings2.vocabulary.Length
                    -> Settings2.vocabulary.[int idx]
                | _ -> "<unk>"
            ) //common placeholder token used in Natural Language Processing (NLP) meaning "unknown".

module Tokenizer22 =
         
    let internal wordToIndex =
        Settings2.vocabulary
        |> Seq.mapi (fun i word -> (word, int64 i))
        |> Map.ofSeq

    let internal eosTokenIdx = Map.find "<eos>" wordToIndex
    let internal padTokenIdx = Map.find "<pad>" wordToIndex

    let internal tokenize (text: string) : int64 list =

        let pattern = @"<[^>\s]+>|[\w]+|[^\w\s]"

        Regex.Matches(text, pattern)
        |> Seq.cast<Match>
        |> Seq.map (fun m -> m.Value)
        |> Seq.map
            (fun word 
                ->
                match Map.tryFind word wordToIndex with
                | Some idx -> idx
                | None     -> failwithf "Unknown word: %s" word
            )
        |> Seq.toList

    // Pad sequence to target length (for batching)
    let private padTo (padIdx: int64) (len: int) (seq: int64 list) =

        match seq.Length >= len with
        | true  -> seq |> Seq.take len |> Seq.toList
        | false -> seq @ List.replicate (len - List.length seq) padIdx

    // Create input/target arrays, padding with <pad> token
    let internal createInputTargetPairs (sequences: string list) : (int64[,] * int64[,]) =

        let tokenized = sequences |> List.map tokenize
        let seqLength = tokenized |> List.map List.length |> List.max
        let numSequences = tokenized.Length

        // Inputs are tokens except the last, targets are tokens except the first (classic LM setup)
        let inputs =
            tokenized
            |> List.map (padTo padTokenIdx seqLength)

        let targets =
            tokenized
            |> List.map 
                (fun seq
                    ->
                    let shifted = (seq |> List.tail) @ [padTokenIdx]
                    padTo padTokenIdx seqLength shifted
                )

        let inputArr = Array2D.init numSequences seqLength (fun i j -> inputs.[i].[j])
        let targetArr = Array2D.init numSequences seqLength (fun i j -> targets.[i].[j])
        inputArr, targetArr

    // Convert indices back to words for inference output
    let internal detokenize (indices: int64 list) : string list =
        indices
        |> List.map
            (fun idx
                ->
                match idx >= 0L && idx < int64 Settings2.vocabulary.Length with
                | true  -> Settings2.vocabulary |> List.item (int idx)
                | false -> "<UNK>"
            )

module TikTokTokenizer =

    // NEBUDE FUNGOVAT, nebot mam vocabSize = 20, coz je malo

    // Uses the cl100k_base encoding from Tiktoken, which has a large vocabulary (typically ~100,000 tokens).    
    // Uses tikToken.Encode to convert text into token IDs, which are typically in the range [0, ~100,000) based on the cl100k_base vocabulary.    
    // Appends a custom <eos> token with ID 100000L, which is far outside the model’s vocabSize = 8.   

    // Tokenize a single text sequence into indices, appending <eos>
    let private tokenize (text: string) : int64 list =

        // Initialize the TikToken encoder with a specific encoding (e.g., for gpt-3.5-turbo)
        let tikToken = TikToken.GetEncoding "cl100k_base"

        // Define a custom <eos> token ID
        let eosTokenId = 100000L  // Custom ID for <eos>

        let tokenIds = tikToken.Encode text |> Seq.map int64 |> List.ofSeq
        List.append tokenIds [eosTokenId]      

    // Create input-target pairs for a list of text sequences (immutable)
    let internal createInputTargetPairs (sequences: string list) : (int64[,] * int64[,]) =
           
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