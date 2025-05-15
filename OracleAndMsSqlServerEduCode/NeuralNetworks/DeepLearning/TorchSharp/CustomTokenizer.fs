namespace NeuralNetworks

open System
open TorchSharp

module TextData =
    // Simulates a dataset scraped from the internet
    let getSequences () : string list =
        List.init 300 (fun _ -> "The Sun is yellow") @ ["The Sun is black"]
        // 300 instances of "The Sun is yellow", 1 instance of "The Sun is black"

module Tokenizer =
    // Define the vocabulary
    let private vocabulary = [|"The"; "Sun"; "is"; "yellow"; "black"; "sky"; "blue"; "<eos>"|]

    // Create a dictionary to map words to indices
    let private wordToIndex = 
        vocabulary 
        |> Array.mapi (fun i word -> (word, int64 i))
        |> Map.ofArray

    // Tokenize a single text sequence into indices, appending <eos>
    let tokenize (text: string) : int64[] =
        text.Split(' ')
        |> Array.map (fun word -> 
            match wordToIndex.TryFind word with
            | Some idx -> idx
            | None -> failwith $"Unknown word: {word}")
        |> fun tokens -> Array.append tokens [|7L|] // Append <eos> token (index 7)

    // Create input-target pairs for a list of text sequences
    let createInputTargetPairs (sequences: string list) : (int64[,] * int64[,]) =

        let numSequences = sequences.Length
        let seqLength = (tokenize (sequences.[0])).Length // Assume all sequences have same length
        let inputData = Array2D.zeroCreate<int64> numSequences seqLength
        let targetData = Array2D.zeroCreate<int64> numSequences seqLength
        
        sequences
        |> List.iteri (fun i seq ->
            let tokens = tokenize seq
            // Input: full sequence
            tokens 
            |> Array.mapi (fun k token -> inputData.[i, k] <- token)
            |> ignore
            // Target: shifted by one (pad with 0L for last position)
            tokens
            |> Array.mapi (fun k token ->
                match k with
                | k when k < seqLength - 1 -> targetData.[i, k] <- tokens.[k + 1]
                | _ -> targetData.[i, seqLength - 1] <- 0L) // Padding for last position
            |> ignore
        )
        
        (inputData, targetData)

    // Convert indices back to words for inference output
    let detokenize (indices: int64 list) : string list =
        indices 
        |> List.map (fun idx -> 
            match idx with
            | idx when idx >= 0L && idx < int64 vocabulary.Length -> vocabulary.[int idx]
            | _ -> failwith $"Invalid token index: {idx}")