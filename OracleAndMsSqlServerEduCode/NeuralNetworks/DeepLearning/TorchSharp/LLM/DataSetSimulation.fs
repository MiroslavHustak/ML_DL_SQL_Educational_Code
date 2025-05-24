namespace NeuralNetworks

open System

//*************************************************************************************************************
// Texts simulating the text for model training (calculating weights and biases) to be included in this module 
//*************************************************************************************************************

module TextData =

    // Simulates a dataset scraped from the internet
    let internal getSequences () : string list =

        (
        List.init 1000 (fun _ -> "The Sun is yellow")
        @ 
        List.init 80 (fun _ -> "The Sun is black")
        )
        @ 
        List.init 100 (fun _ -> "The sky is blue")

    let internal getFineTuningSequences () =

        // PREPARING FINE-TUNING DATA WITH SPECIFIC INPUT-TARGET DATA (SEQUENCES)
        let fineTuneInputData = Array2D.init 10 3 (fun i k -> [|0L; 1L; 2L|] |> Array.item k)
        let fineTuneTargetData = Array2D.init 10 3 (fun i k -> [|2L; 3L; 7L|] |> Array.item k)

        fineTuneInputData, fineTuneTargetData