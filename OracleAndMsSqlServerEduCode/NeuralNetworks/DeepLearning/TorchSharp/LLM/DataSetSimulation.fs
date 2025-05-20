namespace NeuralNetworks

open System

//*************************************************************************************************************
// Texts simulating the text for model training (calculating weights and biases) to be included in this module 
//*************************************************************************************************************

module TextData =

    // Simulates a dataset scraped from the internet
    let getSequences () : string list =

        (
        List.init 300 (fun _ -> "The Sun is yellow")
        @ 
        List.init 80 (fun _ -> "The Sun is black")
        )
        @ 
        List.init 100 (fun _ -> "The sky is blue")