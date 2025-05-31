namespace NeuralNetworks

//*************************************************************************************************************
// Texts simulating the text for model training (calculating weights and biases) to be included in this module 
//*************************************************************************************************************

open Settings

module TextData =

    // Simulate a small, balanced dataset for pre-training
    let internal getSequences () : string list =
        
        let core = List.replicate 100 "The Sun is yellow"
        let rare1 = List.replicate 10 "The Sun is black"
        let blue1 = List.replicate 30 "The sky is blue"
        let blue2 = List.replicate 2 "The Sun is blue"
        let yellow2 = List.replicate 2 "The sky is yellow"        
        
        // Simulation of Fisher-Yates shuffle for efficiency (not true Fisher-Yates)
        let all = core @ rare1 @ blue1 @ blue2 @ yellow2
        let seed = 42
        
        let shuffleList (input: 'a list) (seed: int) : 'a list =
            let rnd = System.Random seed
            input
            |> List.map (fun x -> rnd.Next(), x)
            |> List.sortBy fst
            |> List.map snd

        shuffleList all seed  

    // PREPARING FINE-TUNING DATA WITH SPECIFIC INPUT-TARGET DATA (SEQUENCES)
    let internal getFineTuningSequences () =

        let prompt = [0L; 1L; 2L] // "The Sun is"
        let completion = [3L; eosTokenIdx] // "yellow" <eos>
        let sequence = List.append prompt completion // [|0L; 1L; 2L; 3L; eosTokenIdx|]
               
        let input = sequence |> List.take (List.length sequence - 1)  // [0L; 1L; 2L; 3L]
        let target = sequence |> List.skip 1    
        
        let nExamples = 30

        let fineTuneInputData = Array2D.init nExamples input.Length (fun _ k -> input.[k])
        let fineTuneTargetData = Array2D.init nExamples target.Length (fun _ k -> target.[k])

        fineTuneInputData, fineTuneTargetData