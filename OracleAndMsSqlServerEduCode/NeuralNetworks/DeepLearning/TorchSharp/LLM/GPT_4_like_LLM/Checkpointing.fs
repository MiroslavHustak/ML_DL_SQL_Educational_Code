namespace NeuralNetworks

open TorchSharp

open type torch.nn
open System.IO

module Checkpointing = 
    
    (*
    Checkpointing
    ****************
    This is just template code for future saving and loading:
    * Model weights (learned parameters of the transformer model, e.g., attention matrices and embeddings)
    * LoRA adapter parameters 
    * Optimizer state
    * Scheduler state
    * State dict of learning rate scheduler
    * Current epoch or step so you can resume exactly where you left off
    * Optionally other metadata such as loss values or anything else you want to track       
    *)

    //All template code is without error handling and null handling - Result and Option types shall be used in production code
    let internal saveCheckpoint (model: torch.nn.Module) (epoch: int) (phase: string) (baseDir: string) =
        
        let weightsPath = Path.Combine(baseDir, sprintf "%s_model_epoch_%d.pt" phase epoch)
        let epochPath = Path.Combine(baseDir, sprintf "%s_epoch_%d.txt" phase epoch)       
        model.save(weightsPath) |> ignore<torch.nn.Module>  
        System.IO.File.WriteAllText(epochPath, epoch.ToString())       
        printfn "Saved checkpoint: %s, %s" weightsPath epochPath
   
    let internal loadCheckpoint (model: torch.nn.Module) (weightsPath: string) (epochPath: string) =
        
        model.load(weightsPath) |> ignore<torch.nn.Module>        
        let epoch = System.IO.File.ReadAllText(epochPath) |> int        
        printfn "Loaded checkpoint: %s, %s (epoch %d)" weightsPath epochPath epoch
        
        // Return the epoch to inform the training loop where to resume
        epoch