//namespace Helpers

[<RequireQualifiedAccess>]
module List.Parallel

let map (action : 'a -> 'b) (list : 'a list) =

    match list with
    | [] -> 
         []
    | _  ->
         list
         |> List.map (fun item -> async { return action item })  
         |> Async.Parallel  
         |> Async.RunSynchronously  
         |> List.ofArray