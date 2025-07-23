namespace Helpers

open FsToolkit.ErrorHandling

open System

module Casting =
       
    let internal castAs<'a> (o: obj) : 'a option =   
        match Option.ofNull o with
        | Some (:? 'a as result) -> Some result
        | _                      -> None

module Option = 

    let internal fromBool value =                           
        function   
        | true  -> Some value  
        | false -> None

    let internal ofBool =                           
        function   
        | true  -> Some ()  
        | false -> None

    let internal toBool =                           
        function   
        | Some _ -> true  
        | None   -> false 

    let internal ofStringOption str =
        str 
        |> Option.bind (fun item -> Option.filter (fun item -> not (item.Equals(String.Empty))) (Some (string item))) 