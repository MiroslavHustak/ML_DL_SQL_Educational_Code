module DerivedTables

(*
    SELECT *
    FROM (
        SELECT OperatorID, FirstName, LastName, JobTitle
        FROM Operators
        WHERE JobTitle = 'Manager'
    ) DerivedTable;

    OPERATORID FIRSTNAME                                          LASTNAME                                           JOBTITLE                                          
    ---------- -------------------------------------------------- -------------------------------------------------- --------------------------------------------------
             3 Bob                                                Johnson                                            Manager   

    pro porovnani

    SELECT OperatorID, FirstName, LastName, JobTitle
    FROM Operators
    WHERE JobTitle = 'Manager';

    OPERATORID FIRSTNAME                                          LASTNAME                                           JOBTITLE                                          
    ---------- -------------------------------------------------- -------------------------------------------------- --------------------------------------------------
             3 Bob                                                Johnson                                            Manager    
*)
open System
open Helpers
open FsToolkit.ErrorHandling
open Oracle.ManagedDataAccess.Client
    
[<Struct>]
type private Builder2 = Builder2 with    
    member _.Bind((optionExpr, errDuCase), nextFunc) =
        match optionExpr with
        | Some value -> nextFunc value 
        | _          -> errDuCase  
    member _.Return x : 'a = x
    
let private pyramidOfDoom = Builder2
    
//ja vim, kontrola na Has.Rows (overeni existence tabulky) sice nize je, ale pro jistotu jeste predtim overeni existence tabulky
let queryExists = @"SELECT COUNT(*) FROM Operators" 
    
let querySelectDT = 
    @"
    SELECT *
    FROM
    (
        SELECT OperatorID, FirstName, LastName, JobTitle
        FROM Operators
        WHERE JobTitle = 'Manager'
    ) DerivedTable"

let querySelect = //pro porovnani
    @"
    SELECT OperatorID, FirstName, LastName, JobTitle
    FROM Operators
    WHERE JobTitle = 'Manager'
    "

let internal selectValuesDT getConnection closeConnection =
    
    try
        let connection: OracleConnection = getConnection()
                 
        try
            
            use cmdExists = new OracleCommand(queryExists, connection)                                    
            use cmdSelectDT = new OracleCommand(querySelectDT, connection)   
            use cmdSelect = new OracleCommand(querySelect, connection)
                                
            pyramidOfDoom 
                {
                    let! count = cmdExists.ExecuteScalar() |> Casting.castAs<decimal>, Error "Operators table not existing"                                   
                    let! _ = int count > 0 |> Option.ofBool, Error "Operators table not existing"   
                    let readerDT = cmdSelectDT.ExecuteReader()  
                    let reader = cmdSelect.ExecuteReader() 
                    
                    return Ok [ readerDT; reader ]                     
                }
                                         
            |> function
                | Ok reader ->
                            reader 
                            |> List.map 
                                (fun reader -> 
                                    let getValues =                                                 
                                        Seq.initInfinite (fun _ -> reader.Read() && reader.HasRows = true)
                                        |> Seq.takeWhile ((=) true) 
                                        |> Seq.collect
                                            (fun _ ->                                                                                                                                                                                               
                                                    seq 
                                                        {   
                                                            Casting.castAs<decimal> reader.["OperatorID"] 
                                                            |> Option.bind (fun item -> Option.filter (fun item -> not (item.Equals(String.Empty))) (Some (string item))) 
                                                            Casting.castAs<string> reader.["FirstName"]                                                                               
                                                            Casting.castAs<string> reader.["LastName"]
                                                            Casting.castAs<string> reader.["JobTitle"]  
                                                        } 
                                            ) |> List.ofSeq 
                                                     
                                    //Pozor na nize uvedene problemy, uz jsem to nekde jinde zazil
                                    //In F#, a sequence is lazily evaluated, while a list is eagerly evaluated. 
                                    //This means that certain operations on sequences might not be executed until they are explicitly enumerated. 
                                             
                                    //Jen pro overeni                                         
                                    //getValues |> List.iter (fun item -> printfn "%A" item) 
                                                                                    
                                    let getValues = //u Seq to dava prazdnou kolekci - viz varovani vyse                                             
                                        match getValues |> List.forall _.IsSome with
                                        | true  -> Ok (getValues |> List.choose id)                                       
                                        | false -> Error "ReadingDbError"  
                                             
                                    //Jen pro overeni                                         
                                    getValues |> function Ok value -> value |> List.iter (fun item -> printfn "%s" item) | Error err -> ()
                                             
                                    reader.Close() 
                                    reader.Dispose()     
    
                                    getValues 
                            ) |> Ok
                | Error err ->
                            Error err                            
        finally
            closeConnection connection //just in case :-) 
    with
    | ex ->
            printfn "%s" ex.Message
            Error ex.Message

