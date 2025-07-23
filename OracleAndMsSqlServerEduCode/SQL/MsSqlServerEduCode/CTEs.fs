module CTEsTSQL //buzzword WITH, strucna charakteristika: virtualni castecna tabulka, abychom nemuseli dotazovat celou table, podobna derived table


//Both DTs and CTEs create temporary result sets within a query.
//Derived tables are scoped to the specific query in which they are defined.
//CTEs can be referenced multiple times within the same query and offer a more modular structure.

(*
Using ITVFs and CTEs both leads to a temporary result set, but:

ITVFs can take parameters, CTEs not
ITFV are functions that can be reused everywhere, CTEs are not (CTEs can only be reused within the same query).
*)

(*
WITH MyCTE AS (
SELECT OperatorID, FirstName, LastName, JobTitle
FROM Operators
WHERE JobTitle = 'Valcíř'
)

SELECT OperatorID, FirstName, LastName, JobTitle
FROM MyCTE
WHERE OperatorID >= 1
ORDER BY LastName DESC;


pro porovnani derived table

SELECT OperatorID, FirstName, LastName, JobTitle
FROM (
    SELECT OperatorID, FirstName, LastName, JobTitle
    FROM Operators
    WHERE JobTitle = 'Valcíř'
) myDerivedTable
WHERE OperatorID >= 1
ORDER By LastName DESC;
*)

open System
open Helpers
open FsToolkit.ErrorHandling
open System.Data.SqlClient

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

let querySelect = 
    @"
     WITH MyCTE AS (
     SELECT OperatorID, FirstName, LastName, JobTitle
     FROM Operators
     WHERE JobTitle = 'Valcíř'
     )

     SELECT OperatorID, FirstName, LastName, JobTitle
     FROM MyCTE
     WHERE OperatorID >= 1
     ORDER BY LastName DESC;
    "

//Bez tryWith bloku
let internal selectValuesCTETSQL getConnectionTSQL closeConnectionTSQL =
      
    let connection: SqlConnection = getConnectionTSQL()            
    use cmdExists = new SqlCommand(queryExists, connection)                                    
    use cmdSelect = new SqlCommand(querySelect, connection)

    use readerCTE = cmdSelect.ExecuteReader()  //nejde 2x eagerly evaluated reader (ExecuteReader) //viz Subquery.fs
                                
    [ readerCTE ]   
    |> List.map 
        (fun reader -> 
            let getValues =  
                Seq.initInfinite (fun _ -> reader.Read() && reader.HasRows = true)
                |> Seq.takeWhile ((=) true) 
                |> Seq.collect
                    (fun _ ->    
                            let indexOperatorID = reader.GetOrdinal("OperatorID")
                            let indexFirstName = reader.GetOrdinal("FirstName")
                            let indexLastName = reader.GetOrdinal("LastName")
                            let indexJobTitle = reader.GetOrdinal("JobTitle")

                            seq 
                                {   
                                    reader.GetInt32(indexOperatorID) |> Option.ofNull  
                                    |> Option.bind (fun item -> Option.filter (fun item -> not (item.Equals(String.Empty))) (Some (string item))) 
                                    reader.GetString(indexFirstName) |> Option.ofNull                                                                               
                                    reader.GetString(indexLastName) |> Option.ofNull
                                    reader.GetString(indexJobTitle) |> Option.ofNull
                                } 
                ) |> List.ofSeq  
                                             
            //Jen pro overeni                                         
            getValues |> List.iter (fun item -> printfn "%A" item) 
                                                                                    
            let getValues = //u Seq to dava prazdnou kolekci - viz varovani vyse                                             
                match getValues |> List.forall _.IsSome with
                | true  -> Ok (getValues |> List.choose id)                                       
                | false -> Error "ReadingDbError"  
                                             
            //Jen pro overeni                                         
            getValues |> function Ok value -> value |> List.iter (fun item -> printfn "%s" item) | Error err -> ()
                                             
            reader.Close() 
            reader.Dispose()    
        )
