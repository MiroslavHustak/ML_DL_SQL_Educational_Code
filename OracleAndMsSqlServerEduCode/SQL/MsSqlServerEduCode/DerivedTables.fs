module DerivedTablesTSQL //buzzword FROM, strucna charakteristika: virtualni castecna tabulka, abychom nemuseli dotazovat celou table

(*
    SELECT OperatorID, FirstName, LastName, JobTitle
    FROM (
        SELECT OperatorID, FirstName, LastName, JobTitle
        FROM Operators
        WHERE JobTitle = 'Valcíř'
    ) myDerivedTable
    WHERE OperatorID >= 1
    ORDER By LastName DESC;
   
    pro porovnani

    SELECT OperatorID, FirstName, LastName, JobTitle
    FROM Operators
    WHERE JobTitle = 'Valcíř' AND OperatorID >= 1
    ORDER By LastName;
       
*)

(*
Derived tables, also known as inline views or subquery factoring, are temporary result sets that are created within 
the scope of a larger SQL query. These tables are derived from a subquery and are not stored in the database permanently.
Derived tables are often used to narrow down or filter the data from a larger table. 
*)

open System
open Helpers
open System.Data.SqlClient
open FsToolkit.ErrorHandling
    
[<Struct>]
type private Builder2 = Builder2 with    
    member _.Bind((optionExpr, errDuCase), nextFunc) =
        match optionExpr with
        | Some value -> nextFunc value 
        | _          -> errDuCase  
    member _.Return x : 'a = x
    member _.Using x : 'a = x
    
let private pyramidOfDoom = Builder2
    
//ja vim, kontrola na Has.Rows (overeni existence tabulky) sice nize je, ale pro jistotu jeste predtim overeni existence tabulky
let queryExists = @"SELECT COUNT(*) FROM Operators" 
    
let querySelectDT = 
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

let querySelect = //pro porovnani
    @"
    SELECT OperatorID, FirstName, LastName, JobTitle
    FROM Operators
    WHERE JobTitle = 'Valcíř' AND OperatorID >= 1
    ORDER By LastName;
    "

//Bez tryWith bloku
let internal selectValuesDT getConnectionTSQL closeConnectionTSQL =
      
    let connection: SqlConnection = getConnectionTSQL()            
    use cmdExists = new SqlCommand(queryExists, connection)                                    
    use cmdSelectDT = new SqlCommand(querySelectDT, connection)   
    use cmdSelect = new SqlCommand(querySelect, connection)

    use readerDT = cmdSelectDT.ExecuteReader()  //nejde 2x reader (ExecuteReader)
    //use reader = cmdSelect.ExecuteReader() 
                                
    [ readerDT ]   
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
