
module CTEs

(*

-- ORACLE SQL DEVELOPER

-- INSERT INTO Operators (OperatorID, FirstName, LastName, JobTitle) VALUES (4, 'John', 'Doe', 'Engineer');
-- INSERT INTO Operators (OperatorID, FirstName, LastName, JobTitle) VALUES (5, 'Jane', 'Smith', 'Technician');
-- INSERT INTO Operators (OperatorID, FirstName, LastName, JobTitle) VALUES (6, 'Bob', 'Johnson', 'Manager');

UPDATE Operators
SET JobTitle = 'Engineer'
WHERE OperatorID = 4;

UPDATE Operators
SET JobTitle = 'Valcíř'
WHERE OperatorID = 5;

UPDATE Operators
SET JobTitle = 'Valcíř'
WHERE OperatorID = 6;

-- Common Table Expression 
WITH OperatorCTE AS (
    SELECT OperatorID, FirstName, LastName, JobTitle
    FROM Operators
    WHERE JobTitle = 'Valcíř'
)
SELECT * FROM OperatorCTE;
/

COMMIT;

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

let querySelect = 
    @"
     WITH OperatorCTE AS
     (
         SELECT OperatorID, FirstName, LastName, JobTitle
         FROM Operators
         WHERE JobTitle = 'Valcíř'
     )
     SELECT * FROM OperatorCTE
    "

let internal selectValues getConnection closeConnection =

    try
        let connection: OracleConnection = getConnection()
             
        try                        
            use cmdExists = new OracleCommand(queryExists, connection)
            use cmdSelect = new OracleCommand(querySelect, connection)                                                       
                            
            pyramidOfDoom 
                {
                    //Bohuzel aji tady Oracle vraci decimal misto int 
                    let! count = cmdExists.ExecuteScalar() |> Casting.castAs<decimal>, Error "Operators table not existing"                                   
                    let! _ = int count > 0 |> Option.ofBool, Error "Operators table not existing"   
                    let reader = cmdSelect.ExecuteReader()  

                    return Ok reader                     
                }
                                     
            |> function
                | Ok reader ->
                            let getValues =                                                 
                                Seq.initInfinite (fun _ -> reader.Read() && reader.HasRows = true)
                                |> Seq.takeWhile ((=) true) 
                                |> Seq.collect
                                    (fun _ ->
                                            //V pripade pouziti Oracle zkontroluj skutecny typ sloupce v .NET   

                                            //Jen pro overeni 
                                            let columnType = reader.GetFieldType(reader.GetOrdinal("OperatorID"))
                                            printfn "Column Type: %s" columnType.Name
                                                         
                                            seq 
                                                {    
                                                    //Oracle nema INT !!! Oracle by default prevede INT na NUMBER(38, 0)
                                                    //Oracle.ManagedDataAccess.Client prevede NUMBER(38, 0) v mem pripade na decimal
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
                            getValues |> List.iter (fun item -> printfn "%A" item) 
                                                                                
                            let getValues = //u Seq to dava prazdnou kolekci - viz varovani vyse                                             
                                match getValues |> List.forall _.IsSome with
                                | true  -> Ok (getValues |> List.choose id)                                       
                                | false -> Error "ReadingDbError"  
                                         
                            //Jen pro overeni                                         
                            getValues |> function Ok value -> value |> List.iter (fun item -> printfn "%s" item) | Error err -> ()
                                         
                            reader.Close() 
                            reader.Dispose()     

                            getValues 
                                        
                | Error err ->
                            Error err                            
        finally
            closeConnection connection
    with
    | ex ->
          printfn "%s" ex.Message
          Error ex.Message
    
    