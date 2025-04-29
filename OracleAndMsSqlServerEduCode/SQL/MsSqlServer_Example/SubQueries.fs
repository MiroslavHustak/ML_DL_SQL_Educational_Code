module SubQueriesTSQL
(*
A subquery is a SQL query nested inside a larger query.

A subquery may occur in :
- A SELECT clause
- A FROM clause
- A WHERE clause
The subquery can be nested inside a SELECT, INSERT, UPDATE, or DELETE statement or inside another subquery.
*)

(*
-- Select query for English
SELECT English, Czech, Note
FROM BLAST_FURNACES
WHERE English IN (
    SELECT English
    FROM BLAST_FURNACES
    GROUP BY English
    HAVING COUNT(*) >= 4
)
ORDER BY English ASC;

-- Select query for Czech
SELECT English, Czech, Note
FROM BLAST_FURNACES
WHERE Czech IN (
    SELECT Czech
    FROM BLAST_FURNACES
    GROUP BY Czech
    HAVING COUNT(*) >= 4
)
ORDER BY Czech ASC;

-- Commit -- ten neni u T-SQL nutny, provadi se automaticky
COMMIT;



ENGLISH                                                                                              CZECH                                                                                                NOTE                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
---------------------------------------------------------------------------------------------------- ---------------------------------------------------------------------------------------------------- ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
damper                                                                                               regulátor tahu v potrubí                                                                             Regulátor tahu v potrubí je podobný klapě                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
damper                                                                                               komínová klapa                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
damper                                                                                               kouřové hradítko                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
damper                                                                                               tlumič                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
damper                                                                                               zař. pro reg. průtoku vzduchu do ventilátoru                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 


ENGLISH                                                                                              CZECH                                                                                                NOTE                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
---------------------------------------------------------------------------------------------------- ---------------------------------------------------------------------------------------------------- ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
design calculation                                                                                   pevnostní výpočet                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
strength analysis                                                                                    pevnostní výpočet                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
stress analysis                                                                                      pevnostní výpočet                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
stress-strain analysis                                                                               pevnostní výpočet                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
electric switchboard                                                                                 rozvaděč                                                                                             Large single panel, frame, or assembly of panels on which are mounted switches, over-current and other protective devices, buses, and usually instruments                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
current distribution board                                                                           rozvaděč                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
electric board                                                                                       rozvaděč                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
electrical panel                                                                                     rozvaděč                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
load centre                                                                                          rozvaděč                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
panelboard                                                                                           rozvaděč                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
power breaker                                                                                        rozvaděč                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     

ENGLISH                                                                                              CZECH                                                                                                NOTE                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
---------------------------------------------------------------------------------------------------- ---------------------------------------------------------------------------------------------------- ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
service panel                                                                                        rozvaděč                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
breaker box                                                                                          rozvaděč                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
breaker panel                                                                                        rozvaděč                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     

14 rows selected. 

Commit complete.
*)

open System
open System.Data.SqlClient

open Helpers
open FsToolkit.ErrorHandling
    
[<Struct>]
type private Builder2 = Builder2 with    
    member _.Bind((optionExpr, errDuCase), nextFunc) =
        match optionExpr with
        | Some value -> nextFunc value 
        | _          -> errDuCase  
    member _.Return x : 'a = x
    
let private pyramidOfDoom = Builder2
    
//ja vim, kontrola na Has.Rows (overeni existence tabulky) sice nize je, ale pro jistotu jeste predtim overeni existence tabulky

let queryExists = @"SELECT COUNT(*) FROM Blast_Furnaces" 
   
let querySelectEN = 
    @"
    SELECT English, Czech
    FROM Blast_Furnaces
    WHERE English IN (
        SELECT English
        FROM Blast_Furnaces
        GROUP BY English
        HAVING COUNT(*) >= 4
    )
    ORDER BY English ASC
    "

let querySelectCZ = 
    @"
    SELECT English, Czech
    FROM Blast_Furnaces
    WHERE Czech IN (
        SELECT Czech
        FROM Blast_Furnaces
        GROUP BY Czech
        HAVING COUNT(*) >= 4
    )
    ORDER BY Czech ASC
    "

let internal selectValues4LinesTSQL getConnectionTSQL closeConnectionTSQL =
    
    try
        let connection: SqlConnection = getConnectionTSQL()
                 
        try      
            use cmdExists = new SqlCommand(queryExists, connection)      
            use cmdSelectEN = new SqlCommand(querySelectEN, connection)   
            use cmdSelectCZ = new SqlCommand(querySelectCZ, connection)
                                
            pyramidOfDoom 
                {
                    let! count = cmdExists.ExecuteScalar() |> Casting.castAs<int>, Error "Blast_Furnaces table not existing"                                   
                    let! _ = count > 0 |> Option.ofBool, Error "Blast_Furnaces table not existing"   
                    let readerEN = lazy cmdSelectEN.ExecuteReader()  //bez lazy evaluation to na rozdil od Oracle hodi error, ze reader neni uzavren
                    let readerCZ = lazy cmdSelectCZ.ExecuteReader()  //bez lazy evaluation to na rozdil od Oracle hodi error, ze reader neni uzavren
                    
                    return Ok [ readerEN; readerCZ ]                     
                }
                                         
            |> function
                | Ok reader ->
                            reader 
                            |> List.map 
                                (fun reader -> 
                                            let reader = reader.Force()
                                            let getValues =                                                 
                                                Seq.initInfinite (fun _ -> reader.Read() && reader.HasRows = true)
                                                |> Seq.takeWhile ((=) true) 
                                                |> Seq.collect
                                                    (fun _ ->                                                                                                                                                                                               
                                                            seq 
                                                                {                                                               
                                                                    Casting.castAs<string> reader.["English"]                                                                               
                                                                    Casting.castAs<string> reader.["Czech"]
                                                                } 
                                                    ) |> List.ofSeq //konverze quli toho, ze a sequence is lazily evaluated, while a list is eagerly evaluated.                                                    
                                                                                
                                            //Jen pro overeni                                         
                                            //getValues |> List.iter (fun item -> printfn "%A" item) 
                                                                                    
                                            let getValues =                                    
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
            closeConnectionTSQL connection //just in case :-) 
    with
    | ex ->
            printfn "%s" ex.Message
            Error ex.Message

