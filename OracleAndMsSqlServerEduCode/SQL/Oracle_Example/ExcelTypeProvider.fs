module ExcelTypeProvider

open FSharp.Interop.Excel

open System
open System.Data
open FsToolkit.ErrorHandling
open Oracle.ManagedDataAccess.Client

open Helpers

[<Struct>]
type private Builder2 = Builder2 with    
    member _.Bind((optionExpr, errDuCase), nextFunc) =
        match optionExpr with
        | Some value -> nextFunc value 
        | _          -> errDuCase  
    member _.Return x : 'a = x

let private pyramidOfDoom = Builder2

//a type provider ensures type safety at compile time
type private DataTypesTest = ExcelFile<"e:\\source\\repos\\OracleDB_Excel_Files\\Slovnicek AJ new.xlsx", "AJ-CJ-AJ", HasHeaders = false>
//kdyz se daji headers, vezme to pouze sloupce s headers

let internal insertOrUpdateDictionary getConnection closeConnection query path list =

    try
        let connection: OracleConnection = getConnection() 
        
        try  
            //new = an instance of a type, not a class in the traditional object-oriented programming sense
            let file = new DataTypesTest(path, "AJ-CJ-AJ") // F# type, hopefully no nulls
            let rows = file.Data |> Seq.toArray 
            let epRowsCount = rows.Length     //TODO zjistit, proc to bere o jeden radek vice
            let listRange = [ 0 .. epRowsCount - 1 ] //TODO zjisti, cemu to na rozdil od T-SQL nevadi

            printfn "epRowsCount %i" epRowsCount
            printfn "listRangeLength %i" listRange.Length

            use cmdDropSequence = new OracleCommand(List.item 0 query, connection)
            use cmdDeleteAll = new OracleCommand(List.item 1 query, connection)
            use cmdCreateSequence = new OracleCommand(List.item 2 query, connection)
            use cmdInsert = new OracleCommand(List.item 3 query, connection)
            use cmdUpdate1 = new OracleCommand(List.item 4 query, connection)
            use cmdUpdate2 = new OracleCommand(List.item 5 query, connection)
             
            printfn "drop seq %i" <| cmdDropSequence.ExecuteNonQuery() // -1
            
            printfn "del all %i" <| cmdDeleteAll.ExecuteNonQuery() //number of affected rows
            
            printfn "create seq %i" <| cmdCreateSequence.ExecuteNonQuery() // -1                         
            
            listRange
            |> List.iter
                (fun i ->
                        cmdInsert.Parameters.Clear() // Clear parameters for each iteration                                                
                        cmdInsert.Parameters.Add(":English", OracleDbType.Varchar2).Value <- rows.[i].Column3
                        cmdInsert.Parameters.Add(":Czech", OracleDbType.Varchar2).Value <- rows.[i].Column5
                        cmdInsert.Parameters.Add(":Note", OracleDbType.Varchar2).Value <- rows.[i].Column13

                        cmdInsert.ExecuteNonQuery() |> ignore //number of affected rows
                )
         
            //number of affected rows
            printfn "update1 %i" <| cmdUpdate1.ExecuteNonQuery() //Dynamic SQL needed for stored procedure with the table name and primary key column as parameters

            //-1
            printfn "update2 %i" 
            <|
            (
                let x = 
                    cmdUpdate2.Parameters.Clear() // Clear parameters for each iteration                                                
                    cmdUpdate2.Parameters.Add(":table_name", OracleDbType.Varchar2).Value <- List.item 0 list
                    cmdUpdate2.Parameters.Add(":primary_key_column", OracleDbType.Varchar2).Value <- List.item 1 list
                    cmdUpdate2.ExecuteNonQuery() 
                x
            )               
            
        finally
            closeConnection connection
    with
    | ex ->
          printfn "%s" ex.Message

    //Dynamic SQL is achieved using the EXECUTE IMMEDIATE statement. 
    //The use of bind variables with the USING clause helps prevent SQL injection and provides a way to pass values into the dynamic SQL statement.
    //Using dynamic SQL in the DELETE_NULL_ROWS procedure allows you to pass the table name and primary key column as parameters, 
    //While dynamic SQL (building SQL statements dynamically) is sometimes necessary, avoid it when a static query suffices. This helps reduce the risk of introducing vulnerabilities.