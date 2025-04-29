module ViewsTSQL //buzzword VIEW, strucna charakteristika:  kod je ulozeny, neco jako CTE nebo derived table

open System
open System.Data.SqlClient

open Helpers
open FsToolkit.ErrorHandling

    (*
    -- Create a view with the same results as the derived table
    USE [Test_User_MSSQLS]
    GO
    
    CREATE VIEW MyView AS
    SELECT OperatorID, FirstName, LastName, JobTitle
    FROM dbo.Operators
    WHERE JobTitle = 'Valcíř' AND OperatorID >= 1;

    You can think of views as a way to save and reuse the kind of logical structures or result sets that you might otherwise
    create using CTEs or derived tables in individual queries. 
    Views are saved queries stored in the database. They are virtual tables that can be queried like regular tables.
    *)

    (*
    Views:    
    Views are explicitly designed to represent a virtual table based on the result of a SELECT query. 
    Therefore, they are often given a dedicated folder or section in database management tools for easy identification and management.

    In a sense, views can be thought of as named queries that are saved in the database system for reuse. Both Common Table Expressions (CTEs) 
    and derived tables are temporary constructs used within the context of a specific query, and they don't persist beyond that query. 
    Views, on the other hand, are saved in the database and can be referenced by multiple queries.
    
    Window Functions, Derived Tables, CTEs, and Inline TVFs:    
    These constructs are often considered as parts of queries rather than standalone database objects.
    Window functions, derived tables, CTEs, and inline TVFs are elements within a SELECT statement rather than separately stored entities in the database. 
    They are used within the context of a query but do not persist as separate objects.
    The lack of a dedicated folder for these constructs might be because they are transient and exist only for the duration of the query execution.
    *)
        
let internal callView getConnectionTSQL closeConnectionTSQL =
    
    try
        let connection: SqlConnection = getConnectionTSQL()

        let query = 
            "
            SELECT OperatorID, FirstName, LastName, JobTitle
            FROM MyView
            ORDER BY LastName DESC;
            "
                 
        try  
            use cmdCallFunction = new SqlCommand(query, connection)
           
            cmdCallFunction.Parameters.AddWithValue("@jobTitle", "Valcíř") |> ignore
            
            let reader = cmdCallFunction.ExecuteReader() 

            let getValues =                                                 
                Seq.initInfinite (fun _ -> reader.Read() && reader.HasRows = true)
                |> Seq.takeWhile ((=) true) 
                |> Seq.collect
                    (fun _ ->   
                            //V pripade pouziti Oracle zkontroluj skutecny typ sloupce v .NET   

                            //Jen pro overeni 
                            let columnType = reader.GetFieldType(reader.GetOrdinal("OperatorID"))
                            printfn "Column Type: %s" columnType.Name

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

            printfn "Result from View: %A" getValues
          
        finally
            closeConnectionTSQL connection
    with
    | ex -> printfn "%s" ex.Message

