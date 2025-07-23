module StoredProceduresTSQL //buzzword PROCEDURE, strucna charakteristika:  kod je ulozeny, na rozdil od views ale obsahuje hlavne ne queries, ale logiku

open System
open System.Data.SqlClient

open Helpers
open FsToolkit.ErrorHandling
open System.Data

    (*
       Views: Views are essentially saved queries that can be queried like tables. 
       They provide a way to abstract complex queries and make them reusable. 
       Views don't contain procedural logic; they are declarative and describe what data to retrieve.
       
       Stored Procedures: Stored procedures, on the other hand, contain procedural logic 
       and are used to encapsulate a sequence of SQL statements, control flow structures (like IF, ELSE, etc.), 
       and business logic. Stored procedures are typically used for tasks such as data manipulation, data validation, and other procedural operations.
       
       In summary, while both views and stored procedures are database objects that are stored in the database, 
       views focus on defining the structure of the result set of a query, whereas stored procedures focus on encapsulating procedural 
       logic and actions that can be executed using the EXECUTE command. Stored procedures often involve control flow constructs and are more 
       akin to traditional procedural programming languages.

      -- Enable PRINT output
      SET NOCOUNT ON;
      
      -- Create or replace the procedure
      CREATE OR ALTER PROCEDURE testingProcedure
          @text_param NVARCHAR(MAX),
          @number_param INT
      AS
      BEGIN
          DECLARE @i INT = 10;
      
          PRINT 'Ahoj';
      
          IF @i = 10
          BEGIN
              PRINT @text_param;
          END
          ELSE IF @i <> 10
          BEGIN
              PRINT CAST(@number_param AS NVARCHAR(MAX)); -- Converting INT to NVARCHAR
          END
          ELSE
          BEGIN
              PRINT 'NULL1 (UNKNOWN1)';
          END;
      END;
      

        -- Running the stored procedure in SSMS
        EXEC testingProcedure 
            @text_param = N'Hello from SSMS',
            @number_param = 42;

    
        -- Create or replace the procedure
        CREATE OR REPLACE PROCEDURE quantityAdapter 
        (
            old_quantity IN NUMBER,
            new_quantity IN NUMBER
        )
        AS
            limit NUMBER := 100;
        BEGIN    
            IF limit >= old_quantity THEN
               UPDATE ProductionOrder SET Quantity = new_quantity WHERE OrderID = 103;
            ELSE
               UPDATE ProductionOrder SET Quantity = 6 WHERE OrderID = 103;
            END IF;
        END quantityAdapter;

        -- Running the stored procedure in a worksheet
        DECLARE
            quantity_sum NUMBER;
            addValue NUMBER;
        BEGIN
            SELECT SUM(quantity) INTO quantity_sum
            FROM ProductionOrder
            WHERE quantity IS NOT NULL;

            addValue := quantity_sum + 20.0;

            quantityAdapter(old_quantity => quantity_sum, new_quantity => addValue);

        COMMIT;
    END;
    /
*)

let private queryCreateStoredProcedure =  //jen zkusebni stored procedure
    "       
    -- Create or replace the procedure
    CREATE PROCEDURE testingProcedure
        @text_param NVARCHAR(MAX),
        @number_param INT
    AS
    BEGIN
        DECLARE @i INT = 10;
    
        PRINT 'Ahoj';
    
        IF @i = 10
        BEGIN
            PRINT @text_param;
        END
        ELSE IF @i <> 10
        BEGIN
            PRINT CAST(@number_param AS NVARCHAR(MAX)); -- Converting INT to NVARCHAR
        END
        ELSE
        BEGIN
            PRINT 'NULL1 (UNKNOWN1)';
        END;
    END;
    " 
    
let internal createStoredProcedure getConnectionTSQL closeConnectionTSQL =
    
    try
        let connection: SqlConnection = getConnectionTSQL()
                 
        try         
            use cmdCreateStoredProcedure = new SqlCommand(queryCreateStoredProcedure, connection)                             
                         
            cmdCreateStoredProcedure.ExecuteNonQuery() |> ignore
           
        finally
            closeConnectionTSQL connection
    with
    | ex -> printfn "%s" ex.Message

let internal executeStoredProcedure getConnectionTSQL closeConnectionTSQL =
    
    try
        let connection: SqlConnection = getConnectionTSQL()
                 
        try         
            use cmdExecuteStoredProcedure = new SqlCommand("testingProcedure", connection)    
            
            cmdExecuteStoredProcedure.CommandType <- CommandType.StoredProcedure

            // Add parameters
            cmdExecuteStoredProcedure.Parameters.Add("@text_param", SqlDbType.NVarChar).Value <- "Hello from F#"
            cmdExecuteStoredProcedure.Parameters.Add("@number_param", SqlDbType.Int).Value <- 42

            //cmdExecuteStoredProcedure.Parameters.Add("@text_param", "Hello from F#") |> ignore //tohle nejde
            //cmdExecuteStoredProcedure.Parameters.Add("@number_param", 42) |> ignore  //tohle nejde

            cmdExecuteStoredProcedure.ExecuteNonQuery() |> ignore //zadny vystup jsem tady nerobil
           
        finally
            closeConnectionTSQL connection
    with
    | ex -> printfn "%s" ex.Message


