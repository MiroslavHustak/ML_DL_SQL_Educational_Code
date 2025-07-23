module TriggersTSQL //Buzzword TRIGGER a dalsi viz nize (pro PL/SQL se to drobet lisi)

open System
open System.Data.SqlClient

open Helpers
open System.Data

let private queryCreateTrigger =  //jen zkusebni trigger
    "   
    CREATE OR ALTER TRIGGER trg_UpdateProductionOrderStatus
    ON ProductionOrder
    AFTER UPDATE
    AS
    BEGIN
        SET NOCOUNT ON;
    
        -- Check if Quantity column is updated
        IF UPDATE(Quantity)
        BEGIN
            UPDATE po
            SET po.Status = CASE
                                WHEN i.Quantity > 300 THEN 'High Quantity'
                                ELSE 'Normal Quantity'
                            END
            FROM ProductionOrder po
            INNER JOIN inserted i ON po.OrderID = i.OrderID;
        END
    END;
    " 
    
let internal createTriggerTSQL getConnection closeConnection =
    
    try
        let connection: SqlConnection = getConnection()
                 
        try 
            use cmdCreateStoredProcedure = new SqlCommand(queryCreateTrigger, connection)
                          
            cmdCreateStoredProcedure.ExecuteNonQuery() |> ignore   
          
        finally
            closeConnection connection
    with
    | ex -> printfn "%s" ex.Message
          
(*
CREATE TRIGGER [schema_name.]trigger_name
ON table_name
{FOR | AFTER | INSTEAD OF} {[INSERT] [,] [UPDATE] [,] [DELETE]}
AS
{sql_statements}

The INSTEAD OF trigger shall perform controls and replace the original action with the action in the trigger, while the FOR | AFTER (they mean the same) trigger shall run additional commands after the original statement has completed.

The part {[INSERT] [,] [UPDATE] [,] [DELETE]} denotes which command actually fires this trigger. We must specify at least one option, but we could use multiple if we need it.


CREATE TRIGGER trg_UpdateProductionOrderStatus
ON ProductionOrder
AFTER UPDATE
AS
BEGIN
    SET NOCOUNT ON; --When SET NOCOUNT is set to ON, it prevents SQL Server from sending messages like "(N row(s) affected)" to the client after executing INSERT, UPDATE, DELETE, or SELECT. May improve performance.

    -- Check if Quantity column is updated
    IF UPDATE(Quantity)
    BEGIN
        UPDATE po
        SET po.Status = CASE
                            WHEN i.Quantity > 200 THEN 'High Quantity'
                            ELSE 'Normal Quantity'
                        END
        FROM ProductionOrder po
        INNER JOIN inserted i ON po.OrderID = i.OrderID;
    END
END;

*)

