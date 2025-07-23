module StoredProcedures

open ScalarFunctions
open FsToolkit.ErrorHandling
open Oracle.ManagedDataAccess.Client


    (*
        ORACLE SQL DEVELOPER

        PL/SQL + SQL

        SYS!!!
        GRANT DEBUG CONNECT SESSION TO Test_User;
        GRANT DEBUG ANY PROCEDURE TO Test_User;

        Test_User

        Stored Procedure

        -- Enable DBMS_OUTPUT
        SET SERVEROUTPUT ON;

        -- Create or replace the procedure
        CREATE OR REPLACE PROCEDURE testingProcedure 
        (
            text_param IN VARCHAR2,
            number_param IN NUMBER
        )
        AS
            i NUMBER := 10;
        BEGIN
            DBMS_OUTPUT.PUT_LINE('Ahoj');
            IF i = 10 THEN
                DBMS_OUTPUT.PUT_LINE(text_param);
            ELSIF i != 10 THEN
                DBMS_OUTPUT.PUT_LINE(TO_CHAR(number_param)); -- Converting NUMBER to VARCHAR2
            ELSE
                DBMS_OUTPUT.PUT_LINE('NULL1 (UNKNOWN1)');
            END IF;
        END testingProcedure;
        /

        -- Running the stored procedure in a worksheet (as a standalone procedure)
        EXECUTE testingProcedure(text_param => 'Bleep', number_param => 42);

        -- Running the stored procedure in PL/SQL
        BEGIN
            testingProcedure(text_param => 'Bleep', number_param => 42);
        END;
        /

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
    CREATE OR REPLACE PROCEDURE mySampleProcedure AS
    BEGIN
        TESTINGFUNCTION;
    END;      
    " 
    
let internal createStoredProcedure getConnection closeConnection =
    
    try
        let connection: OracleConnection = getConnection()
                 
        try         
            use cmdCreateStoredProcedure = new OracleCommand(queryCreateStoredProcedure, connection)                             
                         
            cmdCreateStoredProcedure.ExecuteNonQuery() |> ignore
           
        finally
            closeConnection connection
    with
    | ex -> printfn "%s" ex.Message

