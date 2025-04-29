module Triggers

open FsToolkit.ErrorHandling
open Oracle.ManagedDataAccess.Client

(*
ORACLE SQL DEVELOPER

PL/SQL

SYS!!!

GRANT CREATE TRIGGER, ALTER ANY TRIGGER TO Test_User;
/

COMMIT;


Test_User

Triggers

CREATE OR REPLACE TRIGGER my_test_trigger 
BEFORE UPDATE 
OF Quantity
ON ProductionOrder 
FOR EACH ROW
BEGIN 
   IF (:NEW.Quantity + :OLD.Quantity) >= 1.0 THEN
      :NEW.Quantity := 44.0;
   ELSE
      :NEW.Quantity := 68.0;
   END IF;
END;

*)

let private queryCreateTrigger =  //jen zkusebni trigger
    "   
    CREATE OR REPLACE TRIGGER my_test_trigger2 
    BEFORE UPDATE 
    OF Quantity
    ON ProductionOrder 
    FOR EACH ROW
    BEGIN 
       IF (:NEW.Quantity + :OLD.Quantity) >= 1.0 THEN
            :NEW.Quantity := :OLD.Quantity;
       ELSE
            :NEW.Quantity := :OLD.Quantity;
       END IF;
    END;
    " 
    
let internal createTrigger getConnection closeConnection =
    
    try
        let connection: OracleConnection = getConnection()
                 
        try 
            use cmdCreateStoredProcedure = new OracleCommand(queryCreateTrigger, connection)
                          
            cmdCreateStoredProcedure.ExecuteNonQuery() |> ignore   
          
        finally
            closeConnection connection
    with
    | ex -> printfn "%s" ex.Message
          


