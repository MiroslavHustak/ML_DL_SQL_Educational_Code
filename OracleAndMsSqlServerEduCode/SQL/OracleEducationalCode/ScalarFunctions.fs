module ScalarFunctions

open FsToolkit.ErrorHandling
open Oracle.ManagedDataAccess.Client

    (*
        -- Oracle SQL Developer
        --PL/SQL
        
        CREATE OR REPLACE FUNCTION TESTINGFUNCTION 
        RETURN NUMBER AS --standalone (AS), vraci hodnotu
           total NUMBER := 100; 
        BEGIN
           RETURN total; 
        END TESTINGFUNCTION;

        -- Running the function in a worksheet (as a standalone function)
        SELECT TESTINGFUNCTION() FROM DUAL;

        -- Running the function in PL/SQL

        CREATE OR REPLACE FUNCTION TESTINGFUNCTION 
        RETURN NUMBER IS --from a PL/SQL block (IS) , vraci hodnotu
           total NUMBER := 100; 
        BEGIN
           RETURN total; 
        END TESTINGFUNCTION;
          
    *)

let private queryCreateFunction =  //jen zkusebni function
    "   
    CREATE OR REPLACE FUNCTION TESTINGFUNCTION 
    RETURN NUMBER IS --from a PL/SQL block (IS) , vraci hodnotu
       total NUMBER := 100; 
    BEGIN
       RETURN total; 
    END TESTINGFUNCTION;
    " 
    
let internal createScalarFunction getConnection closeConnection =
    
    try
        let connection: OracleConnection = getConnection()
                 
        try   
            use cmdCreateFunction = new OracleCommand(queryCreateFunction, connection)                            
                         
            cmdCreateFunction.ExecuteNonQuery() |> ignore 
          
        finally
            closeConnection connection
    with
    | ex -> printfn "%s" ex.Message
