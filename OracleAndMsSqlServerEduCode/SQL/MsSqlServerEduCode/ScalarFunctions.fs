module ScalarFunctionsTSQL //buzzword FUNCTION, RETURN, strucna charakteristika: vraci hodnotu (scalar value)

open FsToolkit.ErrorHandling
open System.Data.SqlClient
open Helpers

    (*
        CREATE FUNCTION dbo.TESTINGFUNCTION()
        RETURNS INT
        AS
        BEGIN
            DECLARE @total INT;
            SET @total = 100;
            RETURN @total;
        END;

       -- Calling the Scalar Function
       DECLARE @resultFromFunction INT;
       
       -- Example 1: Calling in a SELECT statement
       SELECT dbo.ScalarFunctionExample(5, 3) AS Result;
       
       -- Example 2: Assigning the result to a variable
       SET @resultFromFunction = dbo.ScalarFunctionExample(8, 4);
       PRINT 'Result from function: ' + CAST(@resultFromFunction AS VARCHAR);
    *)

let private queryCreateFunction =  
    //jen zkusebni function, a jo, vytvorilo se to do dane db (connection string stacil)
    //a samo to poznalo scalar function a skoncilo to v danem adresari
    "   
    CREATE FUNCTION dbo.TESTINGFUNCTION()
    RETURNS INT
    AS
    BEGIN
        DECLARE @total INT;
        SET @total = 100;
        RETURN @total;
    END;
    " 
    
let internal createScalarFunctionTSQL getConnectionTSQL closeConnectionTSQL =
    
    try
        let connection: SqlConnection = getConnectionTSQL()
                 
        try   
            use cmdCreateFunction = new SqlCommand(queryCreateFunction, connection)                            
                         
            cmdCreateFunction.ExecuteNonQuery()  |> ignore 
          
        finally
            closeConnectionTSQL connection
    with
    | ex -> printfn "%s" ex.Message


let private queryCallFunction =  
    //jen zkusebni function, a jo, vytvorilo se to do dane db (connection string stacil)
    //a samo to poznalo scalar function a skoncilo to v danem adresari
    "   
    CREATE FUNCTION dbo.TESTINGFUNCTION()
    RETURNS INT
    AS
    BEGIN
        DECLARE @total INT;
        SET @total = 100;
        RETURN @total;
    END;
    " 

let internal callScalarFunctionTSQL getConnectionTSQL closeConnectionTSQL =
    
    try
        let connection: SqlConnection = getConnectionTSQL()
                 
        try  
            
            use cmdCallFunction = new SqlCommand("SELECT dbo.TESTINGFUNCTION() AS Result", connection)
            let result = Casting.castAs<int> <| cmdCallFunction.ExecuteScalar()

            printfn "Result from function: %A" result
          
        finally
            closeConnectionTSQL connection
    with
    | ex -> printfn "%s" ex.Message


(*
In T-SQL (Transact-SQL), a FUNCTION is a database object that encapsulates a specific logic or computation and can be used to perform operations on data. Functions in T-SQL can be classified into two main types: scalar functions and table-valued functions.

Scalar Functions:
Scalar functions return a single scalar (single-valued) result. They can be used in various parts of a SQL statement where an expression is expected, such as in SELECT, WHERE, and ORDER BY clauses. Here's a simple example:

sql
Copy code
-- Scalar Function Example
CREATE FUNCTION dbo.AddNumbers (@num1 INT, @num2 INT)
RETURNS INT
AS
BEGIN
    DECLARE @result INT;
    SET @result = @num1 + @num2;
    RETURN @result;
END;

-- Using the Scalar Function
SELECT dbo.AddNumbers(5, 3) AS SumResult;
In this example, the AddNumbers function takes two input parameters (@num1 and @num2), adds them together, and returns the result as a single integer.

Table-Valued Functions:
Table-valued functions return a table-like result set, and they can be used in the FROM clause of a SELECT statement or joined with other tables. There are two types of table-valued functions: inline table-valued functions and multi-statement table-valued functions.

Inline Table-Valued Function:

sql
Copy code
-- Inline Table-Valued Function Example
CREATE FUNCTION dbo.GetEmployeesByDepartment (@deptID INT)
RETURNS TABLE
AS
RETURN (
    SELECT EmployeeID, FirstName, LastName
    FROM Employees
    WHERE DepartmentID = @deptID
);

-- Using the Inline Table-Valued Function
SELECT * FROM dbo.GetEmployeesByDepartment(1);
In this example, the GetEmployeesByDepartment function takes a department ID as a parameter and returns a table of employees in that department.

Multi-Statement Table-Valued Function:

sql
Copy code
-- Multi-Statement Table-Valued Function Example
CREATE FUNCTION dbo.GetProductDetails ()
RETURNS @ProductTable TABLE (
    ProductID INT,
    ProductName NVARCHAR(255),
    UnitPrice MONEY
)
AS
BEGIN
    INSERT INTO @ProductTable
    SELECT ProductID, ProductName, UnitPrice
    FROM Products;

    RETURN;
END;

-- Using the Multi-Statement Table-Valued Function
SELECT * FROM dbo.GetProductDetails();
In this example, the GetProductDetails function returns a table with product details using a table variable (@ProductTable).
*)
