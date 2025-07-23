module DML_T_SQL

open System
open System.Data.SqlClient
open FsToolkit.ErrorHandling

(*
ExecuteScalar():
Purpose: Used for executing SQL queries that return a single value (typically an aggregate value or the result of a SELECT statement with a single result).
Return Type: Object

ExecuteReader():
Purpose: Used for executing SQL queries that return a result set (multiple rows and columns).
Return Type: DbDataReader (or a specific data reader like SqlDataReader for SQL Server or OracleDataReader for Oracle).

ExecuteNonQuery():
Purpose: Used for executing SQL commands that don't return a result set, such as INSERT, UPDATE, DELETE, or DDL (Data Definition Language) statements.
Return Type: int representing the number of rows affected by the command.

When executing SQL commands, ExecuteScalar() is often used for retrieving a single value, ExecuteReader() for working with result sets,
and ExecuteNonQuery() for making changes to the database (inserting, updating, or deleting records) or executing other non-query commands.
*)

[<Struct>]
type private Builder2 = Builder2 with
    member _.Bind((optionExpr, errDuCase), nextFunc) =
        match optionExpr with
        | Some value -> nextFunc value
        | _          -> errDuCase
    member _.Return x : 'a = x

let private pyramidOfDoom = Builder2

type private Products =
    {
        productID: int
        productName: string
        description: string
    }

let private productsId1 =
    {
        productID = 1
        productName = "Slab No.1"
        description = "Rolled at Q1"
    }

let private productsId2 = 
    {
        productID = 2
        productName = "Slab No.2"
        description ="Rolled at Q2"
    }

let private productsId3 = 
    {
        productID = 3
        productName = "Slab No.3"
        description ="Rolled at Q1"
    }

let private productsId4 = 
    {
        productID = 4
        productName = "Slab No.4"
        description = "Rolled at Q1"
    }

type private Operators = 
    {
        operatorID: int
        firstName: string
        lastName: string
        jobTitle: string
    }

let private operatorsId1 = 
    {
        operatorID = 1
        firstName = "Jakub"
        lastName = "Zválcovaný"
        jobTitle = "Valcíř"
    }

let private operatorsId2 = 
    {
        operatorID = 2
        firstName = "Donald"
        lastName = "Válcempřejetý"
        jobTitle = "Valcíř"
    }

type private Machines = 
    {
        machineID: int
        machineName: string
        location: string
    }

let private machinesId1 = 
    {
        machineID = 1
        machineName = "Q1"
        location = "240"
    }

let private machinesId2 = 
    {
        machineID = 2
        machineName = "Q2"
        location = "240"
    }

type private ProductionOrder =
    {
        orderID: int
        productID: int
        operatorID: int
        machineID: int
        quantity: float
        startTime: DateTime
        endTime: DateTime
        status: string
    }

let private productionOrder101 =
    {
        orderID = 101
        productID = 1
        operatorID = 1
        machineID = 1
        quantity = 2.0
        startTime = new DateTime(2008, 5, 1, 8, 30, 52)
        endTime = new DateTime(2009, 5, 1, 8, 30, 52)
        status = "OK"
    }

let private productionOrder102 =
    {
        orderID = 102
        productID = 2
        operatorID = 2
        machineID = 2
        quantity = 8.0
        startTime = new DateTime(2008, 5, 1, 8, 30, 52)
        endTime = new DateTime(2009, 5, 1, 8, 30, 52)
        status = "OK"
    }

let private productionOrder103 =
    {
        orderID = 103
        productID = 2
        operatorID = 2
        machineID = 2
        quantity = 0.0
        startTime = new DateTime(2008, 5, 1, 8, 30, 52)
        endTime = new DateTime(2009, 5, 1, 8, 30, 52)
        status = "Open"
    }

let private productionOrder104 =
    {
        orderID = 104
        productID = 2
        operatorID = 2
        machineID = 2
        quantity = 2.0
        startTime = new DateTime(2008, 5, 1, 8, 30, 52)
        endTime = new DateTime(2009, 5, 1, 8, 30, 52)
        status = "Open"
    }

(*
In T-SQL, when you encounter conflicts like the ones you described, it means that there are foreign key constraints 
defined on the ProductionOrder table, and you are trying to delete records that are being referenced by records in other tables. 
Unlike PL/SQL, T-SQL is stricter about enforcing referential integrity.
*)

//PRODUCTS
let private queryInsert =
    "INSERT INTO Products (ProductID, ProductName, Description) VALUES (@ProductID, @ProductName, @Description)"      

let internal insertProducts getConnectionTSQL closeConnectionTSQL =

    try
        let connection = getConnectionTSQL()
                     
        try
            use cmdInsert = new SqlCommand(queryInsert, connection)  
            
            [ productsId1; productsId2; productsId3; productsId4 ]
            |> List.iter
                (fun p_Id ->
                            cmdInsert.Parameters.Clear() // Clear parameters for each iteration
                            cmdInsert.Parameters.AddWithValue("@ProductID", p_Id.productID) |> ignore
                            cmdInsert.Parameters.AddWithValue("@ProductName", p_Id.productName) |> ignore
                            cmdInsert.Parameters.AddWithValue("@Description", p_Id.description) |> ignore
                            cmdInsert.ExecuteNonQuery() |> ignore
                ) 
        finally
            closeConnectionTSQL connection
    with
    | ex -> printfn "%s" ex.Message          

//OPERATORS
let private queryInsert1 = "INSERT INTO Operators (OperatorID, FirstName, LastName, JobTitle) VALUES (@OperatorID, @FirstName, @LastName, @JobTitle)"

let internal insertOperators getConnectionTSQL closeConnectionTSQL =

    try
        let connection = getConnectionTSQL()
             
        try             
            use cmdInsert = new SqlCommand(queryInsert1, connection)   
           
            [ operatorsId1; operatorsId2 ]
            |> List.iter
                (fun o_Id ->
                        cmdInsert.Parameters.Clear() // Clear parameters for each iteration
                        cmdInsert.Parameters.AddWithValue("@OperatorID", o_Id.operatorID) |> ignore
                        cmdInsert.Parameters.AddWithValue("@FirstName", o_Id.firstName) |> ignore
                        cmdInsert.Parameters.AddWithValue("@LastName", o_Id.lastName) |> ignore
                        cmdInsert.Parameters.AddWithValue("@JobTitle", o_Id.jobTitle) |> ignore
                        cmdInsert.ExecuteNonQuery() |> ignore
                )    
        finally
            closeConnectionTSQL connection
    with
    | ex -> printfn "%s" ex.Message

//MACHINES
let private queryInsert2 = "INSERT INTO Machines (MachineID, MachineName, Location) VALUES (@MachineID, @MachineName, @Location)"

let internal insertMachines getConnectionTSQL closeConnectionTSQL =

    try
        let connection = getConnectionTSQL()
             
        try                                    
            //use cmdDeleteAll = new SqlCommand(queryDeleteAll2, connection)
            use cmdInsert = new SqlCommand(queryInsert2, connection)
                        
            //cmdDeleteAll.ExecuteNonQuery() |> ignore
           
            [ machinesId1; machinesId2 ]
            |> List.iter
                (fun m_Id ->
                        cmdInsert.Parameters.Clear() // Clear parameters for each iteration
                        cmdInsert.Parameters.AddWithValue("@MachineID", m_Id.machineID) |> ignore
                        cmdInsert.Parameters.AddWithValue("@MachineName", m_Id.machineName) |> ignore
                        cmdInsert.Parameters.AddWithValue("@Location", m_Id.location) |> ignore
                        cmdInsert.ExecuteNonQuery() |> ignore
                )         
          
        finally
            closeConnectionTSQL connection
    with
    | ex -> printfn "%s" ex.Message


let private queryInsert3 = 
    "
    INSERT INTO ProductionOrder (OrderID, ProductID, OperatorID, MachineID, Quantity, StartTime, EndTime, Status)
    VALUES (@OrderID, @ProductID, @OperatorID, @MachineID, @Quantity, @StartTime, @EndTime, @Status);
    "

let internal insertProductionOrder getConnectionTSQL closeConnectionTSQL =

    try
        let connection = getConnectionTSQL()

        try
            use cmdInsert = new SqlCommand(queryInsert3, connection)

            [ productionOrder101; productionOrder102; productionOrder103; productionOrder104 ]
            |> List.iter
                (fun p_Order ->
                    cmdInsert.Parameters.Clear()
                    cmdInsert.Parameters.AddWithValue("@OrderID", p_Order.orderID) |> ignore
                    cmdInsert.Parameters.AddWithValue("@ProductID", p_Order.productID) |> ignore
                    cmdInsert.Parameters.AddWithValue("@OperatorID", p_Order.operatorID) |> ignore
                    cmdInsert.Parameters.AddWithValue("@MachineID", p_Order.machineID) |> ignore
                    cmdInsert.Parameters.AddWithValue("@Quantity", p_Order.quantity) |> ignore
                    cmdInsert.Parameters.AddWithValue("@StartTime", p_Order.startTime) |> ignore
                    cmdInsert.Parameters.AddWithValue("@EndTime", p_Order.endTime) |> ignore
                    cmdInsert.Parameters.AddWithValue("@Status", p_Order.status) |> ignore
                    cmdInsert.ExecuteNonQuery() |> ignore
                )
        finally
            closeConnectionTSQL connection
    with
    | ex -> printfn "%s" ex.Message
    
let queryUpdate = "
    UPDATE ProductionOrder
    SET Quantity = @quantity
    WHERE OrderID = @orderID"

let internal updateProductionOrder getConnectionTSQL closeConnectionTSQL =
    try
        let connection = getConnectionTSQL()
        try
            use cmdUpdate = new SqlCommand(queryUpdate, connection)
            cmdUpdate.Parameters.AddWithValue("@orderID", 101) |> ignore
            cmdUpdate.Parameters.AddWithValue("@quantity", 110.0) |> ignore
            cmdUpdate.ExecuteNonQuery() |> ignore

            printfn "Executing query: %s" cmdUpdate.CommandText

        finally
            closeConnectionTSQL connection
    with
    | ex -> printfn "%s" ex.Message

