module DML

open System
open System.Data
open FsToolkit.ErrorHandling
open Oracle.ManagedDataAccess.Client

//A database manipulation language (DML) statement SELECT, INSERT, DELETE, UPDATE, MERGE, CALL

[<Struct>]
type private Builder2 = Builder2 with    
    member _.Bind((optionExpr, errDuCase), nextFunc) =
        match optionExpr with
        | Some value -> nextFunc value 
        | _          -> errDuCase  
    member _.Return x : 'a = x

let private pyramidOfDoom = Builder2

//zatim nevyuzito
type private Builder(errDuCase) =     
    member _.Bind(condition, nextFunc) =
        match condition with
        | Some value -> nextFunc value 
        | _          -> errDuCase  
    member _.Return x : 'a = x
                  
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
        productID: int  //REFERENCES Products(ProductID),
        operatorID: int //REFERENCES Operators(OperatorID),
        machineID: int  //RFERENCES Machines(MachineID)
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

//PRODUCTIONORDER
//You cannot delete records from the parent tables (Products, Operators, Machines) if there are corresponding child records in the ProductionOrder table.
//To resolve this issue, delete ProductionOrder records first (ale potom do te tabulky nestrkej znovu data, jako v mem pripade)

let private queryDeleteAll3 = "DELETE FROM ProductionOrder"

let private queryInsert3 = 
    "
    INSERT INTO ProductionOrder (OrderID, ProductID, OperatorID, MachineID, Quantity, StartTime, EndTime, Status) 
    VALUES (:OrderID, :ProductID, :OperatorID, :MachineID, :Quantity, :StartTime, :EndTime, :Status)
    "

let private queryUpdate3 = 
    "
    DECLARE
        quantity_sum NUMBER;
        addValue NUMBER;
    BEGIN
        SELECT SUM(quantity) AS quantity_sum -- AS pro nazev sloupce
        INTO quantity_sum -- INTO pro vlozeni vysledku do promenne
        FROM ProductionOrder
        WHERE quantity IS NOT NULL;

        -- Stored Procedure obsahuje UPDATE
        addValue := quantity_sum + 1.0;
        quantityAdapter(old_quantity => quantity_sum, new_quantity => addValue);
    END;
    " 
    
let internal insertOrUpdateProductionOrder getConnection closeConnection =
    
    try
        let connection: OracleConnection = getConnection()
                 
        try                            
            use cmdDeleteAll = new OracleCommand(queryDeleteAll3, connection)
            use cmdInsert = new OracleCommand(queryInsert3, connection)
            use cmdUpdate = new OracleCommand(queryUpdate3, connection) 
                                                          
            cmdDeleteAll.ExecuteNonQuery() |> ignore                        
           
            [ productionOrder101; productionOrder102; productionOrder103 ]
            |> List.iter
                (fun p_Order ->
                             cmdInsert.Parameters.Clear() // Clear parameters for each iteration
                             cmdInsert.Parameters.Add(":OrderID", OracleDbType.Int32).Value <- p_Order.orderID
                             cmdInsert.Parameters.Add(":ProductID", OracleDbType.Int32).Value <- p_Order.productID
                             cmdInsert.Parameters.Add(":OperatorID", OracleDbType.Int32).Value <- p_Order.operatorID
                             cmdInsert.Parameters.Add(":MachineID", OracleDbType.Int32).Value <- p_Order.machineID
                             cmdInsert.Parameters.Add(":Quantity", OracleDbType.Double).Value <- p_Order.quantity
                             cmdInsert.Parameters.Add(":StartTime", OracleDbType.Date).Value <- p_Order.startTime
                             cmdInsert.Parameters.Add(":EndTime", OracleDbType.Date).Value <- p_Order.endTime
                             cmdInsert.Parameters.Add(":Status", OracleDbType.Varchar2).Value <- p_Order.status
                             cmdInsert.ExecuteNonQuery() |> ignore 
                )
                                    
            cmdUpdate.ExecuteNonQuery() |> ignore   
        finally
            closeConnection connection
    with
    | ex -> printfn "%s" ex.Message
    

//PRODUCTS
let private queryDeleteAll = "DELETE FROM Products"
let private queryInsert = "INSERT INTO Products (ProductID, ProductName, Description) VALUES (:ProductID, :ProductName, :Description)"                    

let internal insertOrUpdateProducts getConnection closeConnection =

    try
        let connection: OracleConnection = getConnection()
             
        try
            use cmdDeleteAll = new OracleCommand(queryDeleteAll, connection)
            use cmdInsert = new OracleCommand(queryInsert, connection)            
           
                                                               
            cmdDeleteAll.ExecuteNonQuery() |> ignore
                        
            (
                [ productsId1; productsId2; productsId3; productsId4 ]
                |> List.iter
                    (fun p_Id ->
                              cmdInsert.Parameters.Clear() // Clear parameters for each iteration
                              cmdInsert.Parameters.Add(":ProductID", OracleDbType.Int32).Value <- p_Id.productID
                              cmdInsert.Parameters.Add(":ProductName", OracleDbType.Varchar2).Value <- p_Id.productName
                              cmdInsert.Parameters.Add(":Description", OracleDbType.Varchar2).Value <- p_Id.description
                              cmdInsert.ExecuteNonQuery() |> ignore
                    )
            ) |> ignore
           
        finally
            closeConnection connection
    with
    | ex ->
          printfn "%s" ex.Message

//insertOrUpdateProducts getConnection closeConnection |> ignore

//OPERATORS
let private queryDeleteAll1 = "DELETE FROM Operators"
let private queryInsert1 = "INSERT INTO Operators (OperatorID, FirstName, LastName, JobTitle) VALUES (:OperatorID, :FirstName, :LastName, :JobTitle)"

let internal insertOperators getConnection closeConnection =

    try
        let connection: OracleConnection = getConnection()
             
        try             
            use cmdDeleteAll = new OracleCommand(queryDeleteAll1, connection)
            use cmdInsert = new OracleCommand(queryInsert1, connection)                        
                              
            cmdDeleteAll.ExecuteNonQuery() |> ignore                       
           
            [ operatorsId1; operatorsId2 ]
            |> List.iter
                (fun o_Id ->
                        cmdInsert.Parameters.Clear() // Clear parameters for each iteration
                        cmdInsert.Parameters.Add(":OperatorID", OracleDbType.Int32).Value <- o_Id.operatorID
                        cmdInsert.Parameters.Add(":FirstName", OracleDbType.Varchar2).Value <- o_Id.firstName
                        cmdInsert.Parameters.Add(":LastName", OracleDbType.Varchar2).Value <- o_Id.lastName
                        cmdInsert.Parameters.Add(":JobTitle", OracleDbType.Varchar2).Value <- o_Id.jobTitle
                        cmdInsert.ExecuteNonQuery() |> ignore
                )          
            
        finally
            closeConnection connection
    with
    | ex -> printfn "%s" ex.Message

//insertOperators getConnection closeConnection |> ignore

//MACHINES
let private queryDeleteAll2 = "DELETE FROM Machines"
let private queryInsert2 = "INSERT INTO Machines (MachineID, MachineName, Location) VALUES (:MachineID, :MachineName, :Location)"

let internal insertMachines getConnection closeConnection =

    try
        let connection: OracleConnection = getConnection()
             
        try                                    
            use cmdDeleteAll = new OracleCommand(queryDeleteAll2, connection)
            use cmdInsert = new OracleCommand(queryInsert2, connection)
                        
            cmdDeleteAll.ExecuteNonQuery() |> ignore
           
            [ machinesId1; machinesId2 ]
            |> List.iter
                (fun m_Id ->
                        cmdInsert.Parameters.Clear() // Clear parameters for each iteration
                        cmdInsert.Parameters.Add(":MachineID", OracleDbType.Int32).Value <- m_Id.machineID
                        cmdInsert.Parameters.Add(":MachineName", OracleDbType.Varchar2).Value <- m_Id.machineName
                        cmdInsert.Parameters.Add(":Location", OracleDbType.Varchar2).Value <- m_Id.location
                        cmdInsert.ExecuteNonQuery() |> ignore
                )         
          
        finally
            closeConnection connection
    with
    | ex -> printfn "%s" ex.Message

//insertMachines getConnection closeConnection |> ignore

