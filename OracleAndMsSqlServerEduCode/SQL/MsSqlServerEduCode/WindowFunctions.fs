module WindowFunctionsTSQL //buzzword OVER a PARTITION BY, dale jeste ROW_NUMBER(), RANK(), SUM(), or AVG() combined with the OVER clause 

    (*
        Strucna charakteristika: 
        Window functions generate a result set with additional columns derived from 
        the application of the window function over a specified window of rows. 
        The result is still a set of rows, and you can read those rows using .ExecuteReader() just like any other query.

    
        SELECT
            ProductID,
            ProductName,
            Description,
            ROW_NUMBER() OVER (ORDER BY ProductID) AS RowNum
        FROM
            Products;       

        ProductID	ProductName	Description	RowNum
        1	         Slab No.1	Rolled at Q1	1
        2	         Slab No.2	Rolled at Q2	2
        3	         Slab No.3	Rolled at Q1	3
        4	         Slab No.4	Rolled at Q1	4

         SELECT
            ProductID,
            ProductName,
            Description,
            RANK() OVER (ORDER BY Description) AS RowNum
        FROM
            Products; 
                                                     
        ProductID	ProductName	Description	RowNum
        1	       Slab No.1	Rolled at Q1	1
        3	       Slab No.3	Rolled at Q1	1
        4	       Slab No.4	Rolled at Q1	1
        2	       Slab No.2	Rolled at Q2	4 <--- RANK()

         SELECT
            ProductID,
            ProductName,
            Description,
            DENSE_RANK() OVER (ORDER BY Description) AS RowNum
         FROM
            Products; 

       ProductID	ProductName	Description	RowNum
       1	        Slab No.1	Rolled at Q1	1
       3	        Slab No.3	Rolled at Q1	1
       4	        Slab No.4	Rolled at Q1	1
       2	        Slab No.2	Rolled at Q2	2 <--- DENSE_RANK()


       SELECT
               ProductID,
               ProductName,
               Description,
               ROW_NUMBER() OVER (PARTITION BY Description ORDER BY ProductID) AS RowNum
       FROM
               Products;

               -- PARTITION BY --> the row numbers will be reset for each distinct value in the Description column:
               -- Using the PARTITION BY clause will allow you to begin counting 1 again in each partition.

       ProductID | ProductName | Description      | RowNum
       -----------+-------------+------------------+--------
       1          | Slab No.1   | Rolled at Q1     | 1
       3          | Slab No.3   | Rolled at Q1     | 2
       4          | Slab No.4   | Rolled at Q1     | 3
       2          | Slab No.2   | Rolled at Q2     | 1
       


    Window functions in SQL typically operate over a window of rows defined by the OVER clause, and they produce a result for each row within that window.
    The result is not a single scalar value but is associated with each row in the result set. The window functions are applied to a set of rows related to 

    Aggregate Window Function:
    The AVG function calculates the average length of first names for each job title.
    The result is associated with each row, and you get a result for every row in the output.

    Value Window Function:
    The LAG function retrieves the previous job title for each operator. 
    The result is associated with each row, and you get the previous job title for every operator in the result set.

    Ranking Window Function:
    The DENSE_RANK function assigns a rank to each operator within their job title partition. 
    The result is associated with each row, and you get a rank for every operator in the output.


    Let's create a sample table called Products and populate it with some data to demonstrate the usage of ROW_NUMBER() with different ORDER BY and PARTITION BY clauses in SQL Server. We'll then execute queries to show the result sets and explain the difference between them.
    
    Sample Table Creation and Data Insertion
    First, let's create and populate a sample Products table:
        
    CREATE TABLE Products (
        ProductID INT,
        ProductName VARCHAR(50),
        Description VARCHAR(50)
    );
    
    INSERT INTO Products (ProductID, ProductName, Description)
    VALUES
        (1, 'Laptop', 'Electronics'),
        (2, 'Desktop', 'Electronics'),
        (3, 'Chair', 'Furniture'),
        (4, 'Table', 'Furniture'),
        (5, 'Printer', 'Electronics'),
        (6, 'Sofa', 'Furniture');
    Now, let's execute two queries using ROW_NUMBER() with different window specifications.
    
    Example 1: ROW_NUMBER() OVER (ORDER BY ProductID) AS RowNum
    
    SELECT 
        ProductID,
        ProductName,
        ROW_NUMBER() OVER (ORDER BY ProductID) AS RowNum
    FROM 
        Products
    ORDER BY 
        ProductID;
    Result:
    
    ProductID	ProductName	RowNum
    1	Laptop	1
    2	Desktop	2
    3	Chair	3
    4	Table	4
    5	Printer	5
    6	Sofa	6
    
   
    SELECT 
        ProductID,
        ProductName,
        Description,
        ROW_NUMBER() OVER (PARTITION BY Description ORDER BY ProductID) AS RowNum
    FROM 
        Products
    ORDER BY 
        Description, ProductID;
    Result:
    
    ProductID	ProductName	Description	RowNum
    1	Laptop	Electronics	1
    2	Desktop	Electronics	2
    5	Printer	Electronics	3
    3	Chair	Furniture	1
    4	Table	Furniture	2
    6	Sofa	Furniture	3

    *)

open System
open Helpers
open FsToolkit.ErrorHandling

open System.Data.SqlClient
    
[<Struct>]
type private Builder2 = Builder2 with    
    member _.Bind((optionExpr, errDuCase), nextFunc) =
        match optionExpr with
        | Some value -> nextFunc value 
        | _          -> errDuCase  
    member _.Return x : 'a = x
    
let private pyramidOfDoom = Builder2
    
//ja vim, kontrola na Has.Rows (overeni existence tabulky) sice nize je, ale pro jistotu jeste predtim overeni existence tabulky
let queryExists = @"SELECT COUNT(*) FROM Products" 
    
let querySelect3 = 
    @"
        SELECT
            ProductID,
            ProductName,
            Description,
            RANK() OVER (ORDER BY Description) AS RowNum
        FROM
            Products; 
    "
    
let internal selectValuesWFTSQL getConnectionTSQL closeConnectionTSQL =
    
    try
        let connection: SqlConnection = getConnectionTSQL()            
                         
        try                           
            use cmdExists = new SqlCommand(queryExists, connection) 
            use cmdSelect = new SqlCommand(querySelect3, connection)                                                      
                                
            pyramidOfDoom 
                {
                    let! count = cmdExists.ExecuteScalar() |> Casting.castAs<int>, Error "Operators table not existing"                                   
                    let! _ = int count > 0 |> Option.ofBool, Error "Operators table not existing"   
                    let reader = cmdSelect.ExecuteReader()  
    
                    return Ok reader                     
                }
                                         
            |> function
                | Ok reader ->
                            let getValues =                                                 
                                Seq.initInfinite (fun _ -> reader.Read() && reader.HasRows = true)
                                |> Seq.takeWhile ((=) true) 
                                |> Seq.collect
                                    (fun _ ->
                                            //Pro overeni typu noveho sloupce
                                            let columnType = reader.GetFieldType(reader.GetOrdinal("RowNum"))
                                            printfn "Column Type: %s" columnType.Name                                                             
                                           
                                            let indexProductID = reader.GetOrdinal("ProductID")
                                            let indexProductName = reader.GetOrdinal("ProductName")
                                            let indexDescription = reader.GetOrdinal("Description")
                                            let indexRowNum = reader.GetOrdinal("RowNum")

                                            seq 
                                                {   
                                                    reader.GetInt32(indexProductID)
                                                    |> Option.ofNull  
                                                    |> Option.bind (fun item -> Option.filter (fun item -> not (item.Equals(String.Empty))) (Some (string item))) 
                                                    reader.GetString(indexProductName) |> Option.ofNull                                                                               
                                                    reader.GetString(indexDescription) |> Option.ofNull
                                                    reader.GetInt64(indexRowNum) 
                                                    |> Option.ofNull
                                                    |> Option.bind (fun item -> Option.filter (fun item -> not (item.Equals(String.Empty))) (Some (string item))) 
                                                }                                                    
                                                
                                    ) |> List.ofSeq                                                     
                                                                        
                            //Jen pro overeni                                         
                            getValues |> List.iter (fun item -> printfn "%A" item) 
                                                                                    
                            let getValues = //u Seq to dava prazdnou kolekci - viz varovani vyse                                             
                                match getValues |> List.forall _.IsSome with
                                | true  -> Ok (getValues |> List.choose id)                                       
                                | false -> Error "ReadingDbError"  
                                             
                            //Jen pro overeni                                         
                            getValues |> function Ok value -> value |> List.iter (fun item -> printfn "%s" item) | Error err -> ()
                                             
                            reader.Close() 
                            reader.Dispose()     
    
                            getValues 
                                            
                | Error err ->
                            Error err                            
        finally
            closeConnectionTSQL connection
    with
    | ex ->
            printfn "%s" ex.Message
            Error ex.Message

