module WindowFunctions

    (*
        -- ORACLE SQL DEVELOPER


        Window functions generate a result set with additional columns derived from 
        the application of the window function over a specified window of rows. 
        The result is still a set of rows, and you can read those rows using .ExecuteReader() just like any other query.

    
        //Aggregate Window Function
        SELECT
            OperatorID,
            FirstName,
            LastName,
            JobTitle,
        AVG(LENGTH(FirstName)) OVER (PARTITION BY JobTitle) AS AvgFirstNameLength
        FROM
          Operators;

        OPERATORID FIRSTNAME                                          LASTNAME                                           JOBTITLE                                           AVGFIRSTNAMELENGTH
        ---------- -------------------------------------------------- -------------------------------------------------- -------------------------------------------------- ------------------
                 4 John                                               Doe                                                Engineer                                                            4
                 3 Bob                                                Johnson                                            Manager                                                             3
                 6 Bob                                                Johnson                                            Valcíř                                                            4.5
                 1 Jakub                                              Zválcovaný                                         Valcíř                                                            4.5
                 5 Jane                                               Smith                                              Valcíř                                                            4.5
                 2 Donald                                             Válcempřejetý                                      Valcíř                                                            4.5
    

        //Value Window Function
        SELECT
            OperatorID,
            FirstName,
            LastName,
            JobTitle,
        LAG(JobTitle) OVER (ORDER BY OperatorID) AS PreviousJobTitle
        FROM
          Operators;

    
        OPERATORID FIRSTNAME                                          LASTNAME                                           JOBTITLE                                           PREVIOUSJOBTITLE                                  
        ---------- -------------------------------------------------- -------------------------------------------------- -------------------------------------------------- --------------------------------------------------
                 1 Jakub                                              Zválcovaný                                         Valcíř                                                                                               
                 2 Donald                                             Válcempřejetý                                      Valcíř                                             Valcíř                                            
                 3 Bob                                                Johnson                                            Manager                                            Valcíř                                            
                 4 John                                               Doe                                                Engineer                                           Manager                                           
                 5 Jane                                               Smith                                              Valcíř                                             Engineer                                          
                 6 Bob                                                Johnson                                            Valcíř                                             Valcíř                                            
    
      
        //Ranking Window Function
        SELECT
            OperatorID,
            FirstName,
            LastName,
            JobTitle,
        DENSE_RANK() OVER (PARTITION BY JobTitle ORDER BY OperatorID) AS JobTitleRank
        FROM
          Operators;

          OPERATORID FIRSTNAME                                          LASTNAME                                           JOBTITLE                                           JOBTITLERANK
          ---------- -------------------------------------------------- -------------------------------------------------- -------------------------------------------------- ------------
                   4 John                                               Doe                                                Engineer                                                      1
                   3 Bob                                                Johnson                                            Manager                                                       1
                   1 Jakub                                              Zválcovaný                                         Valcíř                                                        1
                   2 Donald                                             Válcempřejetý                                      Valcíř                                                        2
                   5 Jane                                               Smith                                              Valcíř                                                        3
                   6 Bob                                                Johnson                                            Valcíř                                                        4


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

    *)

open System
open Helpers
open FsToolkit.ErrorHandling
open Oracle.ManagedDataAccess.Client
    
[<Struct>]
type private Builder2 = Builder2 with    
    member _.Bind((optionExpr, errDuCase), nextFunc) =
        match optionExpr with
        | Some value -> nextFunc value 
        | _          -> errDuCase  
    member _.Return x : 'a = x
    
let private pyramidOfDoom = Builder2
    
//ja vim, kontrola na Has.Rows (overeni existence tabulky) sice nize je, ale pro jistotu jeste predtim overeni existence tabulky
let queryExists = @"SELECT COUNT(*) FROM Operators" 
    
let querySelect3 = 
    @"
        SELECT
            OperatorID,
            FirstName,
            LastName,
            JobTitle,
        DENSE_RANK() OVER (PARTITION BY JobTitle ORDER BY OperatorID) AS JobTitleRank
        FROM
            Operators
    "
    
let internal selectValuesWF getConnection closeConnection =
    
    try
        let connection: OracleConnection = getConnection()
                 
        try                           
            use cmdExists = new OracleCommand(queryExists, connection) 
            use cmdSelect = new OracleCommand(querySelect3, connection)                                                      
                                
            pyramidOfDoom 
                {
                    //Bohuzel aji tady Oracle vraci decimal misto int 
                    let! count = cmdExists.ExecuteScalar() |> Casting.castAs<decimal>, Error "Operators table not existing"                                   
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
                                            //V pripade pouziti Oracle zkontroluj skutecny typ sloupce v .NET   
    
                                            //Jen pro overeni 
                                            let columnType = reader.GetFieldType(reader.GetOrdinal("JobTitleRank"))
                                            printfn "Column Type: %s" columnType.Name
                                                             
                                            seq 
                                                {    
                                                    //Oracle nema INT !!! Oracle by default prevede INT na NUMBER(38, 0)
                                                    //Oracle.ManagedDataAccess.Client prevede NUMBER(38, 0) v mem pripade na decimal
                                                    Casting.castAs<decimal> reader.["OperatorID"] 
                                                    |> Option.bind (fun item -> Option.filter (fun item -> not (item.Equals(String.Empty))) (Some (string item))) 
                                                    Casting.castAs<string> reader.["FirstName"]                                                                               
                                                    Casting.castAs<string> reader.["LastName"]
                                                    Casting.castAs<string> reader.["JobTitle"]   
                                                    //In Oracle databases, the DENSE_RANK() function typically returns a result of type NUMBER.
                                                    Casting.castAs<decimal> reader.["JobTitleRank"]   
                                                    |> Option.bind (fun item -> Option.filter (fun item -> not (item.Equals(String.Empty))) (Some (string item))) 
                                                } 
                                    ) |> List.ofSeq 
                                                     
                            //Pozor na nize uvedene problemy, uz jsem to nekde jinde zazil
                            //In F#, a sequence is lazily evaluated, while a list is eagerly evaluated. 
                            //This means that certain operations on sequences might not be executed until they are explicitly enumerated. 
                                             
                            //Jen pro overeni                                         
                            //getValues |> List.iter (fun item -> printfn "%A" item) 
                                                                                    
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
            closeConnection connection
    with
    | ex ->
            printfn "%s" ex.Message
            Error ex.Message

