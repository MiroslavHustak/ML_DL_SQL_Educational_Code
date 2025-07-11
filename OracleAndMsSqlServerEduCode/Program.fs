open DML
open DML_T_SQL

open System
open System.Data.SqlClient
//<PackageReference Include="System.Data.SqlClient" Version="4.8.6" /> musel jsem rucne dodat do fsproj, nevim, cemu nebyla knihovna automaticky aplikovana. Hmmmm....
open Oracle.ManagedDataAccess.Client

open CTEs
open Triggers
open DerivedTables
open ScalarFunctions
open WindowFunctions
open StoredProcedures

open Queries
open SubQueries

open CTEsTSQL
open ITVFsTSQL
open ViewsTSQL
open TriggersTSQL
open DerivedTablesTSQL
open WindowFunctionsTSQL
open ScalarFunctionsTSQL
open StoredProceduresTSQL

open QueriesTSQL
open SubQueriesTSQL

open SQLTypeProviders

module Program = 

    //vse musi byt v try-with bloku

    [<Literal>]
    let private connectionString = 
        //"Data Source=(DESCRIPTION=(ADDRESS_LIST=(ADDRESS=(PROTOCOL=TCP)(HOST=10.0.0.2)(PORT=1521)))(CONNECT_DATA=(SERVICE_NAME=XEPDB1)));User Id=Test_User;Password=Test_User;"
        "Data Source=(DESCRIPTION=(ADDRESS_LIST=(ADDRESS=(PROTOCOL=TCP)(HOST=10.0.0.2)(PORT=1521)))(CONNECT_DATA=(SERVICE_NAME=XEPDB1)));User Id=Dictionary;Password=Dictionary;"
            
    let private getConnection () =
        let connection = new OracleConnection(connectionString)  
        connection.Open()
        connection
        
    let private closeConnection (connection: OracleConnection) =
        connection.Close()
        connection.Dispose()

    //localhost
    //let [<Literal>] private connStringTSQL = @"Data Source=localhost\SQLEXPRESS;Initial Catalog=Test_User_MSSQLS;Integrated Security=True"
    let [<Literal>] private connStringTSQL = @"Data Source=localhost\SQLEXPRESS;Initial Catalog=Dictionary_MSSQLS;Integrated Security=True"
    let [<Literal>] private connStringTP = @"Data Source=localhost\SQLEXPRESS;Initial Catalog=DGSada;Integrated Security=True"


    //shall be in a tryWith block
    let private getConnectionTSQL () =        
        let connection = new SqlConnection(connStringTSQL)
        connection.Open()
        connection
    
    let private closeConnectionTSQL (connection: SqlConnection) =
        connection.Close()
        connection.Dispose()

    let private getConnectionTP () =        
        let connection = new SqlConnection(connStringTP)
        connection.Open()
        connection
    
    let private closeConnectionTP (connection: SqlConnection) =
        connection.Close()
        connection.Dispose()
    
    //uncomment what is needed

    //insertOrUpdateProductionOrder getConnection closeConnection |> ignore

    //insertOrUpdateProducts getConnection closeConnection |> ignore

    //insertOperators getConnection closeConnection |> ignore

    //insertMachines getConnection closeConnection |> ignore

    //createStoredProcedure getConnection closeConnection |> ignore
    
    //createTrigger getConnection closeConnection |> ignore

    //createFunction getConnection closeConnection |> ignore

    //selectValuesCTE getConnection closeConnection |> ignore

    //selectValuesWF getConnection closeConnection |> ignore

    //selectValuesDT getConnection closeConnection |> ignore

    //printfn "%A" <| querySteelStructures getConnection closeConnection 
    //printfn "%A" <| queryWelds getConnection closeConnection  
    //printfn "%A" <| queryBlastFurnaces getConnection closeConnection  
    //printfn "%A" <| selectValues4Lines getConnection closeConnection  
    
    //insertProducts getConnectionTSQL closeConnectionTSQL |> ignore
    //insertOperators getConnectionTSQL closeConnectionTSQL |> ignore
    //insertMachines getConnectionTSQL closeConnectionTSQL |> ignore
    //insertProductionOrder getConnectionTSQL closeConnectionTSQL |> ignore
    //updateProductionOrder getConnectionTSQL closeConnectionTSQL 

    //selectValuesDT getConnectionTSQL closeConnectionTSQL |> ignore
    //selectValuesWFTSQL getConnectionTSQL closeConnectionTSQL |> ignore
    //selectValuesCTETSQL getConnectionTSQL closeConnectionTSQL |> ignore
    //createScalarFunctionTSQL getConnectionTSQL closeConnectionTSQL |> ignore
    //callScalarFunctionTSQL getConnectionTSQL closeConnectionTSQL |> ignore
    //createITVF getConnectionTSQL closeConnectionTSQL |> ignore
    //callITVF getConnectionTSQL closeConnectionTSQL |> ignore
    //callView getConnectionTSQL closeConnectionTSQL |> ignore
    //executeStoredProcedure getConnectionTSQL closeConnectionTSQL |> ignore 
    //createStoredProcedure getConnectionTSQL closeConnectionTSQL |> ignore 
    //createTriggerTSQL getConnectionTSQL closeConnectionTSQL |> ignore 

    //printfn "%A" <| querySteelStructuresTSQL getConnectionTSQL closeConnectionTSQL 
    //printfn "%A" <| queryWeldsTSQL getConnectionTSQL closeConnectionTSQL  
    //printfn "%A" <| queryBlastFurnacesTSQL getConnectionTSQL closeConnectionTSQL  
    //printfn "%A" <| selectValues4LinesTSQL getConnectionTSQL closeConnectionTSQL  

    //insertOrUpdateTP1 () 
    //insertOrUpdateTP2 () 
    //insertOrUpdateTP3 () |> ignore

    let private printCurrentTime () =
        let currentTime = DateTime.Now.ToString("HH:mm:ss:fff")
        printfn "Current time: %s" currentTime    
    
    printCurrentTime () 
    printfn "%s" <| String.replicate 50 "*"
    printfn "!!!!!!!!!!!! LLMs have not been tested with GPUs !!!!!!!!!!!!"
    printfn "%s" <| String.replicate 50 "*"
    printfn "Prompt: %s" NeuralNetworks.Settings.prompt
    printfn "%s" <| String.replicate 50 "*"
    printfn "Enhanced GPT-2 with GPT-3 and GPT-4-like features with normal fine-tuning" 
    NeuralNetworks.Transformer_TorchSharp4Batch.main()
    printfn "%s" <| String.replicate 50 "*"
    printfn "Enhanced GPT-2 with GPT-3 and GPT-4-like features with LoRA fine-tuning" 
    NeuralNetworks.Transformer_TorchSharp4LoRA.main()
    printfn "%s" <| String.replicate 50 "*"
    printfn "Prompt: %s" NeuralNetworks2.Settings2.prompt
    printfn "%s" <| String.replicate 50 "*"
    printfn "GPT-2" 
    NeuralNetworks2.Transformer_TorchSharp2.main()

    (*
    [1 .. 1000]
    |> List.iter
        (fun _
            ->
            NeuralNetworks.Transformer_TorchSharp4.main()
            NeuralNetworks.Transformer_TorchSharp2.main()
        )
    *)

    printCurrentTime ()
   
    //NeuralNetworks.MLP_Churn_TorchSharp.run ()
    //printfn "*************************************" 
    //NeuralNetworks.MLP_Churn_Synapses.run ()
    //NeuralNetworks.MLP_XOR_Synapses.run ()
    //NeuralNetworks.SingleLayerPerceptron2.main2 ()
    //printfn "*************************************" 
    //NeuralNetworks.SingleLayerPerceptron3.main3 ()
    //printfn "*********"
    //NeuralNetworks.SingleLayerPerceptron4.machineLearningSLP ()
    //printfn "*********"

    //printCurrentTime () 
    //MachineLearning.ManualLogisticRegression.trainAndPredictManual ()   
    //printCurrentTime ()
    //printfn "*************************************" 
    //printCurrentTime ()
    //MachineLearning.MLNETLogisticRegression.trainAndPredictML_NET ()
    //solveLinearSystem ()
    //printCurrentTime ()
    //MachineLearning.MachineLearning.machineLearningArray ()
    //printCurrentTime ()
    //printfn "*************************************" 
    //printCurrentTime ()
    //MachineLearning.TorchLinearRegression.torchSharpLR()
    //printfn "*************************************" 
    //printCurrentTime ()
    //MachineLearning.TorchLinearRegressionSequential.torchSharpLR()
    //MachineLearning.MachineLearning.machineLearningList ()
    //printCurrentTime ()
    //printfn "*************************************" 
    //printCurrentTime ()
    //MachineLearning.MachineLearning.machineLearningMLdotNET ()
    //printCurrentTime ()

    Console.ReadLine () |> ignore<string>
    