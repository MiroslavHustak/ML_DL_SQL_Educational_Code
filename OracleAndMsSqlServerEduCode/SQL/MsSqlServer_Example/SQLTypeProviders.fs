module SQLTypeProviders

open System
open System.Data
open FsToolkit.ErrorHandling

open FSharp.Data              //F# SqlCommandProvider, SqlProgrammabilityProvider
open FSharp.Data.SqlClient    //SqlCommandProvider potrebuje knihovnu FSharp.Data.SqlClient

open FSharp.Data.Sql          //SqlProvider

open Helpers

[<Literal>] 
let TypeProviderConn = @"Data Source=Misa\SQLEXPRESS;Initial Catalog=DGSada;Integrated Security=True;Encrypt=False"


//*********************************************************************
//SqlCommandProvider

//bez tryWith bloku
let internal insertOrUpdateTP1 () = 

    let str = @"N/A"

    use cmdInsert = 
        new SqlCommandProvider<
            "
            INSERT INTO SOALNew
            ( 
                [Pracovní značení],
                [Digitalizační sada],
                [Archiv],
                [Fond],
                [Číslo NAD],
                [Číslo pomůcky],
                [Inventární číslo],
                [Signatura],
                [Číslo kartonu],
                [Upřesňující indentifikátor],
                [Regest],
                [Datace vzniku],
                [Poznámka]
            )
            VALUES
            (
                @val02, @val03, @val04, @val05, @val06, @val07,
                @val08, @val09, @val10, @val11, @val12, @val13, @val14
            )     
            ", 
            TypeProviderConn>(TypeProviderConn)                    
            
    cmdInsert.Execute
        (
            val02 = str, val03 = str,
            val04 = str, val05 = str, val06 = str, 
            val07 = str, val08 = str, val09 = str, 
            val10 = str, val11 = str, val12 = str, 
            val13 = str, val14 = str
        ) |> ignore  


//****************************************************************
//SqlProgrammabilityProvider

type MySOALNewTable = SqlProgrammabilityProvider<TypeProviderConn>

let [<Literal>] query1 = //musi to byt konstanta 
    "
    SELECT * 
    FROM SOALNew 
    WHERE 
        [Archiv] NOT LIKE '%SQLTest%' 
        AND 
        [Archiv] NOT LIKE 'test' 
        AND 
        [Archiv] NOT LIKE 'N/A';
    "         

let [<Literal>] query2 = "SELECT * FROM SOALNew;" //musi to byt konstanta 
             
//bez tryWith bloku
let internal insertOrUpdateTP2 () = 
    
    use myTableData = new MySOALNewTable.dbo.Tables.SOALNew() 
                   
    let newRow = 

        let str = @"SQLTest4"

        myTableData.AddRow
            (
                Some str, Some str, 
                Some str, Some str, Some str, 
                Some str, Some str, Some str, 
                Some str, Some str, Some str, 
                Some str, Some str
            )           

    //myTableData.Update() |> ignore

    //TODO podumat, cemu SqlProgrammabilityProvider tady potrebuje SqlCommandProvider
    use loadTableData = new SqlCommandProvider<query1, TypeProviderConn, ResultType.DataReader>(TypeProviderConn)
    do loadTableData.Execute() |> myTableData.Load
     
    let numberOfColumns = myTableData.Columns.Count

    //let numberOfRows = myTableData.Rows.Count  //throws warning  

    (*
    [ 1..numberOfRows - 1 ] //rows to bere od jedne, asi proto, ze Id je od 1, TODO zjistit
    |> List.iter
        (fun row ->  
                  let row = myTableData.Rows.Item(row) //throws warning  
                  
                  [ 0..numberOfColumns - 1 ] 
                  |> List.choose
                      (fun col -> Casting.castAs<string> (string <| row.Item(col)))
                  |> List.iter                                   
                      (fun s -> printfn "%s" s)
        )
    *) 

    let dataRow = myTableData.Select() |> List.ofArray
    let numberOfRows = dataRow.Length

    [ 1..numberOfRows - 1 ] //rows to bere od jedne, asi proto, ze Id je od 1, TODO zjistit
    |> List.iter
        (fun row ->  
                  [ 0..numberOfColumns - 1 ] 
                  |> List.choose
                      (fun col -> Casting.castAs<string> (string <| (List.item row dataRow).Item(col)))
                      //(fun col -> Casting.castAs<string> <| (List.item row dataRow).Item(col)) //tohle hodi Id do None, bo je Int
                  |> List.iter                                   
                      (fun s -> printfn "%s" s)
        )

//******************************************************************************   
//SqlProvider  

[<Literal>]
let resolutionPath = @"e:\source\repos\OracleAndMsSqlServerEduCode\OracleAndMsSqlServerEduCode\bin\Debug\net8.0\Microsoft.Data.SqlClient.dll"

type sql =
    SqlDataProvider<
        ConnectionString = TypeProviderConn,
        DatabaseVendor = Common.DatabaseProviderTypes.MSSQLSERVER,
        ResolutionPath = resolutionPath,
        IndividualsAmount = 1000,        
        UseOptionTypes = Common.NullableColumnType.OPTION
    >

//bez tryWith bloku
let internal insertOrUpdateTP3 () = 

    let ctx = sql.GetDataContext()    

    let example =
        query 
            {
                for item in ctx.Dbo.SoalNew do
              
                    where (item.Id < 10)    
                    select (item.Id, item.PracovníZnačení)  

            } |> Seq.toList
    
    printfn "%A" example   

    