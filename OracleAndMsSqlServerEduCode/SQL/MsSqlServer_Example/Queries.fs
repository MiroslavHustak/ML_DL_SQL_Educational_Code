module QueriesTSQL

open FSharp.Interop.Excel

open System
open System.Data
open FsToolkit.ErrorHandling

open System.Data.SqlClient

open Helpers
open ExcelTypeProviderTSQL

let private queryDeleteNullRows =
    "
    BEGIN
        EXEC DELETE_NULL_ROWS @table_name, @primary_key_column;
    END;  
    "

let internal querySteelStructuresTSQL getConnectionTSQL closeConnectionTSQL =
        
    let queryDropSequence = 
        "
        DECLARE @sequence_exists INT;
        SELECT @sequence_exists = COUNT(*) FROM sys.sequences WHERE name = 'EACH_TABLE_SEQUENCE';          
        IF @sequence_exists > 0
        BEGIN
            EXEC sp_executesql N'DROP SEQUENCE EACH_TABLE_SEQUENCE';
        END;
        "

    let queryDeleteAll = "DELETE FROM Steel_Structures"
       
    let queryCreateSequence = //AUTO_INCREMENT by jinak melo stacit pro T-SQL (poze Oracle nema AUTO_INCREMENT)
        "
        CREATE SEQUENCE EACH_TABLE_SEQUENCE
        START WITH 1
        INCREMENT BY 1
        MINVALUE 1
        MAXVALUE 1000000  
        NO CACHE
        CYCLE;
        "       
         
    let queryInsert = 
         "
         INSERT INTO Steel_Structures (ID_Steel, English, Czech, Note) 
         VALUES (NEXT VALUE FOR Each_Table_Sequence, @English, @Czech, @Note);
         "

    let queryUpdate =
        "
        UPDATE Steel_Structures
        SET Note = NULL
        WHERE Note IS NOT NULL AND CHARINDEX('void', Note) > 0;
        " 
    
    let query = 
        [
            queryDropSequence
            queryDeleteAll
            queryCreateSequence
            queryInsert
            queryUpdate
            queryDeleteNullRows
        ]

    let list = 
        [
            "Steel_Structures"
            "ID_Steel"
        ]
    
    let path = "e:\\source\\repos\\OracleDB_Excel_Files\\Slovnicek AJ new steel structures.xlsx"

    insertOrUpdateDictionaryTSQL getConnectionTSQL closeConnectionTSQL query path list

let internal queryWeldsTSQL getConnectionTSQL closeConnectionTSQL =
    
    let queryDropSequence = 
        "
        DECLARE @sequence_exists INT;
        SELECT @sequence_exists = COUNT(*) FROM sys.sequences WHERE name = 'EACH_TABLE_SEQUENCE';          
        IF @sequence_exists > 0
        BEGIN
            EXEC sp_executesql N'DROP SEQUENCE EACH_TABLE_SEQUENCE';
        END;
        "

    let queryDeleteAll = "DELETE FROM Welds"
   
    let queryCreateSequence = 
        "
        CREATE SEQUENCE EACH_TABLE_SEQUENCE
        START WITH 1
        INCREMENT BY 1
        MINVALUE 1
        MAXVALUE 1000000  
        NO CACHE
        CYCLE;
        "       
     
    let queryInsert = 
         "
         INSERT INTO Welds (ID_Weld, English, Czech, Note) 
         VALUES (NEXT VALUE FOR Each_Table_Sequence, @English, @Czech, @Note);
         "

    let queryUpdate1 =
        "
        UPDATE Welds
        SET Note = NULL
        WHERE Note IS NOT NULL AND CHARINDEX('void', Note) > 0;
        " 
            
    let query = 
        [
            queryDropSequence
            queryDeleteAll
            queryCreateSequence
            queryInsert
            queryUpdate1
            queryDeleteNullRows
        ]

    let list = 
        [
            "Welds"
            "ID_Weld"
        ]

    let path = "e:\\source\\repos\\OracleDB_Excel_Files\\Slovnicek AJ new welding.xlsx"

    insertOrUpdateDictionaryTSQL getConnectionTSQL closeConnectionTSQL query path list

let internal queryBlastFurnacesTSQL getConnectionTSQL closeConnectionTSQL =
    
    let queryDropSequence = 
        "
        DECLARE @sequence_exists INT;
        SELECT @sequence_exists = COUNT(*) FROM sys.sequences WHERE name = 'EACH_TABLE_SEQUENCE';          
        IF @sequence_exists > 0
        BEGIN
            EXEC sp_executesql N'DROP SEQUENCE EACH_TABLE_SEQUENCE';
        END;
        "

    let queryDeleteAll = "DELETE FROM Blast_Furnaces"
   
    let queryCreateSequence = 
        "
        CREATE SEQUENCE EACH_TABLE_SEQUENCE
        START WITH 1
        INCREMENT BY 1
        MINVALUE 1
        MAXVALUE 1000000  
        NO CACHE
        CYCLE;
        "       
     
    let queryInsert = 
         "
         INSERT INTO Blast_Furnaces (ID_BF, English, Czech, Note) 
         VALUES (NEXT VALUE FOR Each_Table_Sequence, @English, @Czech, @Note);
         "

    let queryUpdate1 =
        "
        UPDATE Blast_Furnaces
        SET Note = NULL
        WHERE Note IS NOT NULL AND CHARINDEX('void', Note) > 0;
        " 
        
    let query = 
        [
            queryDropSequence
            queryDeleteAll
            queryCreateSequence
            queryInsert
            queryUpdate1
            queryDeleteNullRows
        ]

    let list = 
        [
            "Blast_Furnaces"
            "ID_BF"
        ]

    let path = "e:\\source\\repos\\OracleDB_Excel_Files\\Slovnicek AJ new.xlsx"

    insertOrUpdateDictionaryTSQL getConnectionTSQL closeConnectionTSQL query path list

(*   

    USE [Dictionary_MSSQLS]
    -- Create sequence
    CREATE SEQUENCE EACH_TABLE_SEQUENCE
    START WITH 1
    INCREMENT BY 1
    MINVALUE 1
    MAXVALUE 1000000  
    NO CACHE
    CYCLE;

    USE [Dictionary_MSSQLS]

    -- Create table
    -- Drop the existing table if it exists
    IF OBJECT_ID('Blast_Furnaces', 'U') IS NOT NULL
        DROP TABLE Blast_Furnaces;

    -- Create the table with modifications
    CREATE TABLE Blast_Furnaces
    (
        ID_BF INT PRIMARY KEY NOT NULL,
        English NVARCHAR(100) NULL,
        Czech NVARCHAR(100) NULL,
        Note NVARCHAR(1000) NULL
    );

    -- Drop the existing table if it exists
    IF OBJECT_ID('Steel_Structures', 'U') IS NOT NULL
        DROP TABLE Steel_Structures;

    -- Create the table with modifications
    CREATE TABLE Steel_Structures
    (
        ID_Steel INT PRIMARY KEY NOT NULL,
        English NVARCHAR(100) NULL,
        Czech NVARCHAR(100) NULL,
        Note NVARCHAR(1000) NULL
    );

    -- Drop the existing table if it exists
    IF OBJECT_ID('Welds', 'U') IS NOT NULL
        DROP TABLE Welds;

    -- Create the table with modifications
    CREATE TABLE Welds
    (
        ID_Weld INT PRIMARY KEY NOT NULL,
        English NVARCHAR(100) NULL,
        Czech NVARCHAR(100) NULL,
        Note NVARCHAR(1000) NULL
    );     

    CREATE TRIGGER Blast_Furnaces_Trigger
    ON Blast_Furnaces
    AFTER INSERT
    AS
    BEGIN
        SET NOCOUNT ON;    
        UPDATE [Dictionary_MSSQLS].[dbo].[Blast_Furnaces]
        SET Note = "TriggeredValue"
        WHERE ID_BF = 1;
    END;
 
    -- Stored Procedure DELETE_NULL_ROWS
    IF OBJECT_ID('DELETE_NULL_ROWS', 'P') IS NOT NULL
        DROP PROCEDURE DELETE_NULL_ROWS;
    GO

    CREATE PROCEDURE DELETE_NULL_ROWS
    @p_table_name NVARCHAR(100),
    @p_primary_key_column NVARCHAR(100)
    AS
    BEGIN
        DECLARE @v_primary_key_value INT;
        DECLARE @v_sql_query NVARCHAR(1000); -- Adjust the size based on your needs

        -- Build the dynamic SQL query with a parameter for the table name
        SET @v_sql_query =
            'SELECT ' + @p_primary_key_column +
            ' FROM ' + @p_table_name +
            ' WHERE English IS NULL AND Czech IS NULL AND Note IS NULL';

        -- Find a row meeting the conditions where all non-primary key columns are NULL
        EXEC sp_executesql @v_sql_query, N'@v_primary_key_value INT OUTPUT', @v_primary_key_value OUTPUT;

        -- Check if a row meeting the conditions is found
        IF @v_primary_key_value IS NOT NULL
        BEGIN
            -- Build the dynamic SQL delete statement
            SET @v_sql_query =
                'DELETE FROM ' + @p_table_name +
                ' WHERE ' + @p_primary_key_column + ' = @v_primary_key_value';

            -- Delete the row using the dynamically determined primary key value
            EXEC sp_executesql @v_sql_query, N'@v_primary_key_value INT', @v_primary_key_value;
        END        
    END;
    *)


    (*   
    SELECT 
        MIN([ID_BF]) AS [ID_BF],
        [English],
        MIN([Czech]) AS [Czech],
        MIN([Note]) AS [Note]
    FROM [Dictionary_MSSQLS].[dbo].[Blast_Furnaces]
    GROUP BY [English];
        
    SELECT
        MIN([ID_Steel]) AS [ID_Steel], 
        [English],
        MIN([Czech]) AS [Czech],
        MIN([Note]) AS [Note]
    FROM [Dictionary_MSSQLS].[dbo].[Steel_Structures]
    GROUP BY [English];

    SELECT
        MIN([ID_Weld]) AS [ID_Weld], 
        [English],
        MIN([Czech]) AS [Czech],
        MIN([Note]) AS [Note]
    FROM [Dictionary_MSSQLS].[dbo].[Welds]
    GROUP BY [English];

    -- Drop trigger
    IF OBJECT_ID('Steel_Structures_Trigger', 'TR') IS NOT NULL
        DROP TRIGGER Steel_Structures_Trigger;
        
    -- Drop sequence
    IF OBJECT_ID('Each_Table_Sequence', 'SO') IS NOT NULL
        DROP SEQUENCE Each_Table_Sequence;
    
    -- Drop table
    IF OBJECT_ID('Steel_Structures', 'U') IS NOT NULL
        DROP TABLE Steel_Structures;
    
    -- Drop sequence
    IF OBJECT_ID('Each_Table_Sequence', 'SO') IS NOT NULL
    BEGIN
        DECLARE @sequence_exists INT;
        SELECT @sequence_exists = COUNT(*) FROM sys.sequences WHERE name = 'Each_Table_Sequence';
        
        IF @sequence_exists > 0
        BEGIN
            EXEC('DROP SEQUENCE Each_Table_Sequence');
        END;
    END; 
    
    -- Update queries
    UPDATE Steel_Structures
    SET Note = NULL
    WHERE Note IS NOT NULL AND CHARINDEX('void', Note) > 0;
    
    UPDATE Steel_Structures
    SET Note = REPLACE(Note, 'void', NULL)
    WHERE Note IS NOT NULL AND CHARINDEX('void', Note) > 0;
    
    -- Stored Procedure DELETE_NULL_ROWS
    IF OBJECT_ID('DELETE_NULL_ROWS', 'P') IS NOT NULL
        DROP PROCEDURE DELETE_NULL_ROWS;
    GO    
    *)


