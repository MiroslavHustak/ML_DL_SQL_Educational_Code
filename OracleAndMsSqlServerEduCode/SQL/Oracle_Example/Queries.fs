module Queries

open FSharp.Interop.Excel

open System
open System.Data
open FsToolkit.ErrorHandling
open Oracle.ManagedDataAccess.Client

open Helpers
open ExcelTypeProvider

let private queryUpdate3 =
    "
    BEGIN
        DELETE_NULL_ROWS(:table_name, :primary_key_column);
    END;  
    "

let internal querySteelStructures getConnection closeConnection =
        
    let queryDropSequence = 
        "
        DECLARE
          sequence_exists NUMBER;
        BEGIN
              SELECT COUNT(*) INTO sequence_exists FROM user_sequences WHERE sequence_name = 'EACH_TABLE_SEQUENCE';          
              IF sequence_exists > 0 THEN
                EXECUTE IMMEDIATE 'DROP SEQUENCE EACH_TABLE_SEQUENCE';
              END IF;
            EXCEPTION
              WHEN OTHERS THEN
                NULL; -- Ignore errors if the sequence doesn't exist
        END;
        "

    let queryDeleteAll = "DELETE FROM Steel_Structures"
       
    let queryCreateSequence = 
        "
        CREATE SEQUENCE EACH_TABLE_SEQUENCE
        START WITH 1
        INCREMENT BY 1
        NOCACHE
        NOCYCLE
        "       
         
    let queryInsert = 
         "
         INSERT INTO Steel_Structures (English, Czech, Note) 
         VALUES (:English, :Czech, :Note)
         "

    let queryUpdate1 =
        "
        UPDATE Steel_Structures
        SET Note = NULL
        WHERE Note IS NOT NULL AND INSTR(Note, 'void') > 0
        " 

    let queryUpdate2 = //nahrazeno stored procedure s nazvem DELETE_NULL_ROWS s prislusnym parametrem
        "
        DECLARE
            v_primary_key_value Steel_Structures.ID_Steel%TYPE;
            v_count NUMBER;
        BEGIN
            -- Find a row meeting the conditions where all non-primary key columns are NULL
            SELECT ID_Steel
            INTO v_primary_key_value
            FROM Steel_Structures
            WHERE English IS NULL
              AND Czech IS NULL
              AND Note IS NULL;
    
            -- Check if a row meeting the conditions is found
            IF v_primary_key_value IS NOT NULL THEN
                -- Delete the row using the dynamically determined primary key value
                DELETE FROM Steel_Structures
                WHERE ID_Steel = v_primary_key_value;
            ELSE
                -- TODO: Handle the case when no row is found
                NULL; 
           END IF;
        END;
        "     
        
    let query = 
        [
            queryDropSequence
            queryDeleteAll
            queryCreateSequence
            queryInsert
            queryUpdate1
            queryUpdate3
        ]

    let list = 
        [
            "Steel_Structures"
            "ID_STEEL"
        ]
    
    let path = "e:\\source\\repos\\OracleDB_Excel_Files\\Slovnicek AJ new steel structures.xlsx"

    insertOrUpdateDictionary getConnection closeConnection query path list

let internal queryWelds getConnection closeConnection =
    
    let queryDropSequence = 
        "
        DECLARE
          sequence_exists NUMBER;
        BEGIN
              SELECT COUNT(*) INTO sequence_exists FROM user_sequences WHERE sequence_name = 'EACH_TABLE_SEQUENCE';          
              IF sequence_exists > 0 THEN
                EXECUTE IMMEDIATE 'DROP SEQUENCE EACH_TABLE_SEQUENCE';
              END IF;
            EXCEPTION
              WHEN OTHERS THEN
                NULL; -- Ignore errors if the sequence doesn't exist
        END;
        "

    let queryDeleteAll = "DELETE FROM WELDS"
   
    let queryCreateSequence = 
        "
        CREATE SEQUENCE EACH_TABLE_SEQUENCE
        START WITH 1
        INCREMENT BY 1
        NOCACHE
        NOCYCLE
        "       
     
    let queryInsert = 
        (*
            When using sequences to generate unique identifiers for primary keys, you don't have to manually manage the assignment of IDs during insertion. 
            The database takes care of generating a unique ID for each new record automatically.
            The cmdInsert statement inserts rows into the table, and you don't explicitly provide a value for the ID column.
            With a sequence-based setup, the database ensures that a unique ID is assigned to each row, and you don't have to worry about specifying or managing those IDs
            in your application code during insertion. It becomes particularly beneficial when dealing with updates or deletions. With a known and automatically assigned ID, 
            you can easily construct update or delete statements based on the primary key without needing to track or remember specific IDs for each row.
        *)
         "
         INSERT INTO WELDS (English, Czech, Note) 
         VALUES (:English, :Czech, :Note)
         "

    let queryUpdate1 =
        "
        UPDATE WELDS
        SET Note = NULL
        WHERE Note IS NOT NULL AND INSTR(Note, 'void') > 0
        " 

    let queryUpdate2 = //nahrazeno stored procedure s nazvem DELETE_NULL_ROWS s prislusnym parametrem
        "
        DECLARE
            v_primary_key_value WELDS.ID_WELD%TYPE;
            v_count NUMBER;
        BEGIN
            -- Find a row meeting the conditions where all non-primary key columns are NULL
            SELECT ID_WELD
            INTO v_primary_key_value
            FROM WELDS
            WHERE English IS NULL
              AND Czech IS NULL
              AND Note IS NULL;

            -- Check if a row meeting the conditions is found
            IF v_primary_key_value IS NOT NULL THEN
                -- Delete the row using the dynamically determined primary key value
                DELETE FROM WELDS
                WHERE ID_WELD = v_primary_key_value;
            ELSE
                -- TODO: Handle the case when no row is found
                NULL; 
           END IF;
        END;
        "
                      
    let query = 
        [
            queryDropSequence
            queryDeleteAll
            queryCreateSequence
            queryInsert
            queryUpdate1
            queryUpdate3
        ]

    let list = 
        [
            "WELDS"
            "ID_WELD"
        ]

    let path = "e:\\source\\repos\\OracleDB_Excel_Files\\Slovnicek AJ new welding.xlsx"

    insertOrUpdateDictionary getConnection closeConnection query path list

let internal queryBlastFurnaces getConnection closeConnection =
    
    let queryDropSequence = 
        "
        DECLARE
          sequence_exists NUMBER;
        BEGIN
              SELECT COUNT(*) INTO sequence_exists FROM user_sequences WHERE sequence_name = 'EACH_TABLE_SEQUENCE';          
              IF sequence_exists > 0 THEN
                EXECUTE IMMEDIATE 'DROP SEQUENCE EACH_TABLE_SEQUENCE';
              END IF;
            EXCEPTION
              WHEN OTHERS THEN
                NULL; -- Ignore errors if the sequence doesn't exist
        END;
        "

    let queryDeleteAll = "DELETE FROM BLAST_FURNACES"
   
    let queryCreateSequence = 
        "
        CREATE SEQUENCE EACH_TABLE_SEQUENCE
        START WITH 1
        INCREMENT BY 1
        NOCACHE
        NOCYCLE
        "       
     
    let queryInsert = 
         "
         INSERT INTO BLAST_FURNACES (English, Czech, Note) 
         VALUES (:English, :Czech, :Note)
         "

    let queryUpdate1 =
        "
        UPDATE BLAST_FURNACES
        SET Note = NULL
        WHERE Note IS NOT NULL AND INSTR(Note, 'void') > 0
        " 

    let queryUpdate2 = //nahrazeno stored procedure s nazvem DELETE_NULL_ROWS s prislusnym parametrem
        "
        DECLARE
            v_primary_key_value BLAST_FURNACES.ID_BF%TYPE;
            v_count NUMBER;
        BEGIN
            -- Find a row meeting the conditions where all non-primary key columns are NULL
            SELECT ID_BF
            INTO v_primary_key_value
            FROM BLAST_FURNACES
            WHERE English IS NULL
              AND Czech IS NULL
              AND Note IS NULL;

            -- Check if a row meeting the conditions is found
            IF v_primary_key_value IS NOT NULL THEN
                -- Delete the row using the dynamically determined primary key value
                DELETE FROM BLAST_FURNACES
                WHERE ID_BF = v_primary_key_value;
            ELSE
                -- TODO: Handle the case when no row is found
                NULL; 
           END IF;
        END;
        "
                
    let query = 
        [
            queryDropSequence
            queryDeleteAll
            queryCreateSequence
            queryInsert
            queryUpdate1
            queryUpdate3
        ]

    let list = 
        [
            "BLAST_FURNACES"
            "ID_BF"
        ]

    let path = "e:\\source\\repos\\OracleDB_Excel_Files\\Slovnicek AJ new.xlsx"

    insertOrUpdateDictionary getConnection closeConnection query path list

    (*
    sys
    GRANT CREATE TRIGGER TO Dictionary;
    
    --**************************************
        
    -- Drop trigger
    DROP TRIGGER Steel_Structures_Trigger;
    -- Drop sequence
    DECLARE
      sequence_exi
    -- Drop table
    DROP TABLE Steel_Structures;
    
    -- Drop sequence
    DECLARE
      sequence_exists NUMBER;
    BEGIN
      SELECT COUNT(*) INTO sequence_exists FROM user_sequences WHERE sequence_name = 'EACH_TABLE_SEQUENCE';
      
      IF sequence_exists > 0 THEN
        EXECUTE IMMEDIATE 'DROP SEQUENCE Each_Table_Sequence';
      END IF;
    EXCEPTION
      WHEN OTHERS THEN
        NULL; -- Ignore errors if the sequence doesn't exist
    END;
    /
    
    CREATE SEQUENCE Each_Table_Sequence
        START WITH 1
        INCREMENT BY 1
        NOCACHE
        NOCYCLE;
        
    CREATE TABLE Steel_Structures
    (
        ID_Steel NUMBER PRIMARY KEY NOT NULL,
        English NVARCHAR2(100),
        Czech NVARCHAR2(100),
        Note NVARCHAR2(1000)
    );
    
    -- To use the sequence for automatic numbering
    CREATE TRIGGER Steel_Structures_Trigger
        BEFORE INSERT ON Steel_Structures
        FOR EACH ROW
        BEGIN
            SELECT Each_Table_Sequence.NEXTVAL
            INTO   :new.ID_Steel
            FROM   dual;
        END;
    /
    COMMIT;
    
    UPDATE Steel_Structures
    SET Note = NULL
    WHERE Note IS NOT NULL AND INSTR(Note, 'void') > 0;
    
    -- The INSTR function is used to check if the string "void" exists in the Note column.
    
    UPDATE Steel_Structures
    SET Note = REPLACE(Note, 'void', NULL)
    WHERE Note IS NOT NULL AND INSTR(Note, 'void') > 0;


    CREATE OR REPLACE PROCEDURE DELETE_NULL_ROWS
    (
        p_table_name IN VARCHAR2,
        p_primary_key_column IN VARCHAR2
    ) 
    AS
        v_primary_key_value NUMBER;
        v_sql_query VARCHAR2(1000); -- Adjust the size based on your needs
    BEGIN
        -- Build the dynamic SQL query with a bind variable for the table name
        v_sql_query :=
            'SELECT ' || p_primary_key_column ||
            ' FROM ' || p_table_name ||
            ' WHERE English IS NULL AND Czech IS NULL AND Note IS NULL';

        -- Find a row meeting the conditions where all non-primary key columns are NULL
        EXECUTE IMMEDIATE v_sql_query INTO v_primary_key_value;

        -- Check if a row meeting the conditions is found
        IF v_primary_key_value IS NOT NULL THEN
            -- Build the dynamic SQL delete statement
            v_sql_query :=
                'DELETE FROM ' || p_table_name ||
                ' WHERE ' || p_primary_key_column || ' = :1';
                -- :1 acts as a placeholder for the actual value that will be substituted during execution. 
                -- The USING clause ensures that the correct value is associated with the bind variable.

            -- Delete the row using the dynamically determined primary key value
            EXECUTE IMMEDIATE v_sql_query USING v_primary_key_value;
        ELSE
            -- TODO: Handle the case when no row is found
            NULL;
        END IF;
    END DELETE_NULL_ROWS;
    
    *)


