module DDL

    //A database definition language (DDL) statement (CREATE, ALTER, or DROP).
    (*
        CREATE
        ALTER
        DROP
        TRUNCATE
        COMMENT
        RENAME
        CREATE INDEX
        DROP INDEX
        CREATE VIEW
        DROP VIEW
    *)

    (*

        ORACLE SQL DEVELOPER
        
        INT is not an Oracle Built-in Datatype but an ANSI datatype, that is transformed in the Oracle built-in datatype.    
        INT transformed to NUMBER(p,0)    
        The important point is that in the Oracle metadata USER_TAB_COLUMNS you see only the transformed Oracle built-in type.    
        Technically the transformed datatype is NUMBER(asterix, 0), which you may see in SQL Developer as the generated table DDL.
    
        This means that the maximal possible precision in Oracle (38) is used. The scale is zero, so only integers may be stored.

        //SQL

        //tabulky vytvorene pres Oracle SQL Developer

        CREATE TABLE Products (
            ProductID INT NOT NULL PRIMARY KEY,
            ProductName VARCHAR2(50),
            Description VARCHAR2(200)
        );

        CREATE TABLE Operators (
            OperatorID INT NOT NULL PRIMARY KEY,
            FirstName VARCHAR2(50),
            LastName VARCHAR2(50),
            JobTitle VARCHAR2(50)
        );

        -- with a named key
        CREATE TABLE Operators (
            OperatorID INT NOT NULL,
            FirstName VARCHAR2(50),
            LastName VARCHAR2(50),
            JobTitle VARCHAR2(50),
            CONSTRAINT PK_Operators PRIMARY KEY (OperatorID)
        );

        CREATE TABLE Machines (
            MachineID INT NOT NULL PRIMARY KEY,
            MachineName VARCHAR2(50),
            Location VARCHAR2(100)
        );

        ALTER TABLE Machines
        ADD UNIQUE (MachineName);

        DROP TABLE ProductionOrder;

        -- with named foreign keys
        CREATE TABLE ProductionOrder (
            OrderID INT NOT NULL PRIMARY KEY,
            ProductID INT NOT NULL,
            OperatorID INT NOT NULL,
            MachineID INT NOT NULL,
            Quantity NUMBER,
            StartTime TIMESTAMP,
            EndTime TIMESTAMP,
            Status VARCHAR2(20),    
            CONSTRAINT fk_Product FOREIGN KEY (ProductID) REFERENCES Products(ProductID),
            CONSTRAINT fk_Operator FOREIGN KEY (OperatorID) REFERENCES Operators(OperatorID),
            CONSTRAINT fk_Machine FOREIGN KEY (MachineID) REFERENCES Machines(MachineID)
        );

        --default-named foreign keys
        CREATE TABLE ProductionOrder (
            OrderID INT NOT NULL PRIMARY KEY,
            ProductID INT NOT NULL,
            OperatorID INT NOT NULL,
            MachineID INT NOT NULL,
            Quantity NUMBER,
            StartTime TIMESTAMP,
            EndTime TIMESTAMP,
            Status VARCHAR2(20),    
            FOREIGN KEY (ProductID) REFERENCES Products(ProductID),
            FOREIGN KEY (OperatorID) REFERENCES Operators(OperatorID),
            FOREIGN KEY (MachineID) REFERENCES Machines(MachineID)
    );
                          

        ALTER TABLE ProductionOrder DROP CONSTRAINT fk_Operator;
        ALTER TABLE Operators MODIFY OperatorID INT;
        ALTER TABLE ProductionOrder
        ADD CONSTRAINT fk_Operator FOREIGN KEY (OperatorID) REFERENCES Operators(OperatorID);
        / 
        COMMIT;

    *)