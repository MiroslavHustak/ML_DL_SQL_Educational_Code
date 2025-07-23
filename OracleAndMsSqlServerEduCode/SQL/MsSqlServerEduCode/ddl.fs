module DDL_T_SQL

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
        //SSMS SQL Server Management Studio

        //tabulky vytvorene pres SSMS

        CREATE TABLE Products (
            ProductID INT NOT NULL PRIMARY KEY,
            ProductName NVARCHAR(50),
            Description NVARCHAR(200)
        );

        CREATE TABLE Products (
            ProductID INT NOT NULL PRIMARY KEY,
            ProductName NVARCHAR(50),
            Description NVARCHAR(200)
        );

        -- with a named key
        CREATE TABLE Operators (
            OperatorID INT NOT NULL,
            FirstName NVARCHAR(50),
            LastName NVARCHAR(50),
            JobTitle NVARCHAR(50),
            CONSTRAINT PK_Operators PRIMARY KEY (OperatorID)
        );

        CREATE TABLE Machines (
            MachineID INT NOT NULL PRIMARY KEY,
            MachineName NVARCHAR(50),
            Location NVARCHAR(100)
        );

        CREATE TABLE ProductionOrder (
            OrderID INT NOT NULL PRIMARY KEY,
            ProductID INT NOT NULL,
            OperatorID INT NOT NULL,
            MachineID INT NOT NULL,
            Quantity DECIMAL, -- or NUMERIC
            StartTime DATETIME,
            EndTime DATETIME,
            Status NVARCHAR(20),    
            CONSTRAINT fk_Product FOREIGN KEY (ProductID) REFERENCES Products(ProductID),
            CONSTRAINT fk_Operator FOREIGN KEY (OperatorID) REFERENCES Operators(OperatorID),
            CONSTRAINT fk_Machine FOREIGN KEY (MachineID) REFERENCES Machines(MachineID)
        );
    *)

    (*
    ALTER TABLE ProductionOrder
        ADD CONSTRAINT fk_Operator FOREIGN KEY (OperatorID) REFERENCES Operators(OperatorID);   
    *)