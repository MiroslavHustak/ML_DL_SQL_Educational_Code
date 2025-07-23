module CursorsTSQL //buzzword: WHILE (Oracle: LOOP)

    (*
       Cursors allow for explicit and sequential traversal of a result set, which can be useful in specific situations:   
       Row-by-row processing: When you need to perform operations that depend on the values of individual rows and cannot be easily expressed as a set-based operation.   
       Complex business logic: In cases where the logic is intricate and involves multiple steps, and a cursor simplifies the flow of processing.

       Cursors are not permanent database objects.
    *)


    (*
     -- Declare variables to store cursor values
     DECLARE @OrderID INT, @ProductID INT, @OperatorID INT, @MachineID INT, @Quantity DECIMAL,
             @StartTime DATETIME, @EndTime DATETIME, @Status NVARCHAR(20);
     
     -- Declare and open the cursor
     DECLARE ProductionOrderCursor CURSOR FOR
     SELECT OrderID, ProductID, OperatorID, MachineID, Quantity, StartTime, EndTime, [Status]
     FROM ProductionOrder;
     
     OPEN ProductionOrderCursor;
     
     -- Fetch the first row
     FETCH NEXT FROM ProductionOrderCursor INTO
         @OrderID, @ProductID, @OperatorID, @MachineID, @Quantity, @StartTime, @EndTime, @Status;
     
     -- Loop through the cursor and print information
     WHILE @@FETCH_STATUS = 0
     BEGIN
         -- Print information about the current row
         PRINT 'OrderID: ' + CAST(@OrderID AS NVARCHAR(10)) +
               ', ProductID: ' + CAST(@ProductID AS NVARCHAR(10)) +
               ', OperatorID: ' + CAST(@OperatorID AS NVARCHAR(10)) +
               ', MachineID: ' + CAST(@MachineID AS NVARCHAR(10)) +
               ', Quantity: ' + CAST(@Quantity AS NVARCHAR(10)) +
               ', StartTime: ' + CONVERT(NVARCHAR(30), @StartTime, 121) +
               ', EndTime: ' + CONVERT(NVARCHAR(30), @EndTime, 121) +
               ', Status: ' + @Status;
     
         -- Fetch the next row
         FETCH NEXT FROM ProductionOrderCursor INTO
             @OrderID, @ProductID, @OperatorID, @MachineID, @Quantity, @StartTime, @EndTime, @Status;
     END
     
     -- Close and deallocate the cursor
     CLOSE ProductionOrderCursor;
     DEALLOCATE ProductionOrderCursor;    
    *)