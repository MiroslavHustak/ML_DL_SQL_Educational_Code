module ExplicitCursors

    (*
    -- Oracle SQL Developer

    -- PL/SQL Block with Explicit Cursor
    
    DECLARE
      v_OrderID ProductionOrder.OrderID%TYPE;
      v_ProductID ProductionOrder.ProductID%TYPE;
      v_OperatorID ProductionOrder.OperatorID%TYPE;
      v_MachineID ProductionOrder.MachineID%TYPE;
      v_Quantity ProductionOrder.Quantity%TYPE;
      v_StartTime ProductionOrder.StartTime%TYPE;
      v_EndTime ProductionOrder.EndTime%TYPE;
      v_Status ProductionOrder.Status%TYPE;
    
      CURSOR order_cursor IS
        SELECT OrderID, ProductID, OperatorID, MachineID, Quantity, StartTime, EndTime, Status
        FROM ProductionOrder;
    
    BEGIN
      OPEN order_cursor;
    
      LOOP
        FETCH order_cursor INTO v_OrderID, v_ProductID, v_OperatorID, v_MachineID, v_Quantity, v_StartTime, v_EndTime, v_Status;
        
        EXIT WHEN order_cursor%NOTFOUND;
    
        DBMS_OUTPUT.PUT_LINE('Order ID: ' || v_OrderID);
        DBMS_OUTPUT.PUT_LINE('Product ID: ' || v_ProductID);
        DBMS_OUTPUT.PUT_LINE('Operator ID: ' || v_OperatorID);
        DBMS_OUTPUT.PUT_LINE('Machine ID: ' || v_MachineID);
        DBMS_OUTPUT.PUT_LINE('Quantity: ' || v_Quantity);
        DBMS_OUTPUT.PUT_LINE('Start Time: ' || TO_CHAR(v_StartTime, 'YYYY-MM-DD HH24:MI:SS'));
        DBMS_OUTPUT.PUT_LINE('End Time: ' || TO_CHAR(v_EndTime, 'YYYY-MM-DD HH24:MI:SS'));
        DBMS_OUTPUT.PUT_LINE('Status: ' || v_Status);
        DBMS_OUTPUT.PUT_LINE('----------------------');
      END LOOP;
    
      CLOSE order_cursor;
    END;
    /
    
    //jen priklad pro vysvetleni LOOP (while(true) v C#)
    DECLARE
    counter NUMBER := 1;
    BEGIN
    LOOP
      -- Print the counter value
      DBMS_OUTPUT.PUT_LINE('Counter: ' || counter);

      -- Increment the counter
      counter := counter + 1;

      -- Exit the loop when counter exceeds 5
      EXIT WHEN counter = 5;
    END LOOP;
    END;
    /

    COMMIT;
    /

    *)
