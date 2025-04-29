module Records

    (*
    -- Oracle SQL Developer

    PL/SQL (v T-SQL nesu records)

    DECLARE

      TYPE ProductionOrderRecord IS RECORD (
        OrderID INT,
        ProductID INT,
        OperatorID INT,
        MachineID INT,
        Quantity NUMBER,
        StartTime TIMESTAMP,
        EndTime TIMESTAMP,
        Status VARCHAR2(20)
      );

      -- Declare a variable of the record type
      production_order_info ProductionOrderRecord;

    BEGIN
      -- Assign values to the record
      production_order_info.OrderID := 101;
      production_order_info.ProductID := 201;
      production_order_info.OperatorID := 301;
      production_order_info.MachineID := 401;
      production_order_info.Quantity := 100;
      production_order_info.StartTime := TO_TIMESTAMP('2022-01-10 08:00:00', 'YYYY-MM-DD HH24:MI:SS');
      production_order_info.EndTime := TO_TIMESTAMP('2022-01-10 10:00:00', 'YYYY-MM-DD HH24:MI:SS');
      production_order_info.Status := 'Completed';

      -- Use the record
      DBMS_OUTPUT.PUT_LINE('Order ID: ' || production_order_info.OrderID);
      DBMS_OUTPUT.PUT_LINE('Product ID: ' || production_order_info.ProductID);
      DBMS_OUTPUT.PUT_LINE('Operator ID: ' || production_order_info.OperatorID);
      DBMS_OUTPUT.PUT_LINE('Machine ID: ' || production_order_info.MachineID);
      DBMS_OUTPUT.PUT_LINE('Quantity: ' || production_order_info.Quantity);
      DBMS_OUTPUT.PUT_LINE('Start Time: ' || TO_CHAR(production_order_info.StartTime, 'YYYY-MM-DD HH24:MI:SS'));
      DBMS_OUTPUT.PUT_LINE('End Time: ' || TO_CHAR(production_order_info.EndTime, 'YYYY-MM-DD HH24:MI:SS'));
      DBMS_OUTPUT.PUT_LINE('Status: ' || production_order_info.Status);
    END;
    /

    *)

