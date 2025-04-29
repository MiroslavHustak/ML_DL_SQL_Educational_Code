module PipelinedFunctions
(*
//pouze PL/SQL

CREATE TYPE EmployeeRecord AS OBJECT (
id NUMBER,
name VARCHAR2(50),
salary NUMBER,
departmentId NUMBER
);

CREATE TYPE EmployeeTable AS TABLE OF EmployeeRecord;

CREATE OR REPLACE FUNCTION MyFunction (cid IN NUMBER) RETURN EmployeeTable PIPELINED IS
BEGIN
FOR emp_rec IN (SELECT id, name, salary, departmentId FROM Employee1 WHERE salary > 70000) LOOP
    PIPE ROW (EmployeeRecord(emp_rec.id, emp_rec.name, emp_rec.salary, emp_rec.departmentId));
END LOOP;

RETURN;
END MyFunction;
/

*)
