/*
These simple SQL queries are designed to run on the following database schema:
	Emp(eid: integer, ename: string, age: integer, salary: real)
	Works(eid: integer, did: integer, pct_time: integer)
	Dept(did: integer, dname: string, budget: real, managerid: integer)
The goal of each query is described in the comment above it
*/

/* Retrieves the name and age of each employee who works in the 'Hardware' department */
SELECT ename, age
FROM emp
INNER JOIN works
    ON emp.eid = works.eid
INNER JOIN dept
    ON works.did = dept.did
WHERE dept.dname = 'Hardware';

/* Retrieves the names of departments in which there are employees under age 25 working */
SELECT distinct dname
FROM dept
INNER JOIN works
    ON dept.did = works.did
INNER JOIN emp
    ON emp.eid = works.eid
WHERE emp.age < 25;

/* Retrieves the names of the employees who work in the 'Operations' department less than 50% of time */
SELECT ename
FROM emp
INNER JOIN works
    ON emp.eid = works.eid
INNER JOIN dept
    ON works.did = dept.did
WHERE dept.dname = 'Operations'
AND works.pct_time < 50;

/* Retrieves the names of the employees who work in a department with budget more than 1,000,000 or
work in the ‘Hardware’ department more than 20% of the time */
SELECT ename
FROM emp
INNER JOIN works
    ON emp.eid = works.eid
INNER JOIN dept
    ON works.did = dept.did
WHERE (dept.dname = 'Hardware' AND works.pct_time > 20)
OR dept.budget > 1000000
GROUP BY emp.eid
HAVING COUNT(distinct emp.eid) = 1;

/* Retrieves the names of the employees who work in a department with budget more than 1,000,000
and in the ‘Hardware’ department more than 20% of the time */
SELECT ename
FROM emp
INNER JOIN works
    ON emp.eid = works.eid
INNER JOIN dept
    ON works.did = dept.did
WHERE dept.dname = 'Hardware'
AND works.pct_time > 20
AND dept.budget > 1000000
GROUP BY emp.eid
HAVING COUNT(distinct emp.eid) = 1;

/* Retrieves the names of the departments that employ only employees over 25 years old */
SELECT dname
FROM
  (SELECT *
  FROM dept
  INNER JOIN works
      ON dept.did = works.did
  INNER JOIN emp
      ON works.eid = emp.eid) AS t1
GROUP BY dname
HAVING MIN(age) > 25;

/* Retrieves the names of the departments that employ all the employees who are over 69 years old */
SELECT distinct dname
FROM dept
    INNER JOIN (SELECT *
                FROM works
                INNER JOIN emp
                    ON works.eid = emp.eid
                WHERE emp.age > 69) AS t1
      ON dept.did = t1.did;

/*
These simple SQL queries are designed to run on the following cube database schema:
	Sales(item_name: string, color: string, size: string, number: integer)
The goal of each query is described in the comment above it
*/
/* For each item size, reports the total sales for that size, but only if the total is greater than 20 */
SELECT size, sum(number) AS total_sales
FROM sales
GROUP BY
    ROLLUP(size)
HAVING sum(number) > 20;

/* For each item size, reports the number of distinct item colors that have sold, and the total sales
for items of that size */
SELECT size,
       COUNT(distinct color) AS distinct_colors,
       sum(number) AS total_sales
FROM sales
GROUP BY
    ROLLUP(size);

/* For each item size that has sales for 2 or more distinct colors, reports the total sales for items
of that size */
SELECT size,
       sum(number) AS total_sales
FROM sales
GROUP BY
    ROLLUP(size)
HAVING COUNT(distinct color) >= 2;

/* Returns the maximum sales for any item that has color 'pastel' */
SELECT max(number) AS max_pastel_sales
FROM sales
GROUP BY
    ROLLUP(color)
HAVING color = 'pastel';

/* Finds the 'pastel' item with maximum sale and reports its item name, size, and sales */
SELECT item_name, size, number AS sales
FROM sales
  WHERE number IN
        (SELECT max(number) AS max_pastel_sales
         FROM sales
         GROUP BY
            ROLLUP(color)
         HAVING color = 'pastel');


/* Computes a data cube over attributes item_name, color, and size */
SELECT item_name, color, size
FROM sales
GROUP BY
    CUBE(item_name, color, size);