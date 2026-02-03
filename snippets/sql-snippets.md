# SQL Snippets

Common SQL queries and patterns for quick reference.

## 📊 Basic Queries

### Select with Conditions
```sql
-- Basic SELECT
SELECT column1, column2 
FROM table_name 
WHERE condition = 'value';

-- Multiple conditions
SELECT * 
FROM users 
WHERE age > 18 
  AND status = 'active' 
  AND created_at >= '2025-01-01';

-- LIKE pattern matching
SELECT * 
FROM products 
WHERE name LIKE '%laptop%';

-- IN operator
SELECT * 
FROM orders 
WHERE status IN ('pending', 'processing', 'shipped');

-- BETWEEN
SELECT * 
FROM transactions 
WHERE amount BETWEEN 100 AND 1000;
```

### Ordering and Limiting
```sql
-- ORDER BY
SELECT * 
FROM users 
ORDER BY created_at DESC, name ASC;

-- LIMIT and OFFSET (pagination)
SELECT * 
FROM products 
ORDER BY created_at DESC 
LIMIT 10 OFFSET 20;

-- TOP (SQL Server)
SELECT TOP 10 * 
FROM users 
ORDER BY created_at DESC;
```

## 🔗 Joins

### Inner Join
```sql
SELECT 
    u.id,
    u.name,
    o.order_date,
    o.total
FROM users u
INNER JOIN orders o ON u.id = o.user_id
WHERE o.status = 'completed';
```

### Left Join
```sql
-- Get all users and their orders (including users with no orders)
SELECT 
    u.id,
    u.name,
    o.id as order_id,
    o.total
FROM users u
LEFT JOIN orders o ON u.id = o.user_id;
```

### Multiple Joins
```sql
SELECT 
    u.name,
    o.order_date,
    p.name as product_name,
    oi.quantity,
    oi.price
FROM users u
INNER JOIN orders o ON u.id = o.user_id
INNER JOIN order_items oi ON o.id = oi.order_id
INNER JOIN products p ON oi.product_id = p.id
WHERE o.status = 'completed';
```

## 📈 Aggregations

### Basic Aggregations
```sql
-- COUNT, SUM, AVG, MIN, MAX
SELECT 
    COUNT(*) as total_orders,
    SUM(total) as revenue,
    AVG(total) as average_order,
    MIN(total) as smallest_order,
    MAX(total) as largest_order
FROM orders
WHERE status = 'completed';
```

### GROUP BY
```sql
-- Orders per user
SELECT 
    user_id,
    COUNT(*) as order_count,
    SUM(total) as total_spent
FROM orders
GROUP BY user_id
ORDER BY total_spent DESC;

-- HAVING clause (filter after grouping)
SELECT 
    user_id,
    COUNT(*) as order_count
FROM orders
GROUP BY user_id
HAVING COUNT(*) > 5;
```

### GROUP BY with Dates
```sql
-- Daily sales
SELECT 
    DATE(created_at) as date,
    COUNT(*) as orders,
    SUM(total) as revenue
FROM orders
GROUP BY DATE(created_at)
ORDER BY date DESC;

-- Monthly aggregation
SELECT 
    DATE_FORMAT(created_at, '%Y-%m') as month,
    COUNT(*) as orders,
    SUM(total) as revenue
FROM orders
GROUP BY DATE_FORMAT(created_at, '%Y-%m')
ORDER BY month DESC;
```

## 🔄 Subqueries

### Subquery in WHERE
```sql
-- Users who placed orders in the last 30 days
SELECT * 
FROM users 
WHERE id IN (
    SELECT DISTINCT user_id 
    FROM orders 
    WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
);
```

### Subquery in SELECT
```sql
-- Users with order count
SELECT 
    u.id,
    u.name,
    (SELECT COUNT(*) FROM orders WHERE user_id = u.id) as order_count
FROM users u;
```

### Subquery in FROM
```sql
-- Average order value by month
SELECT 
    month,
    AVG(monthly_total) as avg_monthly_revenue
FROM (
    SELECT 
        DATE_FORMAT(created_at, '%Y-%m') as month,
        SUM(total) as monthly_total
    FROM orders
    GROUP BY DATE_FORMAT(created_at, '%Y-%m')
) as monthly_orders
GROUP BY month;
```

## 🛠️ Data Manipulation

### INSERT
```sql
-- Single row
INSERT INTO users (name, email, created_at)
VALUES ('John Doe', 'john@example.com', NOW());

-- Multiple rows
INSERT INTO users (name, email, created_at)
VALUES 
    ('Alice', 'alice@example.com', NOW()),
    ('Bob', 'bob@example.com', NOW()),
    ('Charlie', 'charlie@example.com', NOW());

-- Insert from SELECT
INSERT INTO archived_orders (order_id, user_id, total, archived_at)
SELECT id, user_id, total, NOW()
FROM orders
WHERE created_at < DATE_SUB(NOW(), INTERVAL 1 YEAR);
```

### UPDATE
```sql
-- Basic update
UPDATE users 
SET status = 'active', updated_at = NOW()
WHERE id = 123;

-- Update with conditions
UPDATE products 
SET price = price * 0.9
WHERE category = 'electronics' 
  AND stock > 10;

-- Update with JOIN
UPDATE users u
INNER JOIN orders o ON u.id = o.user_id
SET u.last_order_date = o.created_at
WHERE o.id = (
    SELECT MAX(id) FROM orders WHERE user_id = u.id
);
```

### DELETE
```sql
-- Basic delete
DELETE FROM users 
WHERE status = 'inactive' 
  AND last_login < DATE_SUB(NOW(), INTERVAL 1 YEAR);

-- Delete with JOIN (be careful!)
DELETE o
FROM orders o
INNER JOIN users u ON o.user_id = u.id
WHERE u.status = 'deleted';
```

## 🔍 Window Functions

### ROW_NUMBER
```sql
-- Rank users by order count
SELECT 
    user_id,
    COUNT(*) as order_count,
    ROW_NUMBER() OVER (ORDER BY COUNT(*) DESC) as rank
FROM orders
GROUP BY user_id;
```

### PARTITION BY
```sql
-- Running total of orders per user
SELECT 
    user_id,
    created_at,
    total,
    SUM(total) OVER (
        PARTITION BY user_id 
        ORDER BY created_at
    ) as running_total
FROM orders;
```

### LAG and LEAD
```sql
-- Compare current order with previous order
SELECT 
    user_id,
    created_at,
    total,
    LAG(total) OVER (PARTITION BY user_id ORDER BY created_at) as previous_order,
    LEAD(total) OVER (PARTITION BY user_id ORDER BY created_at) as next_order
FROM orders;
```

## 🗄️ Table Management

### CREATE TABLE
```sql
CREATE TABLE users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_email (email)
);
```

### ALTER TABLE
```sql
-- Add column
ALTER TABLE users 
ADD COLUMN phone VARCHAR(20);

-- Modify column
ALTER TABLE users 
MODIFY COLUMN name VARCHAR(500) NOT NULL;

-- Add index
ALTER TABLE orders 
ADD INDEX idx_user_date (user_id, created_at);

-- Add foreign key
ALTER TABLE orders
ADD CONSTRAINT fk_user
FOREIGN KEY (user_id) REFERENCES users(id);
```

## 📊 Common Patterns

### Find Duplicates
```sql
-- Find duplicate emails
SELECT 
    email, 
    COUNT(*) as count
FROM users
GROUP BY email
HAVING COUNT(*) > 1;
```

### Delete Duplicates (keep oldest)
```sql
DELETE t1 
FROM users t1
INNER JOIN users t2 
WHERE t1.id > t2.id 
  AND t1.email = t2.email;
```

### Pivot Data
```sql
-- Orders by status (pivot)
SELECT 
    user_id,
    SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
    SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) as cancelled
FROM orders
GROUP BY user_id;
```

### Running Totals
```sql
-- Cumulative revenue
SELECT 
    DATE(created_at) as date,
    SUM(total) as daily_revenue,
    SUM(SUM(total)) OVER (ORDER BY DATE(created_at)) as cumulative_revenue
FROM orders
GROUP BY DATE(created_at);
```

## 🔐 Indexes and Performance

### Create Indexes
```sql
-- Single column index
CREATE INDEX idx_user_email ON users(email);

-- Composite index
CREATE INDEX idx_order_user_date ON orders(user_id, created_at);

-- Unique index
CREATE UNIQUE INDEX idx_user_username ON users(username);

-- Full-text index (MySQL)
CREATE FULLTEXT INDEX idx_product_description ON products(description);
```

### Analyze Query Performance
```sql
-- EXPLAIN query
EXPLAIN SELECT * 
FROM orders 
WHERE user_id = 123 
  AND created_at >= '2025-01-01';

-- Show indexes
SHOW INDEXES FROM orders;
```

## 🧪 Testing Queries

### Safe Testing Pattern
```sql
-- Always test with SELECT first
SELECT * 
FROM users 
WHERE status = 'inactive' 
  AND last_login < DATE_SUB(NOW(), INTERVAL 1 YEAR)
LIMIT 10;

-- Then wrap in transaction for safety
START TRANSACTION;

UPDATE users 
SET status = 'deleted' 
WHERE status = 'inactive' 
  AND last_login < DATE_SUB(NOW(), INTERVAL 1 YEAR);

-- Check results
SELECT * FROM users WHERE status = 'deleted' LIMIT 10;

-- If good: COMMIT; If bad: ROLLBACK;
ROLLBACK;
```

## 🎯 PostgreSQL Specific

### JSON Operations
```sql
-- Query JSON column
SELECT * 
FROM products 
WHERE metadata->>'color' = 'blue';

-- Array operations
SELECT * 
FROM users 
WHERE 'admin' = ANY(roles);

-- JSONB aggregation
SELECT 
    user_id,
    jsonb_agg(jsonb_build_object('id', id, 'total', total)) as orders
FROM orders
GROUP BY user_id;
```

### Array Functions
```sql
-- Array contains
SELECT * 
FROM posts 
WHERE tags @> ARRAY['python', 'tutorial'];

-- Array overlap
SELECT * 
FROM posts 
WHERE tags && ARRAY['python', 'javascript'];
```

---
Tags: #snippets #sql #database #reference
