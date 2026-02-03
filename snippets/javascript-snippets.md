# JavaScript/TypeScript Snippets

A collection of reusable JavaScript and TypeScript code snippets.

## 📦 Modern JavaScript

### Array Operations
```javascript
// Map, filter, reduce combo
const result = items
    .filter(item => item.active)
    .map(item => item.value)
    .reduce((sum, val) => sum + val, 0);

// Find and find index
const found = items.find(item => item.id === targetId);
const index = items.findIndex(item => item.id === targetId);

// Array destructuring with rest
const [first, second, ...rest] = array;

// Flatten array
const flattened = nested.flat(Infinity);

// Remove duplicates
const unique = [...new Set(array)];

// Sort with custom comparator
items.sort((a, b) => b.date - a.date);
```

### Object Operations
```javascript
// Object destructuring with rename and default
const { name: userName, age = 18 } = user;

// Spread and merge objects
const merged = { ...defaults, ...userSettings };

// Dynamic property names
const key = 'dynamicKey';
const obj = { [key]: value };

// Object.entries iteration
Object.entries(obj).forEach(([key, value]) => {
    console.log(`${key}: ${value}`);
});

// Deep clone (simple objects only)
const clone = JSON.parse(JSON.stringify(original));
```

### Async/Await Patterns
```javascript
// Basic async function
async function fetchData(url) {
    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error('Network error');
        return await response.json();
    } catch (error) {
        console.error('Fetch error:', error);
        throw error;
    }
}

// Parallel execution
const [users, posts, comments] = await Promise.all([
    fetchUsers(),
    fetchPosts(),
    fetchComments()
]);

// Sequential with error handling
async function processItems(items) {
    const results = [];
    for (const item of items) {
        try {
            const result = await processItem(item);
            results.push(result);
        } catch (error) {
            console.error(`Failed to process ${item}:`, error);
        }
    }
    return results;
}

// Race condition (timeout)
const fetchWithTimeout = async (url, timeout = 5000) => {
    const timeoutPromise = new Promise((_, reject) => 
        setTimeout(() => reject(new Error('Timeout')), timeout)
    );
    return Promise.race([fetch(url), timeoutPromise]);
};
```

### String Manipulation
```javascript
// Template literals with expressions
const message = `Hello ${name}, you have ${count} new messages`;

// Multi-line strings
const html = `
    <div>
        <h1>${title}</h1>
        <p>${content}</p>
    </div>
`;

// String methods
const trimmed = str.trim();
const lowercase = str.toLowerCase();
const replaced = str.replace(/pattern/g, 'replacement');
const split = str.split(',').map(s => s.trim());

// Check string content
const hasSubstring = str.includes('search');
const startsWith = str.startsWith('prefix');
const endsWith = str.endsWith('.js');
```

## ⚛️ React Patterns

### Functional Component with Hooks
```typescript
import React, { useState, useEffect } from 'react';

interface Props {
    initialValue: number;
    onUpdate?: (value: number) => void;
}

const MyComponent: React.FC<Props> = ({ initialValue, onUpdate }) => {
    const [value, setValue] = useState(initialValue);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        // Effect runs on mount and when value changes
        console.log('Value changed:', value);
        onUpdate?.(value);
        
        // Cleanup function
        return () => {
            console.log('Cleanup');
        };
    }, [value, onUpdate]);

    const handleIncrement = () => {
        setValue(prev => prev + 1);
    };

    if (loading) return <div>Loading...</div>;

    return (
        <div>
            <p>Value: {value}</p>
            <button onClick={handleIncrement}>Increment</button>
        </div>
    );
};

export default MyComponent;
```

### Custom Hook
```typescript
import { useState, useEffect } from 'react';

function useFetch<T>(url: string) {
    const [data, setData] = useState<T | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<Error | null>(null);

    useEffect(() => {
        const fetchData = async () => {
            try {
                setLoading(true);
                const response = await fetch(url);
                if (!response.ok) throw new Error('Failed to fetch');
                const json = await response.json();
                setData(json);
                setError(null);
            } catch (err) {
                setError(err instanceof Error ? err : new Error('Unknown error'));
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, [url]);

    return { data, loading, error };
}

// Usage
function MyComponent() {
    const { data, loading, error } = useFetch<User[]>('/api/users');
    
    if (loading) return <div>Loading...</div>;
    if (error) return <div>Error: {error.message}</div>;
    
    return <div>{/* render data */}</div>;
}
```

### Context API
```typescript
import React, { createContext, useContext, useState, ReactNode } from 'react';

interface AuthContextType {
    user: User | null;
    login: (username: string, password: string) => Promise<void>;
    logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
    const [user, setUser] = useState<User | null>(null);

    const login = async (username: string, password: string) => {
        // Login logic
        const userData = await loginAPI(username, password);
        setUser(userData);
    };

    const logout = () => {
        setUser(null);
    };

    return (
        <AuthContext.Provider value={{ user, login, logout }}>
            {children}
        </AuthContext.Provider>
    );
};

export const useAuth = () => {
    const context = useContext(AuthContext);
    if (!context) {
        throw new Error('useAuth must be used within AuthProvider');
    }
    return context;
};
```

## 🎯 TypeScript Patterns

### Type Guards
```typescript
// User-defined type guard
function isString(value: unknown): value is string {
    return typeof value === 'string';
}

// Discriminated unions
type Success = { status: 'success'; data: any };
type Error = { status: 'error'; message: string };
type Result = Success | Error;

function handleResult(result: Result) {
    if (result.status === 'success') {
        // TypeScript knows result.data exists here
        console.log(result.data);
    } else {
        // TypeScript knows result.message exists here
        console.error(result.message);
    }
}
```

### Utility Types
```typescript
// Partial - make all properties optional
type PartialUser = Partial<User>;

// Required - make all properties required
type RequiredConfig = Required<Config>;

// Pick - select specific properties
type UserPreview = Pick<User, 'id' | 'name' | 'email'>;

// Omit - exclude specific properties
type UserWithoutPassword = Omit<User, 'password'>;

// Record - create object type with specific keys
type PageInfo = Record<string, { title: string; url: string }>;

// Custom utility type
type Nullable<T> = T | null;
type AsyncFunction<T> = () => Promise<T>;
```

### Generics
```typescript
// Generic function
function identity<T>(arg: T): T {
    return arg;
}

// Generic interface
interface Repository<T> {
    findById(id: string): Promise<T | null>;
    findAll(): Promise<T[]>;
    save(item: T): Promise<T>;
    delete(id: string): Promise<boolean>;
}

// Generic class
class DataStore<T extends { id: string }> {
    private items: T[] = [];

    add(item: T): void {
        this.items.push(item);
    }

    findById(id: string): T | undefined {
        return this.items.find(item => item.id === id);
    }
}

// Constrained generics
function merge<T extends object, U extends object>(obj1: T, obj2: U): T & U {
    return { ...obj1, ...obj2 };
}
```

## 🔧 Node.js/Express

### Express Router Setup
```typescript
import express, { Router, Request, Response, NextFunction } from 'express';

const router: Router = express.Router();

// Middleware
const authMiddleware = (req: Request, res: Response, next: NextFunction) => {
    const token = req.headers.authorization;
    if (!token) {
        return res.status(401).json({ error: 'Unauthorized' });
    }
    // Verify token
    next();
};

// Routes
router.get('/users', authMiddleware, async (req: Request, res: Response) => {
    try {
        const users = await getUsersFromDB();
        res.json(users);
    } catch (error) {
        res.status(500).json({ error: 'Internal server error' });
    }
});

router.post('/users', async (req: Request, res: Response) => {
    try {
        const { name, email } = req.body;
        const user = await createUser({ name, email });
        res.status(201).json(user);
    } catch (error) {
        res.status(400).json({ error: 'Bad request' });
    }
});

export default router;
```

### Error Handling Middleware
```typescript
import { Request, Response, NextFunction } from 'express';

class AppError extends Error {
    statusCode: number;
    isOperational: boolean;

    constructor(message: string, statusCode: number) {
        super(message);
        this.statusCode = statusCode;
        this.isOperational = true;
        Error.captureStackTrace(this, this.constructor);
    }
}

const errorHandler = (
    err: Error,
    req: Request,
    res: Response,
    next: NextFunction
) => {
    if (err instanceof AppError) {
        return res.status(err.statusCode).json({
            status: 'error',
            message: err.message
        });
    }

    console.error('ERROR:', err);
    res.status(500).json({
        status: 'error',
        message: 'Internal server error'
    });
};

// Usage
app.use(errorHandler);

// In routes
if (!user) {
    throw new AppError('User not found', 404);
}
```

## 🎨 DOM Manipulation

### Event Handling
```javascript
// Add event listener with cleanup
const button = document.querySelector('#myButton');
const handler = (event) => {
    console.log('Clicked:', event.target);
};

button.addEventListener('click', handler);

// Remove when done
button.removeEventListener('click', handler);

// Delegate events
document.addEventListener('click', (event) => {
    if (event.target.matches('.dynamic-button')) {
        console.log('Dynamic button clicked');
    }
});

// Prevent default and stop propagation
element.addEventListener('click', (event) => {
    event.preventDefault();
    event.stopPropagation();
});
```

### Element Creation and Manipulation
```javascript
// Create element
const div = document.createElement('div');
div.className = 'container';
div.id = 'myContainer';
div.textContent = 'Hello World';
div.setAttribute('data-id', '123');

// Append to parent
parent.appendChild(div);

// Insert HTML
element.innerHTML = '<p>New content</p>';

// Query selectors
const element = document.querySelector('.class-name');
const elements = document.querySelectorAll('.items');

// Modify classes
element.classList.add('active');
element.classList.remove('inactive');
element.classList.toggle('visible');
element.classList.contains('active'); // true/false
```

## 🛠️ Utility Functions

### Debounce
```typescript
function debounce<T extends (...args: any[]) => any>(
    func: T,
    delay: number
): (...args: Parameters<T>) => void {
    let timeoutId: NodeJS.Timeout;
    
    return (...args: Parameters<T>) => {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => func(...args), delay);
    };
}

// Usage
const debouncedSearch = debounce((query: string) => {
    console.log('Searching for:', query);
}, 300);
```

### Throttle
```typescript
function throttle<T extends (...args: any[]) => any>(
    func: T,
    limit: number
): (...args: Parameters<T>) => void {
    let inThrottle: boolean;
    
    return (...args: Parameters<T>) => {
        if (!inThrottle) {
            func(...args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// Usage
const throttledScroll = throttle(() => {
    console.log('Scroll event');
}, 1000);
```

### Local Storage Helper
```typescript
const storage = {
    set<T>(key: string, value: T): void {
        try {
            localStorage.setItem(key, JSON.stringify(value));
        } catch (error) {
            console.error('Failed to save to localStorage:', error);
        }
    },

    get<T>(key: string): T | null {
        try {
            const item = localStorage.getItem(key);
            return item ? JSON.parse(item) : null;
        } catch (error) {
            console.error('Failed to read from localStorage:', error);
            return null;
        }
    },

    remove(key: string): void {
        localStorage.removeItem(key);
    },

    clear(): void {
        localStorage.clear();
    }
};

// Usage
storage.set('user', { name: 'John', age: 30 });
const user = storage.get<User>('user');
```

### Date Formatting
```javascript
// Format date
const formatDate = (date) => {
    return new Intl.DateTimeFormat('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    }).format(date);
};

// Relative time
const getRelativeTime = (date) => {
    const now = new Date();
    const diff = now - date;
    const seconds = Math.floor(diff / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);

    if (days > 0) return `${days} day${days > 1 ? 's' : ''} ago`;
    if (hours > 0) return `${hours} hour${hours > 1 ? 's' : ''} ago`;
    if (minutes > 0) return `${minutes} minute${minutes > 1 ? 's' : ''} ago`;
    return 'just now';
};
```

---
Tags: #snippets #javascript #typescript #react #nodejs #reference
