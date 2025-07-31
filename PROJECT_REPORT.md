# Quiz Master V2 - Project Report

## Author
**[Your Full Name]**  
**Roll Number: [Your Roll Number]**  
**Email: [Your Student Email]**  
**[Add a couple of lines about yourself]**

## Description
This project is a comprehensive quiz management system designed for educational institutions, featuring role-based access control, automated scheduling, and real-time notifications. The system allows administrators to create and manage quizzes while providing students with an intuitive interface to take quizzes and track their performance. The application implements modern web development practices with microservices architecture using background job processing for enhanced user experience.

**AI/LLM Usage: [X]% - [Specify percentage and mention if used for code generation, debugging, documentation, or research assistance]**

## Technologies Used

### Backend Technologies
- **Flask 3.0.0** - Primary web framework for REST API development
- **Flask-SQLAlchemy 3.1.1** - ORM for database operations and models
- **Flask-JWT-Extended 4.5.3** - JWT token-based authentication and authorization
- **Flask-CORS 4.0.0** - Cross-origin resource sharing for frontend integration
- **Flask-Limiter 3.5.0** - API rate limiting for security and performance

### Database & Caching
- **SQLite** - Primary database for data persistence
- **Redis 5.0.1** - Caching layer and message broker for Celery
- **SQLAlchemy 2.0.23** - Database toolkit and ORM

### Background Processing
- **Celery 5.3.4** - Distributed task queue for background jobs (reminders, exports, reports)
- **Redis** - Message broker for Celery task distribution

### Frontend Technologies
- **Vue.js 3.3.8** - Progressive JavaScript framework for reactive UI
- **Vue Router 4.2.5** - Client-side routing for single-page application
- **Vuex 4.1.0** - State management pattern and library
- **Bootstrap 5.3.2** - CSS framework for responsive design
- **Chart.js 4.4.1** - Data visualization for analytics and reports
- **Axios 1.6.2** - HTTP client for API communication

### Additional Tools
- **Vite 5.0.0** - Fast build tool and development server
- **ReportLab 4.0.7** - PDF generation for reports and exports
- **Pandas 2.1.4** - Data manipulation for analytics and CSV exports
- **bcrypt 4.1.2** - Password hashing for security

**Purpose**: These technologies were chosen to create a scalable, secure, and maintainable system. Flask provides a lightweight yet powerful backend, Vue.js ensures a modern reactive frontend, Redis enables caching and background processing, while Celery handles automated tasks like reminders and report generation.

## DB Schema Design

### Core Tables

**Users Table**
- `id` (Primary Key, Integer) - Unique user identifier
- `username` (String, Unique, Not Null) - User login name
- `email` (String, Unique, Not Null) - User email address
- `password_hash` (String, Not Null) - Bcrypt hashed password
- `full_name` (String, Not Null) - User's full name
- `qualification` (String, Nullable) - User's educational qualification
- `role` (String, Default: 'user') - Role-based access (user/admin)
- `is_active` (Boolean, Default: True) - Account status
- `created_at`, `updated_at`, `last_login` (DateTime) - Audit timestamps

**Subjects Table**
- `id` (Primary Key) - Subject identifier
- `name` (String, Unique, Not Null) - Subject name
- `slug` (String, Unique) - URL-friendly identifier
- `description` (Text) - Subject description
- `is_active` (Boolean) - Subject status

**Chapters Table**
- `id` (Primary Key) - Chapter identifier
- `title` (String, Not Null) - Chapter title
- `slug` (String, Not Null) - URL-friendly identifier
- `subject_id` (Foreign Key) - Reference to subjects table
- `description` (Text) - Chapter description
- `order_index` (Integer) - Chapter ordering

**Quizzes Table**
- `id` (Primary Key) - Quiz identifier
- `title` (String, Not Null) - Quiz title
- `slug` (String, Not Null) - URL-friendly identifier
- `chapter_id` (Foreign Key) - Reference to chapters table
- `description` (Text) - Quiz description
- `duration_minutes` (Integer, Default: 30) - Time limit
- `passing_score` (Integer, Default: 60) - Minimum passing percentage
- `max_attempts` (Integer, Default: 3) - Maximum allowed attempts
- `start_date`, `end_date` (DateTime) - Quiz availability schedule
- `is_active` (Boolean) - Quiz status

**Questions Table**
- `id` (Primary Key) - Question identifier
- `quiz_id` (Foreign Key) - Reference to quizzes table
- `content` (Text, Not Null) - Question text
- `options` (JSON) - Multiple choice options
- `correct_answer` (String, Not Null) - Correct answer
- `marks` (Integer, Default: 1) - Points for correct answer

**Scores Table**
- `id` (Primary Key) - Score record identifier
- `user_id` (Foreign Key) - Reference to users table
- `quiz_id` (Foreign Key) - Reference to quizzes table
- `score` (Integer, Not Null) - Points earned
- `total_marks` (Integer, Not Null) - Maximum possible points
- `percentage` (Float) - Calculated percentage
- `time_taken` (Integer) - Time taken in seconds
- `passed` (Boolean) - Whether user passed
- `created_at` (DateTime) - Attempt timestamp

**Reminders Table**
- `id` (Primary Key) - Reminder identifier
- `user_id` (Foreign Key) - Reference to users table
- `message` (Text, Not Null) - Reminder message
- `reminder_type` (String) - Type of reminder
- `is_read` (Boolean, Default: False) - Read status
- `created_at` (DateTime) - Creation timestamp

**Design Rationale**: The schema follows normalized design principles with proper foreign key relationships. The design supports role-based access, quiz scheduling, attempt tracking, and automated reminders. Audit fields (created_at, updated_at) enable proper tracking, while slug fields provide SEO-friendly URLs.

## API Design

The application implements a RESTful API architecture with the following key endpoints:

### Authentication & Authorization
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User authentication with JWT token generation
- `POST /api/auth/logout` - User logout and token invalidation
- `GET /api/auth/profile` - Get current user profile

### Admin Management APIs
- `GET /api/admin/users` - List all users with pagination and search
- `POST /api/admin/subjects` - Create new subjects
- `PUT /api/admin/subjects/{id}` - Update existing subjects
- `POST /api/admin/quizzes` - Create quizzes with questions
- `GET /api/admin/analytics` - Get comprehensive analytics data
- `POST /api/admin/export` - Trigger CSV export tasks

### User APIs
- `GET /api/user/dashboard` - User dashboard with stats and available quizzes
- `GET /api/user/subjects` - List available subjects and chapters
- `POST /api/quiz/attempt` - Submit quiz attempt
- `GET /api/user/scores` - Get user's score history
- `GET /api/user/reminders` - Get user notifications

### Common APIs
- `GET /api/search` - Global search across subjects, quizzes, and content
- `GET /api/health` - API health check endpoint

**Implementation Features**: All APIs implement JWT-based authentication, role-based authorization decorators, request validation, error handling, and Redis caching for performance optimization. Rate limiting is applied to prevent abuse.

## Architecture and Features

### Project Organization
The project follows a modular architecture with clear separation of concerns. The backend is organized using Flask blueprints with separate modules for routes (`/routes`), models (`/models`), utilities (`/utils`), and background tasks (`/celery_tasks`). The frontend uses Vue.js with component-based architecture, centralized state management via Vuex, and routing through Vue Router. Static assets and styles are organized in the `/assets` directory.

### Implemented Features

**Core Features:**
- **User Authentication & RBAC**: JWT-based authentication with role-based access control supporting admin and user roles
- **Subject & Chapter Management**: Hierarchical content organization with CRUD operations
- **Quiz Management**: Complete quiz lifecycle including creation, scheduling, attempt tracking, and automatic scoring
- **Real-time Analytics**: Dashboard with performance metrics, score trends, and statistical analysis using Chart.js
- **Global Search**: Full-text search across subjects, quizzes, and user data with Redis caching

**Advanced Features:**
- **Background Job Processing**: Celery-powered automated tasks including daily reminder notifications via Google Chat webhooks, monthly activity reports via email, and CSV export generation
- **Caching Layer**: Redis-based caching for API responses, user sessions, and frequently accessed data
- **Rate Limiting**: API protection with configurable rate limits per user/endpoint
- **Responsive Design**: Mobile-first design using Bootstrap with progressive web app features
- **Data Export**: Automated CSV generation for user scores, admin reports, and analytics data with email notifications

**Additional Features:**
- **Quiz Scheduling**: Time-based quiz availability with automatic start/end enforcement
- **Attempt Limiting**: Configurable maximum attempts per quiz with attempt tracking
- **Performance Analytics**: Detailed score analysis, pass/fail statistics, and trending data
- **Notification System**: In-app reminder system with database persistence and real-time updates

## Video
**[Insert link to your online video of not more than 3 minutes length]**
