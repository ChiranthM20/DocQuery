# PaperLens v2.0 - Complete Architecture & Implementation Guide

## 📁 RECOMMENDED FOLDER STRUCTURE

```
paperlens/
├── 📄 README.md                     # Project documentation
├── 📄 ARCHITECTURE.md                # This file
│
├── api/                              # FastAPI Backend
│   ├── main.py                       # App entry point
│   ├── requirements.txt               # Backend dependencies
│   │
│   ├── routes/                       # API Endpoints
│   │   ├── __init__.py
│   │   ├── auth.py                   # Sign up, login, logout
│   │   ├── chat.py                   # Chat endpoints
│   │   ├── documents.py              # Upload, list, delete
│   │   ├── ai_actions.py             # Quick actions (summarize, etc)
│   │   └── insights.py               # Smart insights generation
│   │
│   ├── models/                       # Data models & schemas
│   │   ├── __init__.py
│   │   ├── schemas.py                # Pydantic models
│   │   └── database.py               # User/chat/document models
│   │
│   ├── services/                     # Business logic
│   │   ├── __init__.py
│   │   ├── auth_service.py           # Firebase integration
│   │   ├── rag_service.py            # FAISS + embedding logic
│   │   ├── chat_service.py           # Chat/LLM logic
│   │   ├── ai_service.py             # AI actions orchestrator
│   │   └── document_service.py       # File processing + chunking
│   │
│   ├── utils/                        # Utilities
│   │   ├── __init__.py
│   │   ├── text_processing.py        # Chunking, cleaning
│   │   ├── file_handlers.py          # PDF, DOCX parsing
│   │   ├── prompt_templates.py       # System prompts
│   │   └── constants.py              # Magic numbers
│   │
│   ├── ai/                           # AI/ML modules
│   │   ├── __init__.py
│   │   ├── vector_store.py           # FAISS wrapper (improved)
│   │   ├── chat_memory.py            # Conversation memory
│   │   ├── llm.py                    # Ollama wrapper
│   │   ├── embeddings.py             # SentenceTransformers
│   │   └── response_generator.py     # Format responses
│   │
│   ├── data/                         # Data storage
│   │   ├── indexes/                  # FAISS vector indexes
│   │   ├── temp_uploads/             # Temporary files
│   │   └── cache/                    # Query cache
│   │
│   └── logs/                         # Logs
│
├── ui/                               # React Frontend
│   ├── index.html
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── package.json
│   │
│   ├── src/
│   │   ├── main.jsx
│   │   ├── App.jsx                   # Route container
│   │   ├── api.js                    # API client
│   │   ├── index.css                 # Global styles
│   │   │
│   │   ├── pages/                    # Full pages
│   │   │   ├── LandingPage.jsx       # Marketing page
│   │   │   ├── AuthPage.jsx          # Login/Signup
│   │   │   ├── DashboardPage.jsx     # Main chat interface
│   │   │   └── DocumentViewerPage.jsx # View doc + chat
│   │   │
│   │   ├── components/               # Reusable components
│   │   │   ├── Header.jsx
│   │   │   ├── Sidebar.jsx           # Document library
│   │   │   ├── ChatPanel.jsx         # Chat messages
│   │   │   ├── InputBox.jsx          # Chat input
│   │   │   ├── SourceCard.jsx        # Citation/source
│   │   │   ├── InsightsPanel.jsx     # Smart insights
│   │   │   ├── UploadCard.jsx        # Drag-drop upload
│   │   │   ├── QuickActions.jsx      # Action buttons
│   │   │   ├── ModeSelector.jsx      # Performance mode picker
│   │   │   ├── LoadingSpinner.jsx    # Custom loader
│   │   │   ├── StatBadges.jsx        # Document stats
│   │   │   ├── UserMenu.jsx          # Profile dropdown
│   │   │   └── TutorPanel.jsx        # AI Tutor (WOW feature)
│   │   │
│   │   ├── hooks/                    # Custom React hooks
│   │   │   ├── useChat.js            # Chat logic
│   │   │   ├── useAuth.js            # Auth context
│   │   │   ├── useDocuments.js       # Document CRUD
│   │   │   └── useInsights.js        # Insights generation
│   │   │
│   │   ├── context/                  # React Context
│   │   │   ├── AuthContext.jsx
│   │   │   ├── ThemeContext.jsx
│   │   │   └── ChatContext.jsx
│   │   │
│   │   ├── styles/                   # Tailwind utilities
│   │   │   ├── animations.css
│   │   │   ├── gradients.css
│   │   │   └── glass.css
│   │   │
│   │   └── utils/
│   │       ├── formatting.js
│   │       ├── localStorage.js
│   │       └── constants.js
│   │
│   └── public/
│       ├── icons/
│       └── sample-docs/              # Demo PDFs
│
├── docs/                             # Documentation
│   ├── API.md                        # API reference
│   ├── FEATURES.md                   # Feature descriptions
│   └── DEPLOYMENT.md                 # Deploy guide
│
├── .env.example
├── docker-compose.yml                # Docker setup (bonus)
└── README.md                         # Updated
```

---

## 🔧 BACKEND ARCHITECTURE

### Technology Stack
```
FastAPI 0.104+
Uvicorn (ASGI)
Firebase Auth (or Supabase)
PostgreSQL (user data) or DynamoDB
SQLAlchemy ORM
Pydantic v2

AI/ML:
- Ollama (LLM inference)
- SentenceTransformers (embeddings)
- FAISS (vector search)
- LangChain (orchestration)

Background Jobs:
- Celery (optional, for async tasks)
- Redis cache
```

### API Routes Structure
```
AUTH ROUTES:
  POST   /auth/signup              Sign up
  POST   /auth/login               Login
  POST   /auth/logout              Logout
  GET    /auth/me                  Current user
  POST   /auth/refresh             Refresh token

CHAT ROUTES:
  POST   /chat/message             Send message (dual mode)
  GET    /chat/history/{doc_id}    Get chat history
  DELETE /chat/history/{msg_id}    Delete message
  POST   /chat/clear               Clear conversation

DOCUMENT ROUTES:
  POST   /documents/upload         Upload file
  POST   /documents/paste          Paste text
  GET    /documents                List user documents
  GET    /documents/{id}           Get document details
  DELETE /documents/{id}           Delete document
  GET    /documents/{id}/preview   Get document preview

AI ACTION ROUTES:
  POST   /ai/summarize             Summarize document
  POST   /ai/key-points            Extract key points
  POST   /ai/explain-simple        Explain simply (ELI5)
  POST   /ai/action-items          Extract action items
  POST   /ai/quiz                  Generate quiz
  POST   /ai/entities              Find dates/numbers/names
  POST   /ai/tutor                 Start tutoring session (WOW)

INSIGHTS ROUTES:
  GET    /insights/{doc_id}        Get auto-generated insights
  GET    /insights/{doc_id}/topics Topic extraction
  GET    /insights/{doc_id}/entities Key entities

STATUS:
  GET    /health                   Health check
  GET    /status                   System status
```

### Key Service Flow

**Regular Chat (No Document):**
```
User Input
    ↓
ChatService.process_message()
    ↓
Check if greeting (hi/hello) → Return warm greeting
    ↓
LLM.generate_response(context=chat_history)
    ↓
Store in ChatMemory
    ↓
Return Response
```

**RAG Chat (Document Uploaded):**
```
User Input
    ↓
ChatService.process_message(document_id)
    ↓
VectorStore.search(query, top_k=3)
    ↓
Format sources & context
    ↓
RAGService.generate_rag_response(
    query, 
    context, 
    chat_history,
    mode=balanced
)
    ↓
Add confidence scores
    ↓
Store in ChatMemory
    ↓
Return Response + Sources
```

**AI Tutor (WOW Feature):**
```
Document Uploaded
    ↓
TutorService.initialize_session(doc_id)
    ↓
Extract key concepts from document
    ↓
Generate learning path (3-5 steps)
    ↓
User: "Start tutoring"
    ↓
TutorService.next_step():
    - Explain concept
    - Ask clarifying Qs
    - Ask quiz Q
    ↓
Track understanding score
    ↓
Next step or review
    ↓
Final assessment
```

---

## 🎨 FRONTEND ARCHITECTURE

### Component Hierarchy
```
App
├── AuthContext
├── ChatContext
├── ThemeContext
│
├── Landing Page (public)
│   ├── Header
│   ├── Hero Section
│   ├── Features Grid
│   ├── Testimonials
│   └── CTA Button
│
├── Auth Page
│   ├── Login Form
│   ├── Signup Form
│   └── OAuth Buttons
│
└── Dashboard Page (protected)
    ├── Header
    │   ├── UserMenu
    │   └── ModeSelector
    │
    ├── Sidebar
    │   ├── UploadArea (drag-drop)
    │   ├── DocumentLibrary
    │   │   ├── Document Item
    │   │   ├── Metadata (chunks, size)
    │   │   └── Actions (view, delete)
    │   └── QuickActions
    │
    ├── MainContent
    │   ├── ChatPanel
    │   │   ├── Chat Messages
    │   │   │   ├── UserMessage
    │   │   │   ├── AiMessage
    │   │   │   └── SourceCard (citations)
    │   │   ├── InputBox
    │   │   │   ├── Text input
    │   │   │   ├── Send button
    │   │   │   └── Typing indicator
    │   │   └── QuickActions (button row)
    │   │
    │   └── InsightsPanel (right sidebar)
    │       ├── Auto Summary
    │       ├── Key Topics
    │       ├── Key Entities
    │       ├── Document Stats
    │       └── Reading Time
    │
    └── TutorPanel (modal/sidebar when active)
        ├── Concept Explanation
        ├── Progress Bar
        ├── Current Question
        ├── Navigation buttons
        └── Score Display
```

### State Management
```
AuthContext:
  - user (id, email, name)
  - isAuthenticated
  - userId
  - logout()

ChatContext:
  - messages (array)
  - activeDocumentId
  - chatHistory (dict by doc_id)
  - conversationMemory
  - addMessage(role, content)
  - clearHistory()

DocumentContext:
  - documents (array)
  - activeDocument
  - uploadDocument()
  - deleteDocument()
  - setActive()

InsightContext:
  - insights (auto-generated)
  - topics
  - entities
  - refreshInsights()
```

---

## 🚀 IMPLEMENTATION PRIORITY

### Must Have (Day 1):
1. Auth system (Firebase signup/login)
2. Dual AI mode (chatbot + RAG switching)
3. Chat memory system
4. Basic greeting handling
5. Document CRUD endpoints

### Should Have (Day 1-2):
1. Advanced AI actions
2. Smart insights panel
3. Source quality improvements
4. Premium UI components
5. Landing page

### Nice to Have (Day 2-3):
1. WOW feature (AI Tutor recommended)
2. Mobile responsiveness
3. Demo mode with sample docs
4. Performance optimization
5. Analytics

---

## 📊 DATABASE SCHEMA

### Users Table
```sql
users:
  - id (UUID, primary)
  - email (unique)
  - name
  - password_hash (Firebase handles this)
  - created_at
  - updated_at
  - profile_picture
  - preferences (JSON)
```

### Documents Table
```sql
documents:
  - id (UUID, primary)
  - user_id (FK)
  - original_filename
  - document_type (pdf, txt, docx, md)
  - file_hash (for dedup)
  - filesize
  - chunk_count
  - vector_index_id
  - auto_summary
  - key_topics (JSON array)
  - key_entities (JSON array)
  - created_at
  - updated_at
  - indexed_at
```

### ChatMessages Table
```sql
chat_messages:
  - id (UUID, primary)
  - user_id (FK)
  - document_id (FK, nullable)
  - message_type (user, assistant, system)
  - content (text)
  - sources (JSON array with citations)
  - confidence_score (0-1)
  - mode (fast/balanced/quality)
  - tokens_used
  - timings (JSON: search_time, gen_time)
  - created_at
```

### ChatSessions Table
```sql
chat_sessions:
  - id (UUID, primary)
  - user_id (FK)
  - document_id (FK, nullable)
  - conversation_memory (JSON)
  - message_count
  - started_at
  - ended_at
  - total_tokens_used
```

---

## 🔒 Authentication Flow

### Signup
```
Email + Password → Firebase Auth → User Created
                                 → Store user in DB
                                 → Create JWT token
                                 → Redirect to dashboard
                                 → Pre-load demo docs
```

### Login
```
Email + Password → Firebase Auth → Verify credentials
                                 → Generate JWT token
                                 → Load user documents
                                 → Load chat history
                                 → Redirect to dashboard
```

### Protected Routes
```
All API routes (except /auth/signup, /auth/login)
    ↓
Require Authorization header with JWT
    ↓
Verify token signature
    ↓
Check user_id matches resource
    ↓
Execute endpoint
```

---

## 🎯 Performance Modes Explained

### ⚡ Fast Mode
- max_context_tokens: 1000
- chunk_count: 2-3
- model_params: temperature=0.7
- response_time: ~2-3 seconds
- Best for: Quick facts, summaries

### ⚖️ Balanced Mode (Default)
- max_context_tokens: 2000
- chunk_count: 3-5
- model_params: temperature=0.5
- response_time: ~4-6 seconds
- Best for: Normal Q&A, explanations

### 🎯 Quality Mode
- max_context_tokens: 4000
- chunk_count: 5-8
- model_params: temperature=0.3
- response_time: ~8-12 seconds
- Best for: Deep analysis, comprehensive answers

---

## 🌟 WOW FEATURE DEEP DIVE: AI Tutor Mode

### Why This Feature?
- ✅ Judges love educational applications
- ✅ Highly interactive & engaging
- ✅ Clear before/after learning outcomes
- ✅ "I've never seen this in a RAG app before"

### How It Works

**Phase 1: Content Analysis**
```
Document uploaded
    ↓
TutorService extracts key concepts
    ↓
Build concept dependency graph
    ↓
Identify prerequisites
    ↓
Create learning path (3-5 steps)
```

**Phase 2: Interactive Learning**
```
Step 1: Concept Explanation
  "Here's what X means..."
  
Step 2: Ask Clarifying Questions
  "Do you understand Y? Can you explain Z?"
  
Step 3: Apply Knowledge
  "Can you give an example of X in real life?"
  
Step 4: Quiz Question
  "Which of these is X? Why?"
  
Step 5: Score & Feedback
  Display understanding score
  Suggest next steps
```

### UI for AI Tutor
```
┌─────────────────────────────────┐
│     📚 AI Tutor Mode            │ ← Prominent badge
├─────────────────────────────────┤
│                                 │
│  Learning Path:                 │
│  1. Fundamentals ────────○       │ ← Step indicator
│  2. Applied Concepts            │
│  3. Advanced Topics             │
│        [Start Tutoring]         │
│                                 │
│  Current Concept:               │
│  ┌─────────────────────────────┐│
│  │ Machine Learning Basics     ││
│  │                             ││
│  │ ML is about creating        ││
│  │ algorithms that improve     ││
│  │ with experience...          ││
│  └─────────────────────────────┘│
│                                 │
│  Do you understand?             │
│  [ Yes ] [ No, explain more ]   │
│                                 │
│  Your Score: ████░░ 65%         │
└─────────────────────────────────┘
```

---

## 📈 Success Metrics

Implement to track:

```python
# For analytics/dashboard
metrics = {
    "documents_processed": count,
    "avg_response_time": ms,
    "avg_confidence_score": float,
    "user_questions_count": count,
    "most_used_actions": list,
    "tutor_completion_rate": percent,
    "avg_learning_score": float,
}
```

---

## 🚀 Deployment Checklist

- [ ] Convert to Docker containers
- [ ] Setup PostgreSQL on cloud (Vercel, Railway, Supabase)
- [ ] Setup Firebase project
- [ ] Deploy FastAPI on Render.com or Railway
- [ ] Deploy React on Vercel
- [ ] Setup environment variables
- [ ] Configure CORS properly
- [ ] Enable HTTPS
- [ ] Setup CI/CD pipeline
- [ ] Add monitoring & logging
- [ ] Performance test under load

---

## 💡 Quick Implementation Tips

1. **Firebase Setup** (30 min)
   ```
   Create Firebase project
   Add web app config
   Install firebase SDK
   Setup auth routes
   ```

2. **Add Chat Memory** (1 hour)
   ```
   Add chat_history array to state
   Store messages in DB
   Load on page refresh
   Include in LLM context
   ```

3. **Implement AI Tutor** (2-3 hours)
   ```
   Create TutorService class
   Extract top 3-5 concepts
   Generate step-by-step prompts
   Build interactive UI with progress
   ```

4. **Beautiful UI** (3-4 hours)
   ```
   Add Tailwind gradient backgrounds
   Implement glassmorphism cards
   Add Framer Motion animations
   Create micro-interactions
   ```

---

**THIS ARCHITECTURE IS PRODUCTION-READY AND HACKATHON-OPTIMIZED** ✅
