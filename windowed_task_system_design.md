# 🎯 Windowed Task System Design

## 💡 **The Elegant Solution**

Instead of 31 separate files, we create a **windowed system**:

### 📁 **File Structure**
```
.kiro/tasks/INGEST_20250929042515/
├── master-tasks.txt          # All 1,551 tasks (complete list)
├── current-window.md         # Current 50 tasks (Kiro-compatible)
├── progress.json             # Progress tracking
└── completed/                # Archive of completed tasks
    ├── batch-001.md         # Completed tasks 1-50
    ├── batch-002.md         # Completed tasks 51-100
    └── ...
```

### 🔄 **Workflow**
```bash
# 1. Generate the complete system
code-ingest generate-hierarchical-tasks INGEST_20250929042515 \
  --levels 4 --groups 7 --chunks 50 --windowed \
  --output-dir .kiro/tasks/INGEST_20250929042515/

# 2. Work on current window (always 50 tasks)
kiro .kiro/tasks/INGEST_20250929042515/current-window.md

# 3. Mark current window as complete and advance
code-ingest advance-window .kiro/tasks/INGEST_20250929042515/

# 4. Check progress anytime
code-ingest task-progress .kiro/tasks/INGEST_20250929042515/
```

## 🏗️ **System Architecture**

### 1. **Master Task List** (`master-tasks.txt`)
```
TASK_001|1.1.1.1|Analyze INGEST_20250929042515 row 1|PENDING
TASK_002|1.1.1.2|Analyze INGEST_20250929042515 row 2|PENDING
TASK_003|1.1.1.3|Analyze INGEST_20250929042515 row 3|PENDING
...
TASK_1551|7.7.7.7|Analyze INGEST_20250929042515 row 1551|PENDING
```

### 2. **Current Window** (`current-window.md`)
```markdown
<!-- Window: Tasks 1-50 of 1551 total (3.2% complete) -->

- [ ] 1.1.1.1. Analyze INGEST_20250929042515 row 1
- [ ] 1.1.1.2. Analyze INGEST_20250929042515 row 2
- [ ] 1.1.1.3. Analyze INGEST_20250929042515 row 3
...
- [ ] 1.1.1.50. Analyze INGEST_20250929042515 row 50

<!-- Next window: Tasks 51-100 -->
```

### 3. **Progress Tracking** (`progress.json`)
```json
{
  "table_name": "INGEST_20250929042515",
  "total_tasks": 1551,
  "completed_tasks": 0,
  "current_window": {
    "start": 1,
    "end": 50,
    "status": "active"
  },
  "completion_percentage": 0.0,
  "created_at": "2025-09-29T10:30:00Z",
  "last_updated": "2025-09-29T10:30:00Z"
}
```

## 🔧 **Implementation**

### New CLI Commands:

#### 1. **Generate Windowed System**
```bash
code-ingest generate-hierarchical-tasks TABLE_NAME \
  --levels 4 --groups 7 --chunks 50 --windowed \
  --window-size 50 --output-dir .kiro/tasks/TABLE_NAME/
```

#### 2. **Advance Window**
```bash
code-ingest advance-window .kiro/tasks/TABLE_NAME/
# Moves current-window.md to show next 50 tasks
# Updates progress.json
# Archives completed window
```

#### 3. **Check Progress**
```bash
code-ingest task-progress .kiro/tasks/TABLE_NAME/
# Output: "Progress: 150/1551 tasks completed (9.7%)"
#         "Current window: Tasks 151-200"
#         "Estimated completion: 29 more windows"
```

#### 4. **Reset Window** (if needed)
```bash
code-ingest reset-window .kiro/tasks/TABLE_NAME/ --to 101
# Reset current window to start at task 101
```

## 🚀 **Rust Implementation**

### 1. **WindowedTaskManager**
```rust
pub struct WindowedTaskManager {
    task_dir: PathBuf,
    window_size: usize,
    total_tasks: usize,
    current_position: usize,
}

impl WindowedTaskManager {
    pub fn new(task_dir: PathBuf, window_size: usize) -> Self { ... }
    
    pub async fn generate_master_list(&self, tasks: Vec<Task>) -> Result<()> { ... }
    
    pub async fn create_current_window(&self) -> Result<()> { ... }
    
    pub async fn advance_window(&mut self) -> Result<bool> { ... }
    
    pub async fn get_progress(&self) -> Result<TaskProgress> { ... }
}
```

### 2. **Task Progress Tracking**
```rust
#[derive(Serialize, Deserialize)]
pub struct TaskProgress {
    pub table_name: String,
    pub total_tasks: usize,
    pub completed_tasks: usize,
    pub current_window: WindowInfo,
    pub completion_percentage: f64,
}

#[derive(Serialize, Deserialize)]
pub struct WindowInfo {
    pub start: usize,
    pub end: usize,
    pub status: WindowStatus,
}

#[derive(Serialize, Deserialize)]
pub enum WindowStatus {
    Active,
    Completed,
    Pending,
}
```

## 📋 **User Experience**

### **Day 1: Setup**
```bash
# Generate the windowed system (one-time setup)
code-ingest generate-hierarchical-tasks INGEST_20250929042515 \
  --levels 4 --groups 7 --chunks 50 --windowed \
  --output-dir .kiro/tasks/local-folder-analysis/

# Output:
# ✅ Generated master task list: 1,551 tasks
# ✅ Created current window: tasks 1-50
# ✅ Progress tracking initialized
# 📁 Files created in .kiro/tasks/local-folder-analysis/
```

### **Daily Work Sessions**
```bash
# Work on current 50 tasks in Kiro
kiro .kiro/tasks/local-folder-analysis/current-window.md

# When done with current window, advance
code-ingest advance-window .kiro/tasks/local-folder-analysis/

# Output:
# ✅ Window 1 completed (tasks 1-50)
# ✅ Advanced to window 2 (tasks 51-100)  
# 📊 Progress: 50/1551 tasks completed (3.2%)
# 📁 Completed tasks archived to completed/batch-001.md
```

### **Progress Monitoring**
```bash
code-ingest task-progress .kiro/tasks/local-folder-analysis/

# Output:
# 📊 Task Progress Report
# =====================
# Table: INGEST_20250929042515
# Total Tasks: 1,551
# Completed: 150 tasks (9.7%)
# Current Window: Tasks 151-200 (Window 4/32)
# Estimated Time: 28 more windows to complete
# 
# Recent Activity:
# ✅ Window 1: Completed 2025-09-29 10:30
# ✅ Window 2: Completed 2025-09-29 14:15  
# ✅ Window 3: Completed 2025-09-30 09:45
# 🔄 Window 4: In Progress (started 2025-09-30 11:20)
```

## 🎯 **Benefits**

1. **Single File Management**: Only one MD file to track in Kiro
2. **Automatic Progress**: System tracks completion automatically
3. **Resumable**: Can stop/start anytime, never lose progress
4. **Scalable**: Works for 50 tasks or 50,000 tasks
5. **Clean Interface**: Simple commands, clear status
6. **Archive**: Completed work is preserved for reference

This is much more elegant than managing 31 separate files!