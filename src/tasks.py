"""Task Registry for the OpenEnv code review environment."""

from dataclasses import dataclass, field
from typing import Literal
from itertools import cycle


@dataclass
class TaskDefinition:
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    pr_title: str
    pr_description: str
    diff: str
    file_count: int
    additions: int
    deletions: int
    ground_truth_decision: Literal["approve", "request_changes", "comment"]
    required_keywords: list[str]
    required_inline_file: str | None
    rubric: dict[str, float]


# ---------------------------------------------------------------------------
# Easy Task: Off-by-one error in a loop
# ---------------------------------------------------------------------------
_EASY_TASK = TaskDefinition(
    task_id="easy-001",
    difficulty="easy",
    pr_title="Add pagination to user list endpoint",
    pr_description=(
        "Adds pagination support to GET /users. "
        "The page size is configurable via a query parameter."
    ),
    diff="""\
diff --git a/api/users.py b/api/users.py
index 1a2b3c4..5d6e7f8 100644
--- a/api/users.py
+++ b/api/users.py
@@ -12,7 +12,10 @@ def get_users(page: int = 1, page_size: int = 20):
     users = db.query(User).all()
-    return users
+    start = (page - 1) * page_size
+    end = start + page_size
+    # Return the paginated slice
+    return users[start:end + 1]
""",
    file_count=1,
    additions=4,
    deletions=1,
    ground_truth_decision="request_changes",
    required_keywords=["off-by-one", "end + 1"],
    required_inline_file=None,
    rubric={"decision_correctness": 0.4, "issue_identification": 0.4, "review_quality": 0.2},
)

# ---------------------------------------------------------------------------
# Medium Task: SQL injection vulnerability
# ---------------------------------------------------------------------------
_MEDIUM_TASK = TaskDefinition(
    task_id="medium-001",
    difficulty="medium",
    pr_title="Add user search by username",
    pr_description=(
        "Implements a search endpoint that filters users by username substring. "
        "Uses raw SQL for performance."
    ),
    diff="""\
diff --git a/api/search.py b/api/search.py
index 2b3c4d5..6e7f8a9 100644
--- a/api/search.py
+++ b/api/search.py
@@ -1,5 +1,14 @@
+from db import get_connection
+
+def search_users(username_query: str):
+    conn = get_connection()
+    cursor = conn.cursor()
+    # Search for users matching the query
+    sql = f"SELECT * FROM users WHERE username LIKE '%{username_query}%'"
+    cursor.execute(sql)
+    return cursor.fetchall()
""",
    file_count=1,
    additions=9,
    deletions=0,
    ground_truth_decision="request_changes",
    required_keywords=["sql injection", "parameterized", "f-string", "user input"],
    required_inline_file="api/search.py",
    rubric={"decision_correctness": 0.4, "issue_identification": 0.4, "review_quality": 0.2},
)

# ---------------------------------------------------------------------------
# Hard Task: N+1 query performance issue
# ---------------------------------------------------------------------------
_HARD_TASK = TaskDefinition(
    task_id="hard-001",
    difficulty="hard",
    pr_title="Display order history with product details",
    pr_description=(
        "Renders a user's order history page, showing each order alongside "
        "the product name and category for every line item."
    ),
    diff="""\
diff --git a/views/orders.py b/views/orders.py
index 3c4d5e6..7f8a9b0 100644
--- a/views/orders.py
+++ b/views/orders.py
@@ -5,8 +5,18 @@ from models import Order, Product
 def get_order_history(user_id: int):
     orders = Order.objects.filter(user_id=user_id)
+    result = []
+    for order in orders:
+        items = []
+        for line_item in order.line_items.all():
+            # Fetch product details for each line item individually
+            product = Product.objects.get(id=line_item.product_id)
+            items.append({
+                "product_name": product.name,
+                "category": product.category,
+                "quantity": line_item.quantity,
+            })
+        result.append({"order_id": order.id, "items": items})
-    return orders
+    return result
""",
    file_count=1,
    additions=13,
    deletions=1,
    ground_truth_decision="request_changes",
    required_keywords=["n+1", "select_related", "prefetch_related", "query per", "eager loading"],
    required_inline_file=None,
    rubric={"decision_correctness": 0.4, "issue_identification": 0.4, "review_quality": 0.2},
)

# ---------------------------------------------------------------------------
# Easy Task 2: Unused variable / dead code
# ---------------------------------------------------------------------------
_EASY_TASK_2 = TaskDefinition(
    task_id="easy-002",
    difficulty="easy",
    pr_title="Refactor discount calculation",
    pr_description=(
        "Refactors the discount logic to support tiered pricing. "
        "Cleans up the old flat-rate discount function."
    ),
    diff="""\
diff --git a/billing/discount.py b/billing/discount.py
index 4d5e6f7..8a9b0c1 100644
--- a/billing/discount.py
+++ b/billing/discount.py
@@ -1,8 +1,16 @@
 def calculate_discount(price: float, tier: str) -> float:
-    discount = 0.0
-    return price
+    discount_rate = 0.0
+    if tier == "silver":
+        discount_rate = 0.10
+    elif tier == "gold":
+        discount_rate = 0.20
+    elif tier == "platinum":
+        discount_rate = 0.30
+    final_price = price * (1 - discount_rate)
+    unused_result = price * discount_rate  # never used
+    return final_price
""",
    file_count=1,
    additions=9,
    deletions=2,
    ground_truth_decision="request_changes",
    required_keywords=["unused", "dead code", "unused_result"],
    required_inline_file=None,
    rubric={"decision_correctness": 0.4, "issue_identification": 0.4, "review_quality": 0.2},
)

# ---------------------------------------------------------------------------
# Easy Task 3: Missing input validation / no error handling
# ---------------------------------------------------------------------------
_EASY_TASK_3 = TaskDefinition(
    task_id="easy-003",
    difficulty="easy",
    pr_title="Add divide endpoint to calculator API",
    pr_description=(
        "Adds a /divide endpoint that divides two numbers provided as query params."
    ),
    diff="""\
diff --git a/api/calculator.py b/api/calculator.py
index 5e6f7a8..9b0c1d2 100644
--- a/api/calculator.py
+++ b/api/calculator.py
@@ -0,0 +1,8 @@
+from fastapi import APIRouter
+
+router = APIRouter()
+
+@router.get("/divide")
+def divide(a: float, b: float) -> float:
+    return a / b
""",
    file_count=1,
    additions=7,
    deletions=0,
    ground_truth_decision="request_changes",
    required_keywords=["division by zero", "zero", "ZeroDivisionError", "validate"],
    required_inline_file=None,
    rubric={"decision_correctness": 0.4, "issue_identification": 0.4, "review_quality": 0.2},
)

# ---------------------------------------------------------------------------
# Medium Task 2: Hardcoded credentials / secret in source
# ---------------------------------------------------------------------------
_MEDIUM_TASK_2 = TaskDefinition(
    task_id="medium-002",
    difficulty="medium",
    pr_title="Add email notification service",
    pr_description=(
        "Integrates SendGrid to send transactional emails. "
        "Adds a helper module for sending password reset emails."
    ),
    diff="""\
diff --git a/services/email.py b/services/email.py
index 6f7a8b9..0c1d2e3 100644
--- a/services/email.py
+++ b/services/email.py
@@ -0,0 +1,18 @@
+import sendgrid
+from sendgrid.helpers.mail import Mail
+
+# SendGrid API key
+SENDGRID_API_KEY = "SG.abc123xyz.secretkey_hardcoded_here"
+SENDER_EMAIL = "noreply@company.com"
+
+def send_password_reset(to_email: str, reset_link: str) -> bool:
+    sg = sendgrid.SendGridAPIClient(api_key=SENDGRID_API_KEY)
+    message = Mail(
+        from_email=SENDER_EMAIL,
+        to_emails=to_email,
+        subject="Password Reset",
+        html_content=f"<a href='{reset_link}'>Reset your password</a>",
+    )
+    response = sg.send(message)
+    return response.status_code == 202
""",
    file_count=1,
    additions=17,
    deletions=0,
    ground_truth_decision="request_changes",
    required_keywords=["hardcoded", "secret", "environment variable", "credentials"],
    required_inline_file="services/email.py",
    rubric={"decision_correctness": 0.4, "issue_identification": 0.4, "review_quality": 0.2},
)

# ---------------------------------------------------------------------------
# Medium Task 3: Race condition / missing lock
# ---------------------------------------------------------------------------
_MEDIUM_TASK_3 = TaskDefinition(
    task_id="medium-003",
    difficulty="medium",
    pr_title="Add concurrent download counter",
    pr_description=(
        "Tracks the number of active file downloads using a shared counter. "
        "Increments on download start, decrements on completion."
    ),
    diff="""\
diff --git a/downloads/tracker.py b/downloads/tracker.py
index 7a8b9c0..1d2e3f4 100644
--- a/downloads/tracker.py
+++ b/downloads/tracker.py
@@ -0,0 +1,14 @@
+active_downloads = 0
+
+def start_download(file_id: str) -> None:
+    global active_downloads
+    active_downloads += 1
+    _do_download(file_id)
+
+def finish_download(file_id: str) -> None:
+    global active_downloads
+    active_downloads -= 1
+
+def get_active_count() -> int:
+    return active_downloads
""",
    file_count=1,
    additions=13,
    deletions=0,
    ground_truth_decision="request_changes",
    required_keywords=["race condition", "thread", "atomic", "lock", "concurrent"],
    required_inline_file="downloads/tracker.py",
    rubric={"decision_correctness": 0.4, "issue_identification": 0.4, "review_quality": 0.2},
)

# ---------------------------------------------------------------------------
# Hard Task 2: Missing database index on foreign key
# ---------------------------------------------------------------------------
_HARD_TASK_2 = TaskDefinition(
    task_id="hard-002",
    difficulty="hard",
    pr_title="Add activity feed for user dashboard",
    pr_description=(
        "Implements an activity feed showing recent actions by users the current "
        "user follows. Queries events table filtered by followed user IDs."
    ),
    diff="""\
diff --git a/models/activity.py b/models/activity.py
index 8b9c0d1..2e3f4a5 100644
--- a/models/activity.py
+++ b/models/activity.py
@@ -0,0 +1,20 @@
+from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
+from sqlalchemy.orm import relationship
+from database import Base
+
+class ActivityEvent(Base):
+    __tablename__ = "activity_events"
+
+    id = Column(Integer, primary_key=True)
+    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
+    event_type = Column(String(50), nullable=False)
+    payload = Column(String(500))
+    created_at = Column(DateTime, nullable=False)
+
+    user = relationship("User", back_populates="activities")
+
+def get_feed(followed_ids: list[int], limit: int = 50):
+    return (
+        ActivityEvent.query
+        .filter(ActivityEvent.user_id.in_(followed_ids))
+        .order_by(ActivityEvent.created_at.desc())
+        .limit(limit)
+        .all()
+    )
""",
    file_count=1,
    additions=21,
    deletions=0,
    ground_truth_decision="request_changes",
    required_keywords=["index", "db_index", "missing index", "full table scan", "query performance"],
    required_inline_file=None,
    rubric={"decision_correctness": 0.4, "issue_identification": 0.4, "review_quality": 0.2},
)

# ---------------------------------------------------------------------------
# Hard Task 3: Unbounded memory growth / missing pagination on bulk load
# ---------------------------------------------------------------------------
_HARD_TASK_3 = TaskDefinition(
    task_id="hard-003",
    difficulty="hard",
    pr_title="Export all transactions to CSV",
    pr_description=(
        "Adds a /export/transactions endpoint that returns all transactions "
        "for an account as a downloadable CSV file."
    ),
    diff="""\
diff --git a/api/export.py b/api/export.py
index 9c0d1e2..3f4a5b6 100644
--- a/api/export.py
+++ b/api/export.py
@@ -0,0 +1,22 @@
+import csv
+import io
+from fastapi import APIRouter
+from fastapi.responses import StreamingResponse
+from models import Transaction
+
+router = APIRouter()
+
+@router.get("/export/transactions")
+def export_transactions(account_id: int):
+    # Load all transactions into memory
+    transactions = Transaction.query.filter_by(account_id=account_id).all()
+    output = io.StringIO()
+    writer = csv.writer(output)
+    writer.writerow(["id", "amount", "date", "description"])
+    for tx in transactions:
+        writer.writerow([tx.id, tx.amount, tx.date, tx.description])
+    output.seek(0)
+    return StreamingResponse(
+        iter([output.getvalue()]),
+        media_type="text/csv",
+    )
""",
    file_count=1,
    additions=21,
    deletions=0,
    ground_truth_decision="request_changes",
    required_keywords=["memory", "pagination", "streaming", "unbounded", "batch"],
    required_inline_file=None,
    rubric={"decision_correctness": 0.4, "issue_identification": 0.4, "review_quality": 0.2},
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_ALL_TASKS: list[TaskDefinition] = [
    _EASY_TASK, _EASY_TASK_2, _EASY_TASK_3,
    _MEDIUM_TASK, _MEDIUM_TASK_2, _MEDIUM_TASK_3,
    _HARD_TASK, _HARD_TASK_2, _HARD_TASK_3,
]

_TASKS_BY_DIFFICULTY: dict[str, list[TaskDefinition]] = {
    "easy": [_EASY_TASK, _EASY_TASK_2, _EASY_TASK_3],
    "medium": [_MEDIUM_TASK, _MEDIUM_TASK_2, _MEDIUM_TASK_3],
    "hard": [_HARD_TASK, _HARD_TASK_2, _HARD_TASK_3],
}

# Round-robin state: cycle through difficulties in order
_ROUND_ROBIN_ORDER = ["easy", "medium", "hard"]
_round_robin_iter = cycle(_ROUND_ROBIN_ORDER)
# Per-difficulty index for cycling within a difficulty group
_difficulty_index: dict[str, int] = {"easy": 0, "medium": 0, "hard": 0}


def get_task(difficulty: str | None = None) -> TaskDefinition:
    """Return a task for the given difficulty, or cycle round-robin if None."""
    if difficulty is not None:
        difficulty = difficulty.lower()
        if difficulty not in _TASKS_BY_DIFFICULTY:
            raise ValueError(f"difficulty must be one of: easy, medium, hard")
        tasks = _TASKS_BY_DIFFICULTY[difficulty]
        idx = _difficulty_index[difficulty]
        task = tasks[idx % len(tasks)]
        _difficulty_index[difficulty] = (idx + 1) % len(tasks)
        return task

    # Round-robin across difficulties
    diff = next(_round_robin_iter)
    tasks = _TASKS_BY_DIFFICULTY[diff]
    idx = _difficulty_index[diff]
    task = tasks[idx % len(tasks)]
    _difficulty_index[diff] = (idx + 1) % len(tasks)
    return task


def get_all_tasks() -> list[TaskDefinition]:
    """Return all registered tasks."""
    return list(_ALL_TASKS)
