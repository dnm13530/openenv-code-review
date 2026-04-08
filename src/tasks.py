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
    rubric={"decision_correctness": 1.0, "issue_identification": 0.5, "review_quality": 0.5},
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
    rubric={"decision_correctness": 1.0, "issue_identification": 0.5, "review_quality": 0.5},
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
    rubric={"decision_correctness": 1.0, "issue_identification": 0.5, "review_quality": 0.5},
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_ALL_TASKS: list[TaskDefinition] = [_EASY_TASK, _MEDIUM_TASK, _HARD_TASK]

_TASKS_BY_DIFFICULTY: dict[str, list[TaskDefinition]] = {
    "easy": [_EASY_TASK],
    "medium": [_MEDIUM_TASK],
    "hard": [_HARD_TASK],
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
