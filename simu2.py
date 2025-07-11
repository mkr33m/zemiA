import numpy as np
import matplotlib.pyplot as plt

# ----------------- Parameters (aligned with the C++ version) ------------------
M = 10
N = 1000
SUBJECTS = 5
CAPACITY = N // M

# ----------------- 1. Generate student data -----------------------------------
rng = np.random.default_rng()
students_scores = rng.integers(0, 101, size=(N, SUBJECTS))  # 0‥100
students_gender = rng.integers(0, 2, size=N)                # 0 = male, 1 = female
student_ids = np.arange(N)

# ----------------- 2. School “emphasize” pairs --------------------------------
EMP = np.array([
    [0, 1],  # school 0 emphasises subjects 0 and 1
    [1, 2],  # school 1
    [2, 3],
    [3, 4],
    [4, 0],
    [0, 2],
    [1, 3],
    [2, 4],
    [3, 0],
    [4, 1]
])

# ----------------- 3. Build node rankings -------------------------------------
# For each school we create CAPACITY nodes.
# First 20% nodes → weight on emp[0]
# Next 20% nodes → weight on emp[1]
# Remaining 60%  → uniform weighting
nodes_weights = []
for sch in range(M):
    sub1, sub2 = EMP[sch]
    A = CAPACITY * 20 // 100
    B = CAPACITY * 20 // 100
    for _ in range(A):
        w = np.ones(SUBJECTS, dtype=int)
        w[sub1] = 2
        nodes_weights.append(w)
    for _ in range(B):
        w = np.ones(SUBJECTS, dtype=int)
        w[sub2] = 2
        nodes_weights.append(w)
    for _ in range(CAPACITY - A - B):
        w = np.ones(SUBJECTS, dtype=int)
        nodes_weights.append(w)
nodes_weights = np.stack(nodes_weights)        # shape (N, SUBJECTS)

# ----------------- 4. Pre-compute each node’s ranking order & inverse rank ----
# score_matrix[node, student] = weighted sum
score_matrix = nodes_weights @ students_scores.T  # (N, N)

# Rankings: argsort descending
rankings = np.argsort(-score_matrix, axis=1)      # (N, N) each row is best→worst
# inverse rank: for Gale–Shapley preference comparisons
inv_rank = np.empty_like(rankings)
row_indices = np.repeat(np.arange(N)[:, None], N, axis=1)
inv_rank[row_indices, rankings] = np.arange(N)

# ----------------- 5. Build student preference list over nodes ---------------
# Each student orders schools by:
#   (matchCnt of top2 subjects in EMP) then (fit score)
student_prefs = np.empty((N, N), dtype=int)

top2 = np.argsort(-students_scores, axis=1)[:, :2]           # (N,2)

# compute matchCnt for each student/school
match_matrix = np.zeros((N, M), dtype=int)
fit_score_matrix = np.zeros((N, M), dtype=int)

for sch in range(M):
    sub1, sub2 = EMP[sch]
    # matches with top2
    match_matrix[:, sch] = ((top2 == sub1) | (top2 == sub2)).sum(axis=1)
    # fit score: weight=2 on emphasised subjects
    w = np.ones(SUBJECTS, dtype=int)
    w[sub1] = 2
    w[sub2] = 2
    fit_score_matrix[:, sch] = students_scores @ w

# sort schools for each student
for s in range(N):
    order = np.lexsort((-fit_score_matrix[s], -match_matrix[s]))
    # order is ascending by key → reverse
    order = order[::-1]
    # expand to nodes: for each school append its CAPACITY nodes in order
    idx = 0
    for sch in order:
        for seat in range(CAPACITY):
            student_prefs[s, idx] = sch * CAPACITY + seat
            idx += 1

# ----------------- 6. Gale-Shapley (students propose) -------------------------
free_students = list(range(N))
next_ptr = np.zeros(N, dtype=int)
match_to_node = -np.ones(N, dtype=int)   # student → node
match_to_student = -np.ones(N, dtype=int)  # node → student

while free_students:
    s = free_students.pop(0)
    node = student_prefs[s, next_ptr[s]]
    next_ptr[s] += 1

    current = match_to_student[node]
    if current == -1:
        match_to_node[s] = node
        match_to_student[node] = s
    else:
        # compare ranks: smaller = preferred
        if inv_rank[node, s] < inv_rank[node, current]:
            match_to_node[s] = node
            match_to_node[current] = -1
            match_to_student[node] = s
            free_students.append(current)
        else:
            free_students.append(s)

# Extract school assignments
assigned_school = match_to_node // CAPACITY       # (N,)

# ----------------- 7. Plot 1: Scores heat-map ---------------------------------
plt.figure()
plt.imshow(students_scores, aspect='auto')
plt.colorbar(label='Score')
plt.title('Student Scores (rows: students 0-999, columns: subjects 0-4)')
plt.xlabel('Subject')
plt.ylabel('Student ID')
plt.show()

# ----------------- 8. Plot 2: Assignment scatter -----------------------------
plt.figure()
plt.scatter(np.arange(N), assigned_school, s=4)
plt.title('School Assignment per Student')
plt.xlabel('Student ID')
plt.ylabel('Assigned School ID')
plt.yticks(range(M))
plt.show()
