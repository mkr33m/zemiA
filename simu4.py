import math
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Parameters (aligned with the C++ snippet) ------------------
M = 10
N = 1000
SUBJECTS = 5
CAPACITY = N // M
MALE, FEMALE = 0, 1

# ---------------- Generate student data (7:3 male/female) --------------------
rng = np.random.default_rng(42)  # reproducible
students_scores = rng.integers(0, 101, size=(N, SUBJECTS))
students_gender = rng.choice([MALE, FEMALE], size=N, p=[0.8, 0.2])
students_total = students_scores.sum(axis=1)

# ---------- Figure 1: total-score distribution by gender ---------------------
plt.figure(figsize=(6, 4))
plt.hist(students_total[students_gender == MALE], bins=20, alpha=0.6, label="Male")
plt.hist(students_total[students_gender == FEMALE], bins=20, alpha=0.6, label="Female")
plt.title("Total Score Distribution by Gender (7:3)")
plt.xlabel("Total Score")
plt.ylabel("Number of Students")
plt.legend()
plt.show()


# ---------------- Helper: run Gale–Shapley for quota factor k ---------------
def run_matching(k):
    female_quota = [CAPACITY * ((10 - sch) * k) // 100 for sch in range(M)]

    # Pre-sorted global rankings
    score_rank = np.lexsort((np.arange(N), -students_total))
    # Female-first ranking
    female_first_rank = np.lexsort((np.arange(N), -students_total, -students_gender))

    id_to_rank = []
    for sch in range(M):
        A = female_quota[sch]
        B = CAPACITY - A
        for _ in range(A):
            id_to_rank.append({sid: r for r, sid in enumerate(female_first_rank)})
        for _ in range(B):
            id_to_rank.append({sid: r for r, sid in enumerate(score_rank)})

    # Build student preference list over nodes
    node_prefs = np.empty((N, N), dtype=np.int32)
    for sid in range(N):
        idx = 0
        for sch in range(M):
            quota = female_quota[sch]
            if students_gender[sid] == FEMALE:
                for seat in range(quota):
                    node_prefs[sid, idx] = sch * CAPACITY + seat
                    idx += 1
                for seat in range(quota, CAPACITY):
                    node_prefs[sid, idx] = sch * CAPACITY + seat
                    idx += 1
            else:
                for seat in range(CAPACITY):
                    node_prefs[sid, idx] = sch * CAPACITY + seat
                    idx += 1

    # Gale–Shapley
    match_to_node = -np.ones(N, dtype=int)
    match_to_student = -np.ones(N, dtype=int)
    next_idx = np.zeros(N, dtype=int)
    free = list(range(N))
    while free:
        s = free.pop(0)
        node = node_prefs[s, next_idx[s]]
        next_idx[s] += 1
        cur = match_to_student[node]
        if cur == -1:
            match_to_node[s] = node
            match_to_student[node] = s
        else:
            if id_to_rank[node][s] < id_to_rank[node][cur]:
                match_to_node[s] = node
                match_to_node[cur] = -1
                match_to_student[node] = s
                free.append(cur)
            else:
                free.append(s)

    male_hist = np.zeros(M, dtype=int)
    female_hist = np.zeros(M, dtype=int)
    for sid in range(N):
        school = match_to_node[sid] // CAPACITY
        if students_gender[sid] == FEMALE:
            female_hist[school] += 1
        else:
            male_hist[school] += 1
    return female_quota, male_hist, female_hist

# ---------------- Run for k = 0 .. 10 ----------------------------------------
k_values = list(range(11))
results = [run_matching(k) for k in k_values]

# ---------------- Plot grid --------------------------------------------------
cols = 3
rows = math.ceil(len(k_values) / cols)
fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), sharey=True)
axes = axes.flatten()

for idx, (k, (quota, male_hist, female_hist)) in enumerate(zip(k_values, results)):
    ax = axes[idx]
    x = np.arange(M)
    ax.bar(x - 0.2, male_hist, width=0.4, label='Male', color='tab:blue')
    ax.bar(x + 0.2, female_hist, width=0.4, label='Female', color='tab:orange')
    ax.set_title(f'k={k}  (Sch0 female seats {quota[0]})')
    ax.set_xlabel('School ID')
    if idx % cols == 0:
        ax.set_ylabel('Matched Students')
    ax.set_xticks(x)
    ax.set_ylim(0, 150)

# Hide unused axes
for ax in axes[len(k_values):]:
    ax.axis('off')

axes[0].legend(loc='upper right')
plt.tight_layout()
plt.show()
