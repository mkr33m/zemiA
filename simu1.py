import random
import numpy as np
import matplotlib.pyplot as plt

# --- Parameters (aligned with the C++ code) -----------------------------
M = 10         # number of schools
N = 1000       # number of students
CAPACITY = N // M
SUBJECTS = 5
MAX_GPA = 100
MALE, FEMALE = 0, 1

# --- Helper: generate student data -------------------------------------
students_scores = np.random.randint(0, 101, size=(N, SUBJECTS))
students_gender = np.random.randint(0, 2, size=N)
students_total = students_scores.sum(axis=1)

# --- Build schools & quotas --------------------------------------------
female_quota = [CAPACITY * ((10 - sch) * 3) // 100 for sch in range(M)]
# schools' node ranking: lists of (student index)
node_list_rank = []
id_to_rank = []

def cmp1(a, b):
    # higher total score first, tie-break on smaller ID
    ta, tb = students_total[a], students_total[b]
    return (ta > tb) or (ta == tb and a < b)

def cmp2(a, b):
    # prioritize females, then cmp1 logic
    ga, gb = students_gender[a], students_gender[b]
    if ga != gb:
        return ga > gb  # FEMALE==1 is “larger”; return true if a should come first
    return cmp1(a, b)

# produce sorted rankings for each node
for sch in range(M):
    A = female_quota[sch]
    B = CAPACITY - A
    rank_female = sorted(range(N), key=lambda x: (
        -students_gender[x],  # females (1) first
        -students_total[x],   # higher score
        x                     # lower ID
    ))
    rank_score = sorted(range(N), key=lambda x: (
        -students_total[x],
        x
    ))

    # first A nodes: cmp2 ranking, next B nodes: cmp1 ranking
    for _ in range(A):
        node_list_rank.append(rank_female)
        id_to_rank.append({sid: r for r, sid in enumerate(rank_female)})
    for _ in range(B):
        node_list_rank.append(rank_score)
        id_to_rank.append({sid: r for r, sid in enumerate(rank_score)})

# --- Build student preference lists (node indices) ----------------------
node_prefs = np.empty((N, N), dtype=int)
for sid in range(N):
    idx = 0
    for sch in range(M):
        quota = female_quota[sch]
        if students_gender[sid] == FEMALE:
            # female seats first
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

# --- Gale-Shapley stable matching (students propose) --------------------
match_to_node = np.full(N, -1, dtype=int)
match_to_student = np.full(N, -1, dtype=int)
next_idx = np.zeros(N, dtype=int)

free_students = list(range(N))
while free_students:
    sid = free_students.pop(0)
    node = node_prefs[sid, next_idx[sid]]
    next_idx[sid] += 1

    current = match_to_student[node]
    if current == -1:
        # node empty
        match_to_node[sid] = node
        match_to_student[node] = sid
    else:
        # check preference of the node
        if id_to_rank[node][sid] < id_to_rank[node][current]:
            # node prefers new student
            match_to_node[sid] = node
            match_to_node[current] = -1
            match_to_student[node] = sid
            # old student becomes free again
            free_students.append(current)
        else:
            # node rejects; student stays free
            free_students.append(sid)

# --- Build histograms of matched school rank ---------------------------
male_hist = np.zeros(M, dtype=int)
female_hist = np.zeros(M, dtype=int)

school_prefs = [list(range(M)) for _ in range(N)]  # straight 0..9

for sid in range(N):
    node = match_to_node[sid]
    school = node // CAPACITY
    rank = school_prefs[sid].index(school)
    if students_gender[sid] == FEMALE:
        female_hist[rank] += 1
    else:
        male_hist[rank] += 1

# --- Plot 1: Score distribution by gender ------------------------------
plt.figure()
plt.hist(students_total[students_gender == MALE], bins=20, alpha=0.5, label="Male")
plt.hist(students_total[students_gender == FEMALE], bins=20, alpha=0.5, label="Female")
plt.title("Total Score Distribution by Gender")
plt.xlabel("Total Score")
plt.ylabel("Number of Students")
plt.legend()
plt.show()

# --- Plot 2: Preference rank outcomes (male vs female) -----------------
plt.figure()
ranks = np.arange(1, M + 1)
width = 0.4
plt.bar(ranks - width / 2, male_hist, width=width, label="Male")
plt.bar(ranks + width / 2, female_hist, width=width, label="Female")
plt.title("Matched School Preference Rank Histogram")
plt.xlabel("Preference Rank (1 = top choice)")
plt.ylabel("Number of Students")
plt.xticks(ranks)
plt.legend()
plt.show()
