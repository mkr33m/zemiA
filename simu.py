import numpy as np
import random
import matplotlib.pyplot as plt

M = 10
N = 200
CAPACITY = N // M
SUBJECTS = 5
RATIO = 20

SEED = 5433

rng = random.Random(SEED)
np.random.seed(SEED)

points = np.random.randint(0, 100001, size=(N, SUBJECTS))
total_scores = points.sum(axis=1)
order = np.argsort(-total_scores)
rank_index = np.empty(N, dtype=int)
rank_index[order] = np.arange(N)

TOP_THR    = RATIO * CAPACITY // 100
BOTTOM_THR = CAPACITY - TOP_THR

school_prefs = np.empty((N, M), dtype=int)
node_prefs   = [[] for _ in range(N)]

for rank, sid in enumerate(order):
    g              = rank // CAPACITY
    pos_in_group   = rank % CAPACITY

    pref = [(g + j) % M for j in range(M)]

    p_challenge = 1.0 - pos_in_group / (CAPACITY - 1)
    challenge   = rng.random() < p_challenge
    rot = 0
    if challenge:
        shift = 2 if pos_in_group < TOP_THR else 1
        if g >= shift:
            rot = shift
        elif shift == 2:
            shift -= 1
            if g >= shift:
                rot = shift
    else:
        shift = 1 if pos_in_group >= BOTTOM_THR else 0
        if g + shift < M:
            rot = -shift

    if rot != 0:
        pref = [pref[(j - rot) % M] for j in range(M)]

    school_prefs[sid] = pref
    node_pref = []
    for sch in pref:
        node_pref.extend([sch * CAPACITY + seat for seat in range(CAPACITY)])
    node_prefs[sid] = node_pref

def stable_matching(node_prefs, rank_index):
    match_to_node     = [-1] * N
    match_to_student  = [-1] * (M * CAPACITY)
    next_idx          = [0] * N

    from collections import deque
    q = deque(range(N))

    while q:
        s    = q.popleft()
        node = node_prefs[s][next_idx[s]]
        next_idx[s] += 1

        cur  = match_to_student[node]
        if cur == -1:
            match_to_node[s]    = node
            match_to_student[node] = s
        elif rank_index[s] < rank_index[cur]:
            match_to_node[s]    = node
            match_to_node[cur]  = -1
            match_to_student[node] = s
            q.append(cur)
        else:
            q.append(s)

    return match_to_node

match_stable = stable_matching(node_prefs, rank_index)

def sequential_matching(school_prefs, total_scores):
    enrolled    = [[] for _ in range(M)]
    next_choice = [0] * N
    waiting     = list(range(N))

    while waiting:
        applicants = [[] for _ in range(M)]
        for s in waiting:
            sch = school_prefs[s][next_choice[s]]
            next_choice[s] += 1
            applicants[sch].append(s)
        waiting = []

        for sch in range(M):
            if not applicants[sch]:
                continue
            free = CAPACITY - len(enrolled[sch])
            if free <= 0:
                waiting.extend(applicants[sch])
                continue
            applicants[sch].sort(key=lambda sid: -total_scores[sid])
            take = min(free, len(applicants[sch]))
            enrolled[sch].extend(applicants[sch][:take])
            if take < len(applicants[sch]):
                waiting.extend(applicants[sch][take:])

    school_of_seq = [-1] * N
    for sch in range(M):
        for s in enrolled[sch]:
            school_of_seq[s] = sch
    return school_of_seq

school_of_seq = sequential_matching(school_prefs, total_scores)

std_scores = np.linspace(70, 50, M)
score_norm = (std_scores - std_scores.min()) / (std_scores.max() - std_scores.min())

heat = np.zeros((N, M))
for sid in range(N):
    for j in range(M):
        heat[sid, j] = score_norm[school_prefs[sid, j]]

heat_ordered = heat[order]

fig1 = plt.figure(figsize=(6, 8))
plt.title("Students’ preference lists (darker = higher‑ranked school)")
plt.imshow(heat_ordered, aspect='auto', cmap='Greys', interpolation='nearest')
plt.xlabel("Preference order (1 … {})".format(M))
plt.ylabel("Students (high → low score)")
plt.colorbar(label="Relative school deviation value")
plt.tight_layout()

better = []
worse  = []
same   = []

for s in range(N):
    sch_stable = match_stable[s] // CAPACITY
    sch_seq    = school_of_seq[s]
    if sch_seq < sch_stable:
        better.append(s)
    elif sch_seq > sch_stable:
        worse.append(s)
    else:
        same.append(s)

group_of = rank_index // CAPACITY
group_colors = plt.cm.tab10(np.arange(M))

y_student = np.empty(N)
y_student[order] = np.arange(N) + 0.5
y_school = (np.arange(M) + 0.5) * N / M

fig2, axes = plt.subplots(1, 2, figsize=(12, 9), sharey=True)
titles = ["Stable matching", "Sequence algorithm"]
for k, ax in enumerate(axes):
    ax.set_title(titles[k])
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(N, 0)
    ax.axis('off')

    for s in range(N):
        sch = (match_stable[s] // CAPACITY) if k == 0 else school_of_seq[s]
        ax.plot([0, 1],
                [y_student[s], y_school[sch]],
                color=group_colors[group_of[s]],
                linewidth=0.6, alpha=0.7)

    for s in better:
        sch = (match_stable[s] // CAPACITY) if k == 0 else school_of_seq[s]
        ax.plot([0, 1], [y_student[s], y_school[sch]], color="red",  linewidth=1.3)
    for s in worse:
        sch = (match_stable[s] // CAPACITY) if k == 0 else school_of_seq[s]
        ax.plot([0, 1], [y_student[s], y_school[sch]], color="blue", linewidth=1.3)

    student_colors = group_colors[group_of]
    ax.scatter(np.zeros(N), y_student,
            s=18,
            color=student_colors,
            edgecolors='black',
            linewidths=0.3)
    ax.scatter(np.ones(M), y_school, s=60, color='black')

print("学校ランク比較（安定マッチング vs sequence 法）")
print(f"より良い学校に入学できた生徒: {len(better)} 名")
print(f"より悪い学校に入学した生徒: {len(worse)} 名")
print(f"同じ学校に入学した生徒  : {len(same)} 名")

plt.tight_layout()
plt.show()