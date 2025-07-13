#include <iostream>
#include <random> // random
#include <vector> // vector
#include <algorithm> // sort
#include <numeric> // iota
#include <cassert> // assert
#include <functional> // function
#include <queue> // queue

using namespace std;

/**
 * ★ M ... 学校数
 * ★ N ... 生徒数
 */
#define M 10
#define N 300
#define CAPACITY (N / M) // N%M == 0 で設定した方がいいかも

#define RATIO 20 // 上位・下位何パーセントが 2 校だけ上げる/下げるか

#define SUBJECTS 5
#define MAX_GPA 100
#define MALE 0 // 変えない
#define FEMALE 1 // 変えない

std::random_device rd;
std::mt19937 rng(rd());
std::normal_distribution<double> dist_dev(50.0, 10.0); // 正規分布

using Compare = function<bool(int, int)>;

struct Student {
private:
    vector<int> points;
    int gender;
    int GPA;
    int id;
    
public:
    Student() : points(SUBJECTS, -1), gender(-1), GPA(-1), id(-1) {}

    void set_point(int idx, int p) {
        assert(0 <= idx and idx < SUBJECTS);
        points[idx] = p;
    };
    void set_gender(int g) {
        assert(g == 0 or g == 1);
        gender = g;
    }
    void set_GPA(int G) {
        assert(0 <= G and G <= MAX_GPA);
        GPA = G;
    }
    void set_ID(int i) {
        assert(0 <= i);
        id = i;
    }

    int get_point(int idx) const { 
        return points[idx];
    }
    const vector<int>& get_points() const {
        return points;
    }
    int get_gender() const {
        return gender;
    }
    int get_GPA() const {
        return GPA;
    }
    int get_ID() const {
        return id;
    }
};

/**
 * （学校側の）各ノードで、生徒の順位付けを行う
 * 順位の持ち方は数字
 */
struct Node {
private:
    vector<int> rank_to_id_; // ranking[i] := i 位の生徒の id
    vector<int> id_to_rank_; // rank[id] := id の生徒のランク
    bool sorted;

public:
    Node() : rank_to_id_(N), id_to_rank_(N, 0), sorted(false) {
        iota(rank_to_id_.begin(), rank_to_id_.end(), 0);
    }

    void sort_(Compare cmp) {
        assert(!sorted);
        sorted = true;
        sort(rank_to_id_.begin(), rank_to_id_.end(), cmp);
        for (int r = 0; r < N; r++) {
            id_to_rank_[rank_to_id_[r]] = r;
        }
    }

    int rank_to_id(int rank) const {
        assert(0 <= rank and rank < N);
        return rank_to_id_[rank];
    }

    int id_to_rank(int id) const {
        assert(0 <= id and id < N);
        return id_to_rank_[id];
    }
};

struct School {
private:
    vector<Node> nodes;
    double standard_score; // 偏差値

public:
    School(double std_score = 50.0) : nodes(CAPACITY, Node()), standard_score(std_score) {}

    void divide(const vector<pair<Compare, int>>& cmps) {
        int sum = 0;
        for (auto [_, num] : cmps) {
            sum += num;
        }
        assert(sum == CAPACITY);

        int idx = 0;
        for (auto [cmp, num] : cmps) {
            for (int i = idx; i < idx + num; i++) {
                nodes[i].sort_(cmp);
            }
            idx += num;
        }
    }

    const vector<Node>& get_nodes() const {
        return nodes;
    }

    double get_standard_score() const {
        return standard_score;
    }
};

void stable_matching(
    const vector<vector<int>>& node_prefs, // 生徒たちが提出するランキング
    const vector<Node>& node_list, // ノードのリスト
    vector<int>& match_to_node) // 生徒 -> ノード
    {
    vector<int> match_to_student(N, -1); // ノード → 生徒
    vector<int> next_idx(N, 0);
    queue<int> q;
    
    for (int s = 0; s < N; s++) {
        q.push(s);
    }

    while (not q.empty()) {

        int s = q.front(); q.pop();
        // node：生徒 s が志望していてかつまだ空きがある学校のうち、もっともランキングが高いノード
        int node = node_prefs[s][next_idx[s]++];

        int cur = match_to_student[node];
        // 空いているなら
        if (cur == -1) {
            match_to_node[s] = node;
            match_to_student[node] = s;
        // 新しい生徒（s）の方が、node に対応する空席にとってより上位
        } else if (node_list[node].id_to_rank(s) < node_list[node].id_to_rank(cur)) {
            match_to_node[s] = node;
            match_to_node[cur] = -1;
            match_to_student[node] = s;
            q.push(cur); // 追い出された生徒
        } else {
            q.push(s); // 拒否されたから別ノードへ移動
        }
    }
}

int main() {

    uniform_int_distribution<int> score_d(0, 100);
    uniform_int_distribution<int> gender_d(0, 1);
    vector<Student> students(N); // 各生徒の情報
    vector<School> schools(M); // 各学校の情報

    // 生徒を生成する
    /**
     * ★ N 人の生徒の情報の登録
     * 生徒の ID == students における idx、としたいから students を並び替えるのは NG
     */
    for (int i = 0; i < N; i++) {
        students[i].set_ID(i);
        students[i].set_gender(gender_d(rng));
        students[i].set_GPA(score_d(rng));
        for (int sub = 0; sub < SUBJECTS; sub++) {
            students[i].set_point(sub, score_d(rng));
        }
    }

    vector<int> order(N);
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&](int a, int b) {
        int suma = 0, sumb = 0;
        for (int sub = 0; sub < SUBJECTS; sub++) {
            suma += students[a].get_point(sub);
            sumb += students[b].get_point(sub);
        }
        return suma > sumb;
    });

    // グループごとに志望校ランキングを組み立て
    vector<vector<int>> school_prefs(N, vector<int>(M));
    vector<vector<int>> node_prefs(N);
    uniform_real_distribution<double> uni(0.0, 1.0);

    const int TOP_THR = RATIO * CAPACITY / 100;
    const int BOTTOM_THR = CAPACITY - TOP_THR;

    // 成績が良い生徒から順に
    for (int rank = 0; rank < N; rank++) {
        int sid = order[rank]; // 生徒 ID
        int g = rank / CAPACITY; // グループ番号
        int pos_in_group = rank % CAPACITY; // グループ内順位

        for (int j = 0; j < M; j++) {
            school_prefs[sid][j] = (g + j) % M;
        }

        double p_challenge = 1.0 - double(pos_in_group) / (CAPACITY - 1);
        bool challenge = (uni(rng) < p_challenge);
        int rot = 0;

        if (challenge) {
            int shift = (pos_in_group < TOP_THR ? 2 : 1);
            if (g >= shift) {
                rot = +shift;
            }
        } else {
            int shift = (pos_in_group >= BOTTOM_THR ? 1 : 0);
            if (g + shift < M) {
                rot = -shift;
            }
        }
        
        if (rot != 0) {
            vector<int> tmp(M);
            for (int j = 0; j < M; ++j) {
                int jj = (j + rot + M) % M;
                tmp[jj] = school_prefs[sid][j];
            }
            school_prefs[sid].swap(tmp);
        }

        // ノードのランキング
        node_prefs[sid].reserve(N);
        for (int sch : school_prefs[sid]) {
            for (int seat = 0; seat < CAPACITY; seat++) {
                node_prefs[sid].push_back(sch * CAPACITY + seat);
            }
        }
    }

    // 学校側の、学生に対する優先順序の決定
    vector<int> total_score(N);
    for (int i = 0; i < N; i++) {
        int sum = 0;
        for (int sub = 0; sub < SUBJECTS; sub++) {
            sum += students[i].get_point(sub);
        }
        total_score[i] = sum;
    }
    auto cmp = [&](int a, int b) { return total_score[a] > total_score[b]; };

    vector<Node> node_list(N, Node());
    for (auto& nd : node_list) {
        nd.sort_(cmp);
    }

    // 安定マッチング
    vector<int> match_stable(N, -1); // 生徒 ⇒ ノード
    stable_matching(node_prefs, node_list, match_stable);

    // 第一志望 ⇒ 第二志望 ⇒ ... の形式
    vector<int> school_of_seq(N, -1); // 生徒 ⇒ 学校
    vector<vector<int>> enrolled(M); // 学校ごとの合格者
    vector<int> next_choice(N, 0); // 何番目の学校に出願中？

    vector<int> waiting(N);
    iota(waiting.begin(), waiting.end(), 0);

    while (!waiting.empty()) {
        vector<vector<int>> applicants(M); // 各学校の志願者
        for (int s : waiting) {
            int sch = school_prefs[s][next_choice[s]++];
            applicants[sch].push_back(s);
        }
        waiting.clear();

        for (int sch = 0; sch < M; sch++) {
            if (applicants[sch].empty()) {
                continue;
            }

            int free = CAPACITY - (int)enrolled[sch].size();
            if (free <= 0) {
                // 満員なら、学校 sch への志願者を全員 waiting のキューへ入れ直し
                waiting.insert(waiting.end(), applicants[sch].begin(), applicants[sch].end());
                continue;
            }
            sort(applicants[sch].begin(), applicants[sch].end(), cmp);
            // 受け入れ人数
            int take = min(free, (int)applicants[sch].size());
            // 学校 sch に対して生徒を入学させる
            enrolled[sch].insert(enrolled[sch].end(), applicants[sch].begin(), applicants[sch].begin() + take);
            
            if (take < (int)applicants[sch].size()) { // あふれた分は次ラウンドへ
                waiting.insert(waiting.end(), applicants[sch].begin() + take, applicants[sch].end());
            }
        }
    }

    for (int sch = 0; sch < M; ++sch) {
        for (int s : enrolled[sch]) {
            school_of_seq[s] = sch;
        }
    }

    int diff_sum = 0;
    int better = 0, worse = 0, same = 0;
    for (int stu = 0; stu < N; stu++) {
        int sch_stable = match_stable[stu] / CAPACITY;
        int sch_seq = school_of_seq[stu];
        if (sch_seq < sch_stable) {
            better++;
        } else if (sch_seq > sch_stable) {
            worse++;
        } else {
            same++;
        }

        diff_sum += abs(sch_seq - sch_stable);
    }

    cout << "より良い学校に入れた生徒 : " << better << '\n';
    cout << "より悪い学校に入れた生徒 : " << worse  << '\n';
    cout << "同じ学校に入れた生徒   : " << same   << '\n';

}