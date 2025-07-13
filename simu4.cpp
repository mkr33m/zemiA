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
#define N 1000
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

/**
 * 各区切りで必要になるのは、
 * ・上から何番目の区切りか？
 * ・各生徒（これは外から入力したらよくて、偏差値順と仮定）
 * ・各生徒の志望校ランキング
 * ・上にいる生徒ほどアグレッシブだと仮定（これも外部から入力したい。）
 */

uniform_int_distribution<int> ran(0, 100);

/**
 * school_prefs[id][i] := 生徒 id が i 番目に志望する学校
 * school_prefs（志望校ランキング）を、ノードのランキング（pref）に変換する
 * pref[id][i] := 生徒 id が i 番目に志望するノード
 */
vector<vector<int>> build_student_prefer(
    const vector<vector<int>>& school_prefs,
    const vector<Student>& students) 
    {
    assert(school_prefs.size() == N);

    vector<vector<int>> node_prefs(N, vector<int>(N));
    for (int s = 0; s < N; s++) { // 全生徒を見ていく
        int idx = 0;
        // school：生徒 s が提出する志望校ランキング
        for (int school : school_prefs[s]) {
            for (int seat = 0; seat < CAPACITY; seat++) {
                node_prefs[s][idx++] = school * CAPACITY + seat; 
            }
        }
    }
    for (int s = 0; s < N; s++) { // 全生徒を見ていく
        int idx = 0;
        
        // school：生徒 s が提出する志望校ランキング
        for (int school : school_prefs[s]) { 
            for (int seat = 0; seat < CAPACITY; seat++) {
                node_prefs[s][idx++] = school * CAPACITY + seat;
            }
        }
    }
    return node_prefs;
}

struct Range {
private:
    int id; // 上から何番目の区切りか
    vector<Student> students; // 各生徒
    vector<vector<int>> school_prefs; // 各生徒の志望校ランキング（最初は一意で、aggressive_prod に応じてノイズを入れていく）
    vector<vector<int>> node_prefs;
    vector<int> aggressive_probs; // CAPACITY 人の上位校を目指す確率

    void rotate(int stu, int rot) { // 下方向へどれだけ回転するか
        assert(-M < rot and rot < M);
        vector<int> new_prefs(M);
        for (int i = 0; i < M; i++) {
            int j = (i + rot) % M;
            new_prefs[j] = school_prefs[stu][i];
        }
        school_prefs[stu].swap(new_prefs);
    }

public:
    // 1 区切りで N/M (== CAPACITY) 人の生徒が存在する
    Range(int id) : id(id), students(0), school_prefs(CAPACITY, vector<int>(M)), node_prefs(CAPACITY, vector<int>(N)) {
        int seed = ran(rng);
        std::uniform_int_distribution<int> probs(-seed, seed);
        for (int i = 0; i < N; i++) {
            aggressive_probs.push_back(probs(rng));
        }
        sort(aggressive_probs.begin(), aggressive_probs.end());

        int num = RATIO * CAPACITY / 100; // 上位・下位何人をよりアグレッシブにするか  
        // 生徒のランキングを設定
        for (int stu = 0; stu < CAPACITY; stu++) {
            for (int i = 0; i < M; i++) {
                school_prefs[stu][i] = (i + id) % M;
            }
            int P = probs(rng);
            // アグレッシブを考慮して rotate
            if (aggressive_probs[stu] > 0) { // 正
                if (P >= 0 and P <= aggressive_probs[stu]) {
                    if (stu < num) {
                        rotate(stu, -2);
                    } else {
                        rotate(stu, -1);
                    }
                }
            } else if (aggressive_probs[stu] < 0) {
                if (P <= 0 and -P <= -aggressive_probs[stu]) {
                    if (CAPACITY - stu < num) {
                        rotate(stu, 2);
                    } else {
                        rotate(stu, 1);
                    }
                }
            }
        }

        // 学校の志望校ランキングをノードの志望校ランキングに変換する
        node_prefs = build_student_prefer(school_prefs, students);
    };

    void add_student(const Student& stu) {
        students.push_back(stu);
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

    std::uniform_int_distribution<int> score_d(0, 100);
    std::discrete_distribution<int> gender_d(0, 1);
    vector<Student> students(N); // 各生徒の情報
    vector<Range> ranges(M); // 各 Range を格納
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

    // Range を生成する
    for (int i = 0; i < M; i++) {
        ranges[i] = Range(i);
    }

    // 各学校の偏差値の決定（ここは変更しない）
    vector<double> scores(M);
    for (double& v : scores) {
        v = dist_dev(rng);
    }

    sort(scores.begin(), scores.end(), greater<>());

    for (double sc : scores) { // 偏差値が高い学校ほど idx が小さい
        schools.emplace_back(sc);
    }

    // 各 Range に生徒を追加する
    {
        int id = 0;
        for (int R = 0; R < M; R++) {
            for (int stu = R * CAPACITY; stu < (R + 1) * CAPACITY; stu++) {
                ranges[R].add_student(students[id++]);
            }
        }
    }

    vector<Node> node_list; // 各学校からノードを取り出し、番号を割り当てる
    node_list.reserve(N);
    for (int sch = 0; sch < M; sch++) {
        const auto& v = schools[sch].get_nodes();
        node_list.insert(node_list.end(), v.begin(), v.end());
    }



}