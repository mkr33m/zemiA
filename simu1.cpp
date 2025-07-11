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
➀
生徒 ... 正直に偏差値が高い学校順にランキングを設定
学校 ... 合計点数が高い生徒順にランキングを設定
→ 1 位 100人、2 位 100 人、...（それはそうの結果）
➁
生徒 ... そのまま
学校 ... 各学校について、女子枠を設ける。偏差値が高い学校ほど女子枠の人数が多いものと仮定する
→ 男女別で幸福度を分けて観察してみる
➂
生徒 ... 最高得点を取った科目を重要視してくれる学校をランキングの上位に配置する。 
学校 ... 各学校について、2 科目をピックアップしてその科目を重視するような重み付けをした枠を用意する。
→ 単純に幸福度を観察してみる
 */

/**
 * ★ M ... 学校数
 * ★ N ... 生徒数
 */
#define M 10
#define N 1000
#define CAPACITY (N / M) // N%M == 0 で設定した方がいいかも

#define SUBJECTS 5
#define MAX_GPA 100
#define MALE 0 // 変えない
#define FEMALE 1 // 変えない

using Compare = function<bool(int, int)>;

/**
 * 各生徒が持つ情報は
 * ・統一テストの教科ごとの成績（5 科目と仮定する）
 * ・性別
 * ・内申点
 * ・生徒 ID
 * の 4 つとする。各情報はセッターで設定する。
 */

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
 * 学校側と生徒側のマッチング
 * 学校側で順序を決定しておく。
 */
/**
 * どういう持ち方にするか？
 * ノードの個数は CAPACITY 個
 */
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

/**
 * school_prefs[id][i] := 生徒 id が i 番目に志望する学校
 * school_prefs（志望校ランキング）を、ノードのランキング（pref）に変換する
 * pref[id][i] := 生徒 id が i 番目に志望するノード
 */
vector<vector<int>> build_student_prefer(
    const vector<vector<int>>& school_prefs,
    const vector<Student>& students,
    const vector<int>& female_quota) 
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
        bool is_female = (students[s].get_gender() == FEMALE);
        int idx = 0;
        
        // school：生徒 s が提出する志望校ランキング
        for (int school : school_prefs[s]) { 
            int A = female_quota[school];

            if (is_female) {
                // 女子枠ノードを先に
                for (int seat = 0; seat < A; ++seat) {
                    node_prefs[s][idx++] = school * CAPACITY + seat;
                }
                // 一般枠ノードを後ろに
                for (int seat = A; seat < CAPACITY; ++seat) {
                    node_prefs[s][idx++] = school * CAPACITY + seat;
                }
            } else {
                for (int seat = 0; seat < CAPACITY; seat++) {
                    node_prefs[s][idx++] = school * CAPACITY + seat;
                }
            }
        }
    }
    return node_prefs;
}

/**
 * 安定マッチングの実装
 * 生徒 1 人につき 1 ノードをマッチングさせる
 * ノードが学校に CAPACITY 個ずつある。
 */
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
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> score_d(0, 100);
    std::uniform_int_distribution<int> gender_d(0, 1);
    std::normal_distribution<double> dist_dev(50.0, 10.0); // 正規分布

    vector<Student> students(N); // 各生徒の情報
    vector<School> schools; // 各学校の情報
    vector<int> female_quota(M); // 各学校の女子枠の定員
    schools.reserve(M);
    // 全生徒について、提出する学校の志望ランキング
    // school_prefs[id][i] := id の生徒が i 番目に志望する学校
    vector<vector<int>> school_prefs(N, vector<int>(M));

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

    // 各学校の偏差値の決定（ここは変更しない）
    vector<double> scores(M);
    for (double& v : scores) {
        v = dist_dev(rng);
    }

    sort(scores.begin(), scores.end(), greater<>());

    for (double sc : scores) { // 偏差値が高い学校ほど idx が小さい
        schools.emplace_back(sc);
    }

    /**
     * ★ M 校の学校について、定員 CAPACITY に対応するノードが保持する志望生徒ランキングの決め方
     */
    for (int sch = 0; sch < M; sch++) {
        /**
         * ➀
         */
        Compare cmp1 = [&](int a, int b) {
            int suma = 0, sumb = 0;
            for (int sub = 0; sub < SUBJECTS; sub++) {
                suma += students[a].get_point(sub);
                sumb += students[b].get_point(sub);
            }
            if (suma != sumb) {
                return suma > sumb; // 成績高い順
            }
            return students[a].get_ID() < students[b].get_ID(); // タイブレーク
        };
        /**
         * ➁
         */
        Compare cmp2 = [&](int a, int b) {
            int gender_a = students[a].get_gender();
            int gender_b = students[b].get_gender();
            if (gender_a != gender_b) { // 女子を優先
                return gender_a > gender_b;
            }
            // 性別が同じなら、成績が高い生徒を優先
            int sum_a = 0, sum_b = 0;
            for (int sub = 0; sub < SUBJECTS; sub++) {
                sum_a += students[a].get_point(sub);
                sum_b += students[b].get_point(sub);
            }
            if (sum_a != sum_b) {
                return sum_a > sum_b; // 成績高い順
            }
            return students[a].get_ID() < students[b].get_ID(); // タイブレーク（総合点も同じなら、id が小さい生徒を優先）
        };

        /**
         * ➀
         */
        /* schools[sch].divide({ {cmp1, CAPACITY * rat / 100} }); */

        /**
         * ➁
         */
        int rat = (10 - sch) * 3; // この％だけ女子枠を取る（sch == 0 なら 30 パー、sch == 1 なら 27 パー、...）
        int A = CAPACITY * rat / 100;
        int B = CAPACITY - A;
        schools[sch].divide({ {cmp2, A}, {cmp1, B} });
        female_quota[sch] = A;
    }

    /**
     * ★ N 人の生徒の志望校ランキングの決め方
     */
    for (int s = 0; s < N; s++) {
        iota(school_prefs[s].begin(), school_prefs[s].end(), 0);
    }

    vector<Node> node_list; // 各学校からノードを取り出し、番号を割り当てる
    node_list.reserve(N);
    for (int sch = 0; sch < M; ++sch) {
        const auto& v = schools[sch].get_nodes();
        node_list.insert(node_list.end(), v.begin(), v.end());
    }

    auto node_prefs = build_student_prefer(school_prefs, students, female_quota);

    vector<int> match_to_node(N, -1);
    stable_matching(node_prefs, node_list, match_to_node);

    /**
     * ➀
     */
    /* vector<int> histogram(M, 0);
    for (int s = 0; s < N; ++s) {
        int node = match_to_node[s];
        int school = node / CAPACITY;

        int rank = 0;
        while (rank < M && school_prefs[s][rank] != school) {
            rank++;
        }
        if (rank < M) {
            histogram[rank]++;
        }
    }

    cout << "=== ヒストグラム（何位の学校に入れたか） ===\n";
    for (int r = 0; r < M; r++) {
        cout << r + 1 << " 位: " << histogram[r] << " 人\n";
    } */

    /**
     * ➁
     */
    vector<int> male_histogram(M, 0);
    vector<int> female_histogram(M, 0);

    for (int s = 0; s < N; s++) {
        int node = match_to_node[s];
        int school = node / CAPACITY;

        int rank = 0;
        while (rank < M && school_prefs[s][rank] != school) {
            rank++;
        }

        if (students[s].get_gender() == 1) { // 女子
            female_histogram[rank]++;
        } else { // 男子
            male_histogram[rank]++;
        }
    }

    cout << "\n=== 男女別ヒストグラム ===\n";
    cout << "[女子]\n";
    for (int r = 0; r < M; r++) {
        cout << "  " << r + 1 << " 位: " << female_histogram[r] << " 人\n";
    }
    cout << "[男子]\n";
    for (int r = 0; r < M; r++) {
        cout << "  " << r + 1 << " 位: " << male_histogram[r] << " 人\n";
    }
}