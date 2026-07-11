#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

using namespace std;

// Convert string to lower case
void toLowerCase(string &s) {
    for (char &c : s) {
        c = tolower(c);
    }
}

// Naive Search
vector<int> naiveSearch(const string& text, const string& pattern) {
    vector<int> result;
    int n = text.length();
    int m = pattern.length();
    if (m == 0 || n < m) return result;

    for (int i = 0; i <= n - m; i++) {
        int j;
        for (j = 0; j < m; j++) {
            if (text[i + j] != pattern[j])
                break;
        }
        if (j == m) {
            result.push_back(i);
        }
    }
    return result;
}

// KMP Search
void computeLPSArray(const string& pattern, int m, vector<int>& lps) {
    int len = 0;
    lps[0] = 0;
    int i = 1;
    while (i < m) {
        if (pattern[i] == pattern[len]) {
            len++;
            lps[i] = len;
            i++;
        } else {
            if (len != 0) {
                len = lps[len - 1];
            } else {
                lps[i] = 0;
                i++;
            }
        }
    }
}

vector<int> kmpSearch(const string& text, const string& pattern) {
    vector<int> result;
    int n = text.length();
    int m = pattern.length();
    if (m == 0 || n < m) return result;

    vector<int> lps(m);
    computeLPSArray(pattern, m, lps);

    int i = 0; 
    int j = 0; 
    while (i < n) {
        if (pattern[j] == text[i]) {
            j++;
            i++;
        }
        if (j == m) {
            result.push_back(i - j);
            j = lps[j - 1];
        } else if (i < n && pattern[j] != text[i]) {
            if (j != 0) {
                j = lps[j - 1];
            } else {
                i = i + 1;
            }
        }
    }
    return result;
}

// Rabin-Karp Search
vector<int> rkSearch(const string& text, const string& pattern) {
    vector<int> result;
    int n = text.length();
    int m = pattern.length();
    if (m == 0 || n < m) return result;

    int d = 256;
    int q = 101; 
    int p = 0; // hash value for pattern
    int t = 0; // hash value for txt
    int h = 1;

    for (int i = 0; i < m - 1; i++)
        h = (h * d) % q;

    for (int i = 0; i < m; i++) {
        p = (d * p + pattern[i]) % q;
        t = (d * t + text[i]) % q;
    }

    for (int i = 0; i <= n - m; i++) {
        if (p == t) {
            bool match = true;
            for (int j = 0; j < m; j++) {
                if (text[i + j] != pattern[j]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                result.push_back(i);
            }
        }
        if (i < n - m) {
            t = (d * (t - text[i] * h) + text[i + m]) % q;
            if (t < 0)
                t = (t + q);
        }
    }
    return result;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " <algorithm: naive|kmp|rk> <keyword> <file_path>" << endl;
        return 1;
    }

    string algo = argv[1];
    string keyword = argv[2];
    string filePath = argv[3];

    ifstream file(filePath, ios::binary);
    if (!file.is_open()) {
        cerr << "Error opening file." << endl;
        return 1;
    }

    string text((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
    file.close();

    // Convert both to lowercase for case-insensitive search
    toLowerCase(text);
    toLowerCase(keyword);

    vector<int> matches;
    if (algo == "naive") {
        matches = naiveSearch(text, keyword);
    } else if (algo == "kmp") {
        matches = kmpSearch(text, keyword);
    } else if (algo == "rk") {
        matches = rkSearch(text, keyword);
    } else {
        cerr << "Unknown algorithm." << endl;
        return 1;
    }

    for (size_t i = 0; i < matches.size(); i++) {
        cout << matches[i];
        if (i < matches.size() - 1) cout << " ";
    }
    cout << endl;

    return 0;
}
