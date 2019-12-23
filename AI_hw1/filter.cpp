#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <assert.h>
#include <algorithm>

#define MAX_NUM 2501
#define TEST_SAMPLE_NUM 10
#define P_S 0.5
#define P_H 0.5
#define P_ASSUME 0.1
#define P 0.99

using namespace std;

void init();
void read_from_file(string, string);
void test();
bool cmp(pair<int,int>, pair<int,int>);
void judge(vector<pair<int,int>>);
void calculate();

int h_word_num[MAX_NUM];
int s_word_num[MAX_NUM];
double P_W_in_H[MAX_NUM];
double P_W_in_S[MAX_NUM];
double P_S_in_W[MAX_NUM];

int sample_num = 0;

int main()
{
    init();
    test();
    calculate();
}

void init()
{
    for(int i = 0; i < MAX_NUM; ++i){
        h_word_num[i] = s_word_num[i] = 0;
        P_W_in_H[i] = P_W_in_S[i] = 0;
    }

    read_from_file("./DataPrepared/train-features.txt","./DataPrepared/train-labels.txt");
    read_from_file("./DataPrepared/train-features-50.txt","./DataPrepared/train-labels-50.txt");
    read_from_file("./DataPrepared/train-features-100.txt","./DataPrepared/train-labels-100.txt");
    read_from_file("./DataPrepared/train-features-400.txt","./DataPrepared/train-labels-400.txt");

    for(int i = 0; i < MAX_NUM; ++i){
        P_W_in_H[i] = static_cast<double>(h_word_num[i]) / static_cast<double>(sample_num);
        P_W_in_S[i] = static_cast<double>(s_word_num[i]) / static_cast<double>(sample_num);
        
        if(fabs(P_W_in_H[i] )<1e-15){
            P_S_in_W[i] = 0.99;
        }
        else
            P_S_in_W[i] = (P_W_in_S[i]) / (P_W_in_S[i] + P_W_in_H[i]);
    }

}

void read_from_file(string featrue_filename, string label_fillename)
{
    ifstream label_in(label_fillename.c_str(),ios::in);
    ifstream featrue_in(featrue_filename.c_str(),ios::in);

    assert(label_in.is_open()&&featrue_in.is_open());

    int label, sample_id, word_id, num, temp = 0;

    while(featrue_in>>sample_id>>word_id>>num){
        if (temp != sample_id){
            temp = sample_id;
            label_in>>label;
        }
        if(label == 0){
            h_word_num[word_id] += num;
        } 
        else {
            s_word_num[word_id] += num;
        }
    }

    sample_num += sample_id;

    label_in.close();
    featrue_in.close();
}

void test()
{
    ifstream f("./DataPrepared/test-features.txt",ios::in);
    ofstream out("./DataPrepared/ans.txt",ios::out);
    assert(f.is_open() && out.is_open());

    vector<pair<int,int>> input;
    int  sample_id, word_id, num, temp = 0;
    
    while(f>>sample_id>>word_id>>num){
        if(temp != sample_id){
            if(temp != 0){
                judge(input);
            }
            input.clear();
            temp = sample_id;
        }
        input.push_back(make_pair(word_id,num));
    }

    judge(input);
    
    f.close();
    out.close();
}

bool cmp(pair<int,int> a,pair<int,int> b)
{
    return a.second>b.second;
}

void judge(vector<pair<int, int>> v)
{
    sort(v.begin(),v.end(), cmp);
    double x = 0, y = 0;

    int size = v.size() < TEST_SAMPLE_NUM ? v.size() : TEST_SAMPLE_NUM;

    for(int i = 0; i   < size; ++i ){
        if(fabs(P_S_in_W[v[i].first] < 1e-15)){
            P_S_in_W[v[i].first] = P_ASSUME;
        }

        if(fabs(x)<1e-15){
            x = P_S_in_W[v[i].first];
        }
        else{
            x *= P_S_in_W[v[i].first];
        }

        if(fabs(y)<1e-15){
            y = 1 - P_S_in_W[v[i].first];
        }
        else {
            y *= (1-P_S_in_W[v[i].first]);
        }
    }
    
    double P_CALCULATE = x / (x +y);

    ofstream out_file("./DataPrepared/ans.txt",ios::app);
    assert(out_file.is_open());

    if(P_CALCULATE > P ){
        out_file<<1<<"\n";
    }
    else {
        out_file<<0<<"\n";
    }

    out_file.close();
}

void calculate()
{
    int x, y, count = 0;
    ifstream ans("./DataPrepared/ans.txt");
    ifstream real("./DataPrepared/test-labels.txt");
    assert(ans.is_open() && real.is_open());

    for(int i = 0; i < 260; ++i){
        ans>>x;
        real>>y;
        if(x!=y){
            count++;
        }
    }
    cout<<"The number of different labels  is "<<count
            <<".\nThe accuracy rate of spam is "<<1 - static_cast<double>(count)/260<<".\n";
    ans.close();
    real.close();
}