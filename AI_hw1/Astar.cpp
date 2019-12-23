#include <iostream>
#include <algorithm>
#include <vector>
#include <stack>
#include <iomanip>
#include <queue>
#include <string>

//u can change the start city and end city here (read array "cities" for matching relation)
//plz update the array "gred" synchronously when change the end city.
#define START_CITY 2
#define END_CITY 12
#define CITY_NUM 20

using namespace std;

int start = START_CITY, finish = END_CITY;		
int star = 1;	//judge whether iteration should be continue		
int sum = 0, num = 0;	//record the cost

const string cities[CITY_NUM] =
{
	"Oradea","Zerind","Arad","Timisoara","Lugoj","Mehadia","Drobeta","Sibiu","Rimnicu_vilcea","Craiova",
	"Fagaras","Pitesti","Bucharest","Giurgiu","Neamt","Iasi","Vaslui","Urziceni","Hirsova","Eforie"
};

const int map[CITY_NUM][CITY_NUM] =
{
	{0,71,-1,-1,-1,-1,-1,151,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},	//Oradea
	{71,0,75,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},	//Zerind
	{-1,75,0,118,-1,-1,-1,140,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},	//Arad
	{-1,-1,118,0,111,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},	//Timisoara
	{-1,-1,-1,111,0,70,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},	//Lugoj
	{-1,-1,-1,-1,70,0,75,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},	//Mehadia
	{-1,-1,-1,-1,-1,75,0,-1,-1,120,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},	//Drobeta
	{151,-1,140,-1,-1,-1,-1,0,80,-1,99,-1,-1,-1,-1,-1,-1,-1,-1,-1},	//Sibiu
	{-1,-1,-1,-1,-1,-1,-1,80,0,146,-1,97,-1,-1,-1,-1,-1,-1,-1,-1},	//Rimnicu_vilcea
	{-1,-1,-1,-1,-1,-1,120,-1,146,0,-1,138,-1,-1,-1,-1,-1,-1,-1,-1},//Craiova
	{-1,-1,-1,-1,-1,-1,-1,99,-1,-1,0,-1,211,-1,-1,-1,-1,-1,-1,-1},	//Fagaras
	{-1,-1,-1,-1,-1,-1,-1,-1,97,138,-1,0,101,-1,-1,-1,-1,-1,-1,-1},	//Pitesti
	{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,211,101,0,90,-1,-1,-1,85,-1,-1},	//Bucharest
	{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,90,0,-1,-1,-1,-1,-1,-1,},	//Giurgiu
	{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,87,-1,-1,-1,-1,},	//Neamt
	{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,87,0,92,-1,-1,-1},	//Iasi
	{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,92,0,142,-1,-1},	//Vaslui
	{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,85,-1,-1,-1,142,0,98,-1},	//Urziceni
	{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,98,0,86},	//Hirsova
	{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,86,0}	//Eforie
};

//stright line distance to Bucharest, change this array when the finished city is changed
const int gred[CITY_NUM] =
{
	380,374,366,329,244,241,242,253,193,160,176,100,0,77,234,226,199,80,151,161
};

void A_star_search(int begin, stack<int> path, int *sign, vector<int> load_cities, vector<int> cost)
{
	int p = -1, parent = -1;

	//find the index of begin city in cost
	for (int i = 0; i < cost.size(); i++){
		if (begin == load_cities[i]){
			parent = i;
			break;
		}
	}

	//load new city
	for (int i = 0; i < CITY_NUM; i++){
		if (map[begin][i] > 0 && star){
			num++;
			p = 1;
			load_cities.push_back(i);
			cost.push_back(cost[parent] + map[begin][i]);
		}
	}
	int size = load_cities.size();

	//sort the city with cost(f+g) 
	for (int i = 0; i < size; i++){
		for (int j = i; j < size; j++){
			if (cost[j] + gred[load_cities[j]] < cost[i] + gred[load_cities[i]]) {		
				int temp = cost[j];
				cost[j] = cost[i];
				cost[i] = temp;

				temp = load_cities[j];
				load_cities[j] = load_cities[i];
				load_cities[i] = temp;
			}
		}
	}

	//delete the cities except the city with minial cost 
	for (int i = 0; i < load_cities.size(); i++){		 
		if (sign[load_cities[i]] == 1) {
			load_cities.erase(load_cities.begin() + i);
			cost.erase(cost.begin() + i);
			sign[i] == 0;
		}
	}

	if (p != -1 && star){
		size = load_cities.size();
		for (int i = 0; i < load_cities.size() && star; i++){
			sign[load_cities[i]] = 1;
			if (map[begin][load_cities[i]] != -1){
				path.push(begin);
				
				//find the end city, end the iteration
				if (load_cities[i] == finish) {	 
					path.pop();
					star = 0;
					cout << "Path: ";
					vector<int> result;
					while (!path.empty()){
						result.push_back(path.top());
						path.pop();
					}
					int size = result.size();
					for (int i = size - 1; i >= 0; i--){
						if (i != size - 1)
							sum += map[result[i + 1]][result[i]];
						else
							sum += map[result[i]][2];
						cout << cities[result[i]] << " -> ";
					}
					sum += map[result[0]][finish];
					cout << cities[finish];
					cout << endl;
					cout << "Cost:" << sum << endl;
					cout << "Visited " << num << " cities in total." << endl;
					return;
				}

				//continue
				p = load_cities[i];		
				A_star_search(p, path, sign, load_cities, cost);
				sign[load_cities[i]] = 0;
				path.pop();
			}
		}
	}
}


int main()
{
	int sign[CITY_NUM];
	stack<int> path;
	vector<int> load_cities;
	vector<int> cost;

	for (int i = 0; i < CITY_NUM; ++i) {
		if (i == START_CITY)
			sign[i] = 1;
		else
			sign[i] = 0;
	}
	cost.push_back(0);
	load_cities.push_back(start);

	A_star_search(start, path, sign, load_cities, cost);
	return 0;
}
