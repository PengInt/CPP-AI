/*
Author: Penguin Interactive
Date: 19.02.2026 ---> Now
Version 0.0.1

Description:
	A 'simple' neural network that tries to stay away from one target and keep close to another.
	Weights: staying close = 1.1
	         staying far   = 0.9
	If the far target is too close* the AI loses
	If the close target is close enough* the AI wins

	*(within 1 distance unit)
*/

#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <exception>
#include <cmath>
#include <cstdlib>


using namespace std;

float RANDOM(float min, float max, int precision) {
	static random_device rd;
	static mt19937 gen(rd());

	uniform_real_distribution<float> dist(min, max);
	float rNum = dist(gen);

	float p = pow(10.0f, precision);
	return round(rNum * p) / p;
}



class Node {
public:
	float Value, Bias;
	int Index, Layer;
	string Type;
	Node(int index, int layer, string type) {
		Value = 0;
		Bias = 0;
		Index = index;
		Layer = layer;
		Type = type;
	}
	void FeedThrough(vector<vector<Node*>>& Nodes, vector<vector<vector<float>>>& Weights) {
		if (Type != "INPUT") {
			Value = Bias;
			for (int i = 0; i < Nodes[Layer-1].size(); i++) {
				Value += Nodes[Layer-1][i]->Value * Weights[Layer-1][i][Index];
			}
		} else {
			throw new runtime_error("INVALID NODE FOR 'FeedThrough' - TRIED TO FEED THROUGH AN INPUT NODE");
		}
	}
};

namespace NeuralNetwork {
	vector<vector<Node*>> Nodes = {};
	vector<vector<vector<float>>> Weights = {};
	void INIT(int inputs, int layers, int layer_width, int outputs) {
		// Initialise the Nodes
		Nodes.push_back({});
		for (int i = 0; i < inputs; i++) {
			Nodes[0].push_back(new Node(i, 0, "INPUT"));
		}
		for (int l = 1; l < layers+1; l++) {
			Nodes.push_back({});
			for (int i = 0; i < layer_width; i++) {
				Nodes[l].push_back(new Node(i, l, "NORMAL"));
			}
		}
		Nodes.push_back({});
		for (int i = 0; i < outputs; i++) {
			Nodes[Nodes.size()-1].push_back(new Node(i, layers+1, "OUTPUT"));
		}

		// Initialise the Weights
		Weights.push_back({});
		for (int i = 0; i < inputs; i++) {
			Weights[0].push_back({});
			for (int j = 0; j < layer_width; j++) {
				Weights[0][i].push_back(RANDOM(-0.3, 0.3, 3));
			}
		}
		for (int l = 1; l < layers+1; l++) {
			Weights.push_back({});
			for (int i = 0; i < layer_width; i++) {
				Weights[l].push_back({});
				for (int j = 0; j < layer_width; j++) {
					Weights[l][i].push_back(RANDOM(-0.3, 0.3, 3));
				}
			}
		}
	}
	vector<float> Feed(vector<float> input) {
		for (int i = 0; i < input.size(); i++) {
			Nodes[0][i]->Value = input[i];
		}
		for (int l = 1; l < Nodes.size(); l++) {
			for (int i = 0; i < Nodes[l].size(); i++) {
				Nodes[l][i]->FeedThrough(Nodes, Weights);
			}
		}
		vector<float> toReturn = {};
		for (Node* node : Nodes[Nodes.size()-1]) {
			toReturn.push_back(node->Value);
		}
		return toReturn;
	}
}


int main() {
	NeuralNetwork::INIT(8, 8, 16, 2);
	vector<float> result = NeuralNetwork::Feed({1, 2, 3, 4, 5, 6, 7, 8});
	for (float r : result) {
		cout << r << "\n" << flush;
	}
	float BOT_X = 0;
	float BOT_Y = 0;
	float GOOD_X = 2;
	float GOOD_Y = 3;
	float BAD_X = -2;
	float BAD_Y = 4;
	system("cls");
	int botx_r = round(BOT_X);
	int boty_r = round(BOT_Y);
	int goodx_r = round(GOOD_X);
	int goody_r = round(GOOD_Y);
	int badx_r = round(BAD_X);
	int bady_r = round(BAD_Y);
	string toPrint = "+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+";
	for (int x = -10; x <= 10; x++) {
		toPrint += "|\n+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+\n";
		for (int y = 10; y >= -10; y--) {
			toPrint += "|";
			if (botx_r == x && boty_r == y) {
				toPrint += "00";
			} else if (badx_r == x && bady_r == y) {
				toPrint += "--";
			} else if (goodx_r == x && goody_r == y) {
				toPrint += "++";
			} else {
				toPrint += "  ";
			}
		}
	}
	toPrint += "|\n+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+";
	cout << toPrint << flush;
	return 0;
}
