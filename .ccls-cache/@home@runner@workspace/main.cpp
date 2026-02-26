/*
Author: Penguin Interactive
Date: 19.02.2026 ---> Now
Version 0.2.1

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
#include <numbers>
#include <format>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "PerlinNoise.hpp"



using namespace std;

typedef vector<int> Pixel;
typedef vector<Pixel> Row;
typedef vector<Row> Image3D;

float RANDOM(float min, float max, int precision) {
	static random_device rd;
	static mt19937 gen(rd());

	uniform_real_distribution<float> dist(min, max);
	float rNum = dist(gen);

	float p = pow(10.0f, precision);
	return round(rNum * p) / p;
}
void FRANDOM(vector<float>& vec, float min, float max) {
	static random_device rd;
	static mt19937 gen(rd());
	uniform_real_distribution<float> dist(min, max);

	for (auto& val : vec) {
		val = dist(gen);
	}
}

float clear() {
	#ifdef _WIN32
		float random_number = std::system("cls"); // For Windows
	#else
		float random_number = std::system("clear"); // Assume POSIX (Linux/macOS)
	#endif
	return random_number;
}

Image3D pngToVector(const char* filename) {
	int width, height, channels;

	// Load image data (forces 3 channels: RGB)
	unsigned char* data = stbi_load(filename, &width, &height, &channels, 3);

	if (data == nullptr) {
		cerr << "Error: Could not load image " << filename << endl;
		return {};
	}

	// Initialize the 3D vector: [width][height][3]
	Image3D imgVector(width, Row(height, Pixel(3)));

	for (int x = 0; x < width; ++x) {
		for (int y = 0; y < height; ++y) {
			// stbi_load provides data in a flat row-major array [y][x][channels]
			// We map that to your requested [x][y][channel] format
			int pixel_index = (y * width + x) * 3;

			imgVector[x][y][0] = static_cast<int>(data[pixel_index]);     // R
			imgVector[x][y][1] = static_cast<int>(data[pixel_index + 1]); // G
			imgVector[x][y][2] = static_cast<int>(data[pixel_index + 2]); // B
		}
	}

	// Free the memory allocated by stb_image
	stbi_image_free(data);

	return imgVector;
}


class Node {
public:
	float Value, Bias, Delta;
	int Index, Layer;
	string Type;
	Node(int index, int layer, string type) {
		Value = 0;
		Bias = 0;
		Delta = 0;
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
	float LR = 0.2;
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
	float Sigmoid(float x) {
		return 1/(1+pow(numbers::e, -x));
	}
	vector<float> Feed(vector<float> input) {
		for (int i = 0; i < input.size(); i++) {
			Nodes[0][i]->Value = input[i];
		}
		for (int l = 1; l < Nodes.size(); l++) {
			for (int i = 0; i < Nodes[l].size(); i++) {
				Nodes[l][i]->FeedThrough(Nodes, Weights);
				Nodes[l][i]->Value = Sigmoid(Nodes[l][i]->Value);
			}
		}
		vector<float> toReturn = {};
		for (Node* node : Nodes[Nodes.size()-1]) {
			toReturn.push_back(node->Value);
		}
		return toReturn;
	}
	void Backpropagate(vector<float> target) {
		for (int i = 0; i < Nodes[Nodes.size()-1].size(); i++){
			float actual = Nodes[Nodes.size()-1][i]->Value;
			float d = (target[i] - actual) * actual*(1-actual);
			Nodes[Nodes.size()-1][i]->Delta = d;
		}
		for (int l = Nodes.size() - 2; l > 0; l--) {
			for (int i = 0; i < Nodes[l].size(); i++) {
				float actual = Nodes[l][i]->Value;
				float sumNext = 0;
				for (int j = 0; j < Nodes[l+1].size(); j++) {
					sumNext += Weights[l][i][j] * Nodes[l+1][j]->Delta;
				}
				Nodes[l][i]->Delta = sumNext * (actual * (1 - actual));
			}
		}
		for (int l = 0; l < Weights.size(); l++) {
			for (int i = 0; i < Nodes[l].size(); i++) {
				for (int j = 0; j < Nodes[l+1].size(); j++) {
					Weights[l][i][j] += LR * Nodes[l][i]->Value * Nodes[l+1][j]->Delta;
				}
			}
		}
		for (int l = 1; l < Nodes.size(); l++) {
			for (int i = 0; i < Nodes[l].size(); i++) {
				Nodes[l][i]->Bias += LR * Nodes[l][i]->Delta;
			}
		}
	}
}

string colour(string_view text, int r, int g, int b) {
	return format("\x1b[38;2;{};{};{}m{}\x1b[0m", r, g, b, text);
}

namespace GUESS_FUNC {
	const vector<vector<float>> LRs = {{1, 0.8}, {0.4, 0.5}, {0.08, 0.3}, {0.05, 0.2}, {0.03, 0.1}, {0.02, 0.05}, {0.01, 0.025}, {0.005, 0.0175}, {0.001, 0.01}};
	void INIT() {
		NeuralNetwork::INIT(2, 4, 16, 1);
	}
	float TRUE_INTENSITY(float x, float y) {
		//return (sin(x) + 1) * (tanh(y) + 1) / 4;
		//return (sin(x) + sin(y)) / 4 + 0.5;
		//return (sin(x+y)+1)/2;
		//return (sin(0.5*x)*cos(x+0.5*y)+1)/2;
		//return (sin(x)+sin(y/1.5)+sin((x+y))+3)/6;
		//return (sin(pow(pow(x,2)+pow(y,2),0.5))+1)/2;
		//return ((sin(0.5*x)*cos(x+0.5*y)+1)/2)*((sin(x)+1)/2)*((sin(y)+1)/2);
		//return (sin(x+y)+1)/2 * ((sin(x) + sin(y)) / 4 + 0.5);
		//return (sin(x/2)*sin(x/2)+sin(y/2)*sin(y/2))/2;
		//return (sin(x+y)/2+0.5)/2+(sin(x)*sin(y)/4+0.5)/2;
		return (1/(1+abs(tan(x/5))))*(1/(1+abs(tan(y/5))));
	}
	float RUN(float min, float max, float step, int i) {
		float sum_err = 0;
		int count = pow(floor((max-min)/step),2);
		vector<vector<float>> map = {};
		vector<vector<float>> target = {};
		vector<vector<float>> error = {};
		float result;
		for (float x = min; x <= max; x += step) {
			map.push_back({});
			target.push_back({});
			error.push_back({});
			for (float y = min; y <= max; y += step) {
				result = NeuralNetwork::Feed({x, y})[0];
				map[map.size()-1].push_back(result);
				target[target.size()-1].push_back(TRUE_INTENSITY(x, y));
				sum_err += pow(TRUE_INTENSITY(x, y) - result, 2);
				error[error.size()-1].push_back(abs(result-TRUE_INTENSITY(x, y)));
			}
		}
		string tocout;
		for (int y = 0; y < map.size(); y++) {
			tocout += "\n";
			for (int x = 0; x < map.size(); x++) {
				tocout += colour("\u2588\u2588", map[x][y]*255, map[x][y]*255, map[x][y]*255);
			}
			tocout += "    ";
			for (int x = 0; x < map.size(); x++) {
				tocout += colour("\u2588\u2588", target[x][y]*255, target[x][y]*255, target[x][y]*255);
				/*if (error[x][y] > 0.5) {
					tocout += colour("\u2588\u2588", (2*error[x][y]-0.5)*255, 0, 0);
				} else {
					tocout += colour("\u2588\u2588", 0, 255-2*error[x][y]*255, 0);
				}*/
			}
		}
		sum_err /= count;
		clear();
		cout << tocout << "\nIteration: " << i << "\nLearning Rate: " << NeuralNetwork::LR << "\n" << sum_err << flush;
		return sum_err;
	}
	void GO(int iterations, float min, float max, float step) {
		INIT();
		if (iterations == -1) {
			iterations = 10000;
		}
		for (int i = 0; i < iterations; i++) {
			//clear();
			for (int t = 0; t < 10000; t++) {
				float rx = RANDOM(min, max, 4);
				float ry = RANDOM(min, max, 4);

				vector<float> actual = NeuralNetwork::Feed({rx, ry});
				float target = TRUE_INTENSITY(rx, ry);
				NeuralNetwork::Backpropagate({target});
			}
			float result = RUN(min, max, step, i);
			if (result < 0.0001) { return; }
			if (result > LRs[0][0]) {
				NeuralNetwork::LR = 1;
			} else {
				for (vector<float> LR : LRs) {
					if (result < LR[0]) {
						NeuralNetwork::LR = LR[1];
					} else {
						break;
					}
				}
			}
		}
	}
}

namespace RECREATE_IMG {
	const vector<vector<float>> LRs = {{1, 0.8}, {0.4, 0.5}, {0.08, 0.3}, {0.05, 0.2}, {0.03, 0.1}, {0.02, 0.05}, {0.01, 0.025}, {0.005, 0.0175}, {0.001, 0.01}};
	void INIT() {
		NeuralNetwork::INIT(2, 4, 4, 3);
	}
	vector<float> TRUE_PIXEL(int x, int y, Image3D& img) {
		return {img[x][y][0]/255.0f, img[x][y][1]/255.0f, img[x][y][2]/255.0f};
	}
	float RUN(Image3D& image_data, int i) {
		float sum_err = 0;
		int countX = image_data.size();
		int countY = image_data[0].size();
		Image3D map = {};
		Image3D target = {};
		//Image3D error = {};
		vector<float> result;
		for (float x = 0; x < countX; x += 1) {
			//cout << "LOOPX" << endl;
			map.push_back({});
			target.push_back({});
			//error.push_back({});
			for (float y = 0; y < countY; y += 1) {
				//cout << "LOOPY" << endl;
				result = NeuralNetwork::Feed({x, y});
				float rR = result[0];
				float rG = result[1];
				float rB = result[2];
				float tR = TRUE_PIXEL(x, y, image_data)[0];
				float tG = TRUE_PIXEL(x, y, image_data)[1];
				float tB = TRUE_PIXEL(x, y, image_data)[2];
				map[map.size()-1].push_back({(int) (rR*255), (int) (rG*255), (int) (rB*255)});
				target[target.size()-1].push_back({(int) (tR*255), (int) (tG*255), (int) (tB*255)});
				sum_err += pow(tR - rR, 2)/3;
				sum_err += pow(tG - rG, 2)/3;
				sum_err += pow(tB - rB, 2)/3;
				//error[error.size()-1].push_back(abs(result-TRUE_PIXEL(x, y)));
			}
		}
		string tocout;
		for (int y = 0; y < map[0].size(); y++) {
			tocout += "\n";
			for (int x = 0; x < map.size(); x++) {
				tocout += colour("\u2588\u2588", map[x][y][0], map[x][y][1], map[x][y][2]);
			}
			tocout += "    ";
			for (int x = 0; x < map.size(); x++) {
				tocout += colour("\u2588\u2588", target[x][y][0], target[x][y][1], target[x][y][2]);
				/*if (error[x][y] > 0.5) {
					tocout += colour("\u2588\u2588", (2*error[x][y]-0.5)*255, 0, 0);
				} else {
					tocout += colour("\u2588\u2588", 0, 255-2*error[x][y]*255, 0);
				}*/
			}
		}
		sum_err /= countX*countY;
		clear();
		cout << tocout << "\nIteration: " << i << "\nLearning Rate: " << NeuralNetwork::LR << "\n" << sum_err << flush;
		return sum_err;
	}
	void GO(int iterations, string image_source) {
		Image3D img = pngToVector(image_source.c_str());
		INIT();
		if (iterations == -1) {
			iterations = 10000;
		}
		for (int i = 0; i < iterations; i++) {
			//clear();
			for (int t = 0; t < 10000; t++) {
				float rx = round(RANDOM(0, img.size()-1, 0));
				float ry = round(RANDOM(0, img[0].size()-1, 0));

				vector<float> actual = NeuralNetwork::Feed({rx, ry});
				vector<float> target = TRUE_PIXEL(rx, ry, img);
				NeuralNetwork::Backpropagate(target);
			}
			float result = RUN(img, i);
			if (result < 0.0001) { return; }
			if (result > LRs[0][0]) {
				NeuralNetwork::LR = 1;
			} else {
				for (vector<float> LR : LRs) {
					if (result < LR[0]) {
						NeuralNetwork::LR = LR[1];
					} else {
						break;
					}
				}
			}
		}
	}
}

typedef vector<vector<float>> PerlinNoise;

/*namespace TOPOLOGY {
	const vector<vector<float>> LRs = {{1, 0.8}, {0.4, 0.5}, {0.08, 0.3}, {0.05, 0.2}, {0.03, 0.1}, {0.02, 0.05}, {0.01, 0.025}};
	void INIT() {
		NeuralNetwork::INIT(2, 4, 16, 1);
	}
	float TRUE_INTENSITY(float x, float y, PerlinNoise perlin_noise) {
		
	}
	float RUN(float min, float max, float step, int i) {
		float sum_err = 0;
		int count = pow(floor((max-min)/step),2);
		vector<vector<float>> map = {};
		vector<vector<float>> target = {};
		vector<vector<float>> error = {};
		float result;
		for (float x = min; x <= max; x += step) {
			map.push_back({});
			target.push_back({});
			error.push_back({});
			for (float y = min; y <= max; y += step) {
				result = NeuralNetwork::Feed({x, y})[0];
				map[map.size()-1].push_back(result);
				target[target.size()-1].push_back(TRUE_INTENSITY(x, y));
				sum_err += pow(TRUE_INTENSITY(x, y) - result, 2);
				error[error.size()-1].push_back(abs(result-TRUE_INTENSITY(x, y)));
			}
		}
		string tocout;
		for (int y = 0; y < map.size(); y++) {
			tocout += "\n";
			for (int x = 0; x < map.size(); x++) {
				tocout += colour("\u2588\u2588", map[x][y]*255, map[x][y]*255, map[x][y]*255);
			}
			tocout += "    ";
			for (int x = 0; x < map.size(); x++) {
				tocout += colour("\u2588\u2588", target[x][y]*255, target[x][y]*255, target[x][y]*255);
				if (error[x][y] > 0.5) {
					tocout += colour("\u2588\u2588", (2*error[x][y]-0.5)*255, 0, 0);
				} else {
					tocout += colour("\u2588\u2588", 0, 255-2*error[x][y]*255, 0);
				}
			}
		}
		sum_err /= count;
		clear();
		cout << tocout << "\nIteration: " << i << "\nLearning Rate: " << NeuralNetwork::LR << "\n" << sum_err << flush;
		return sum_err;
	}
	void GO(int iterations, float min, float max, float step) {
		INIT();
		if (iterations == -1) {
			iterations = 10000;
		}
		for (int i = 0; i < iterations; i++) {
			//clear();
			for (int t = 0; t < 10000; t++) {
				float rx = RANDOM(min, max, 4);
				float ry = RANDOM(min, max, 4);

				vector<float> actual = NeuralNetwork::Feed({rx, ry});
				float target = TRUE_INTENSITY(rx, ry);
				NeuralNetwork::Backpropagate({target});
			}
			float result = RUN(min, max, step, i);
			if (result < 0.001) { return; }
			if (result > LRs[0][0]) {
				NeuralNetwork::LR = 1;
			} else {
				for (vector<float> LR : LRs) {
					if (result < LR[0]) {
						NeuralNetwork::LR = LR[1];
					} else {
						break;
					}
				}
			}
		}
	}
}*/

namespace HideAndSeek {
	typedef vector<float> LineFunc;
	float intersect(LineFunc& a, LineFunc& b) {
		return (b[1]-a[1])/(a[0]-b[0]);
	}
	const vector<vector<float>> LRs = {{1, 0.8}, {0.4, 0.5}, {0.08, 0.3}, {0.05, 0.2}, {0.03, 0.1}, {0.02, 0.05}, {0.01, 0.025}, {0.005, 0.0175}, {0.001, 0.01}};
	void INIT() {
		NeuralNetwork::INIT(6, 4, 16, 1);
	}
	float RUN() {
		vector<float> AI = {RANDOM(33, 40, 1), RANDOM(33, 40, 1)};
		vector<float> SEEK = {RANDOM(0, 7, 1), RANDOM(0, 7, 1)};

		vector<int> wall_bottom = {(int) RANDOM(8, 16, 0), (int) RANDOM(24, 32, 0)};
		vector<int> wall_top = {(int) RANDOM(24, 32, 0), (int) RANDOM(8, 16, 0)};
		float a = ((float) (wall_top[1]-wall_bottom[1]))/((float) (wall_top[0]-wall_bottom[0]));
		float b = wall_bottom[1]-a*wall_bottom[0];
		LineFunc line_func = {a, b};

		cout << "LB" << endl;
		while (true) {
			cout << "IL" << endl;
			vector<float> actual = NeuralNetwork::Feed({(float) wall_bottom[0], (float) wall_bottom[1], AI[0], AI[1], SEEK[0], SEEK[1]});
			cout << "BLF" << endl;
			LineFunc actual_func = {tan(actual[0]), AI[1]-tan(actual[0])*AI[0]};
			cout << "ALF" << endl;
			float tx = intersect(line_func, actual_func);
			vector<float> p2 = {cos(actual[0])+AI[0], sin(actual[0])*actual[1] + actual[1]};
			cout << "FIRST IF BEFORE" << flush;
			if (wall_bottom[0] <= tx && tx <= wall_top[0]) {
				if (AI[0] < p2[0]) {
					if (AI[0] <= tx && tx <= p2[0]) {
						break;
					}
				} else {
					if (p2[0] <= tx && tx <= AI[0]) {
						break;
					}
				}
			}
			AI[0] = p2[0];
			AI[1] = p2[1];

			float a = ((float) (AI[1]-SEEK[1]))/((float) (AI[0]-SEEK[0]));
			float b = SEEK[1]-a*SEEK[0];
			LineFunc hide_seek_func = {a, b};

			float x = intersect(line_func, hide_seek_func);
			if (wall_bottom[0] <= x && x <= wall_top[0]) {
				break;
			}

			if (40 < AI[0] || AI[0] < 0 || 40 < AI[1] || AI[1] < 0) {
				break;
			}

			clear();
			for (int x = 0; x <= 40; x++) {
				for (int y = 0; y <= 40; y++) {
					if (round(AI[0]) == x && round(AI[1]) == y) {
						cout << colour("\u2588\u2588", 0, 255, 255) << flush;
					} else if (round(SEEK[0]) == x && round(SEEK[1]) == y) {
						cout << colour("\u2588\u2588", 255, 0, 0) << flush;
					} else if (x >= wall_bottom[0] && x <= wall_top[0] && round(line_func[0]*x+line_func[1]) == y) {
						cout << colour("\u2588\u2588", 255, 255, 255) << flush;
					} else {
						cout << "  " << flush;
					}
				}
				cout << "\n";
			}
		}
		return 0;
	}
	void GO() {
		
	}
}

int main() {
	std::ios_base::sync_with_stdio(false);
	std::cin.tie(NULL);
	//GUESS_FUNC::GO(-1, -8, 8, 0.4);
	//RECREATE_IMG::GO(-1, "Colour Mix 2.png");
	//TOPOLOGY::GO(-1);
	HideAndSeek::RUN();
	return 0;
}
