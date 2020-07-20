#pragma once
#include "my_vector.h"
#include <unordered_map>
#include <fstream>
#include <string>
#include <time.h>

std::vector<Vector<float, 784>> read_data(bool train)
{
	const char* source;
	int size;
	if (train) {
		source = "data/train/train-images.idx3-ubyte";
		size = 60000 * 28 * 28 + 16;
	}
	else {
		source = "data/test/t10k-images.idx3-ubyte";
		size = 10000 * 28 * 28 + 16;
	}
	std::ifstream f;
	f.open(source, std::ios::binary);
	if (f.is_open())
	{
		char* train = new char[size];
		f.read(train, size);
		f.close();
		std::vector<Vector<float, 784>> imgs;
		Vector<float, 784> v;
		for (int i = 0; i < size - 15; i++)
		{

			if (i % 784 == 0 && i != 0) {
				imgs.push_back(v);
			}
			v[i % 784] = uint8_t(train[i + 16]);

		}
		delete[] train;
		return imgs;
	}
}

std::vector<uint8_t> read_labels(bool train) {
	const char* source;
	int size;
	if (train) {
		source = "data/train/train-labels.idx1-ubyte";
		size = 60008;
	}
	else {
		source = "data/test/t10k-labels.idx1-ubyte";
		size = 10008;
	}
	std::ifstream f;
	f.open(source, std::ios::binary);
	if (f.is_open())
	{
		char* train = new char[size];
		f.read(train, size);
		f.close();
		std::vector<uint8_t> labels;
		for (int i = 0; i < size; i++) {
			if (i > 7)
				labels.push_back(train[i]);
		}
		delete[] train;
		return labels;
	}
}

// Used this function to check whether the images were loaded correctly
/* 
void ToImage(Vector<float, 784>& v, const std::string& name)
{
	std::string fname = "imgs/";
	fname += name;
	fname += ".pgm";
	std::ofstream f(fname);
	std::cout << fname << std::endl;
	if (f.is_open()) {
		f << "P2\n";
		f << "28 28\n";
		f << "255\n";
		for (int i = 0; i < 784; i++)
			f << unsigned int(v[i]) << ' ';
		f.close();
	}
}
*/

void Normalize(std::vector<Vector<float, 784>>& images)
{
	for (int i = 0; i < images.size(); i++)
		images[i] /= 255.0f;
}

template <size_t S>
void Shuffle(std::vector<Vector<float, S>>& images, std::vector<uint8_t>& labels)
{
	std::vector<int> indexes;
	for (int i = 0; i < labels.size(); i++)
		indexes.push_back(i);
	srand(time(NULL));
	std::random_shuffle(indexes.begin(), indexes.end());
	std::vector<Vector<float, 784>> s_images;
	std::vector<uint8_t> s_labels;
	for (int i = 0; i < indexes.size(); i++) {
		s_images.push_back(images[indexes[i]]);
		s_labels.push_back(labels[indexes[i]]);
	}
	images = s_images;
	labels = s_labels;
}

// Returns a vector that is the target output for given label
Vector<float, 10> Desired(const uint8_t& label)
{
	Vector<float, 10> res;
	for (int i = 0; i < 10; i++)
	{
		if (i == label)
			res[i] = 1.0f;
		else
			res[i] = 0.0f;
	}
	return res;
}

// Returns a map that maps each digit to the corresponding desired output
std::unordered_map<uint8_t, Vector<float, 10>> GetDesiredMap()
{
	std::unordered_map<uint8_t, Vector<float, 10>> desired_map;
	for (int i = 0; i < 10; i++)
		desired_map[i] = Desired(i);
	return desired_map;
}

