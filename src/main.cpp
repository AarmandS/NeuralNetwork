#include "network.h"

int main()
{
	std::vector<Vector<float, 784>> data = read_data(true);
	std::vector<uint8_t> labels = read_labels(true);
	std::vector<Vector<float, 784>> test_data = read_data(false);
	std::vector<uint8_t> test_labels = read_labels(false);
	Normalize(data);
	Normalize(test_data);
	Network<784, 30, 10> n("test_30");
	n.Evaluate(test_data, test_labels);
	n.SGD(data, labels, 100, 1.0, 10);
	n.Evaluate(test_data, test_labels);
	n.Save("test_30");
	std::cin.get();
	
}


