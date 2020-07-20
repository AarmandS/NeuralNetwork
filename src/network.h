#pragma once
#include <iostream>
#include <vector>
#include <math.h>
#include "my_vector.h"
#include "my_matrix.h"
#include "read_data.h"

// Function to get weighted input
template <typename T, size_t R, size_t C>
Vector<T, C> ApplyWeights(const Matrix<T, R, C>& weights, const Vector<T, R> inputs)
{
	Vector<T, C> res;
	for (int col = 0; col < C; col++)
	{
		res[col] = 0;
		for (int row = 0; row < R; row++)
			res[col] += weights.Get(row, col) * inputs[row];
	}
	return res;
}


// Vectorized sigmoid function
template <typename T, size_t S>
Vector<T, S> Sigmoid(const Vector<T, S>& v)
{
	Vector<T, S> res;
	for (int i = 0; i < S; i++)
		res[i] = 1.0f / (1.0f + std::exp(-v[i]));
	return res;
}

// Vectorized derivative of sigmoid function
template <typename T, size_t S>
Vector<T, S> SigmoidPrime(const Vector<T, S>& v)
{
	Vector<T, S> res = Sigmoid(v);
	Vector<T, S> negative = res;
	negative *= -1.0f;
	negative += 1.0f;
	res *= negative;
	return res;

}

// Struct to store gradients calculated by backprop
template <size_t I, size_t H, size_t O>
struct Gradients
{
	Matrix<float, I, H> g_hidden_weigths;
	Matrix<float, H, O> g_output_weights;
	Vector<float, H> g_hidden_biases;
	Vector<float, O> g_output_biases;
};

// Calculate the gradient for the weights
template<typename T, size_t R, size_t C>
Matrix<T, R, C> GetWeightGradient(const Vector<T, R>& prev_activation, const Vector<T, C>& error)
{
	Matrix<T, R, C> res;
	for (int i = 0; i < R; i++)
	{
		for (int j = 0; j < C; j++)
			res.Get(i, j) = prev_activation[i] * error[j];
	}
	return res;
}

// I = input- , H = hidden- , O = output neurons
template <size_t I, size_t H, size_t O>
class Network
{
private:
	Matrix<float, I, H> m_hidden_weights;
	Matrix<float, H, O> m_output_weights;
	Vector<float, H> m_hidden_biases;
	Vector<float, O> m_output_biases;
	

public:
	// Initialize network with random weights and biases
	Network()
	{
		m_hidden_weights.Random();
		m_output_weights.Random();
		m_hidden_biases.Random();
		m_output_biases.Random();
	}

	// Load existing network
	Network(std::string name)
	{
		std::ifstream f;
		f.open("networks/" + name, std::ios::binary);
		if (f.is_open())
		{
			// Read in the sizes of the network
			uint32_t i = I; uint32_t h = H; uint32_t o = O;
			f.read((char*)&i, sizeof(uint32_t));
			f.read((char*)&h, sizeof(uint32_t));
			f.read((char*)&o, sizeof(uint32_t));

			// Set the weights to the ones stored in the file
			for (int row = 0; row < m_hidden_weights.Rows(); row++)
				for (int col = 0; col < m_hidden_weights.Cols(); col++)
					f.read(reinterpret_cast<char*>(&m_hidden_weights.Get(row, col)), sizeof(float));
			for (int row = 0; row < m_output_weights.Rows(); row++)
				for (int col = 0; col < m_output_weights.Cols(); col++)
					f.read(reinterpret_cast<char*>(&m_output_weights.Get(row, col)), sizeof(float));
			// Set the biases
			for (int b = 0; b < H; b++)
				f.read(reinterpret_cast<char*>(&m_hidden_biases[b]), sizeof(float));
			for (int b = 0; b < O; b++)
				f.read(reinterpret_cast<char*>(&m_output_biases[b]), sizeof(float));
		}
		// If the file couldn't be loaded, initialize random weights and biases
		else 
		{
			m_hidden_weights.Random();
			m_output_weights.Random();
			m_hidden_biases.Random();
			m_output_biases.Random();
		}
	}

	// Get output of network
	Vector<float, O> Feedforward(const Vector<float, I>& inputs)
	{
		Vector<float, H> z1 = ApplyWeights(m_hidden_weights, inputs);
		z1 += m_hidden_biases;
		Vector<float, O> z2 = ApplyWeights(m_output_weights, Sigmoid(z1));
		z2 += m_output_biases;
		return Sigmoid(z2);
	}


	Gradients<I, H, O> Backprop(const Vector<float, I>& inputs, const uint8_t& label, const std::unordered_map<uint8_t, Vector<float, O>>& desired_map)
	{
		// Feedforward phase, storing weighted + biased inputs, and activations ( a = activation, w + b input = z)
		Vector<float, I> a1 = inputs;
		Vector<float, H> z1 = ApplyWeights(m_hidden_weights, inputs);
		z1 += m_hidden_biases;
		Vector<float, H> a2 = Sigmoid(z1);
		Vector<float, O> z2 = ApplyWeights(m_output_weights, Sigmoid(z1));
		z2 += m_output_biases;
		Vector<float, O> a3 = Sigmoid(z2);

		// Calculate the error of last layer
		Vector<float, O> output_error = a3;
		output_error -= desired_map.find(label)->second;
		output_error *= SigmoidPrime(z2);

		// Backpropagate error to hidden layer
		Vector<float, H> hidden_error = ApplyWeights(m_output_weights.Transposed(), output_error);
		hidden_error *= SigmoidPrime(z1);

		// Calculate the gradients using the errors

		Gradients<I, H, O> grad;
		grad.g_hidden_biases = hidden_error;
		grad.g_output_biases = output_error;
		grad.g_hidden_weigths = GetWeightGradient(a1, hidden_error);
		grad.g_output_weights = GetWeightGradient(a2, output_error);
		return grad;
	}

	// Stochastic gradient descent
	void SGD(std::vector<Vector<float, I>>& images, std::vector<uint8_t>& labels, int batch_size, float eta, int epochs)
	{
		auto desired = GetDesiredMap();
		for (int e = 1; e <= epochs; e++)
		{
			// Shuffle the dataset and divide into batches
			Shuffle(images, labels);
			int batches = labels.size() / batch_size;
			for (int b = 0; b < batches; b++)
			{
				// Average the gradients over the whole batch
				Gradients<I, H, O> grad = Backprop(images[b * batch_size], labels[b * batch_size], desired);
				for (int i = 1; i < batch_size; i++)
				{
					Gradients<I, H, O> new_grad = Backprop(images[b * batch_size + i], labels[b * batch_size + i], desired);
					grad.g_hidden_biases += new_grad.g_hidden_biases;
					grad.g_output_biases += new_grad.g_output_biases;
					grad.g_hidden_weigths += new_grad.g_hidden_weigths;
					grad.g_output_weights += new_grad.g_output_weights;
				}
				float ratio = eta / batch_size;
				grad.g_hidden_biases *= ratio;
				grad.g_output_biases *= ratio;
				grad.g_hidden_weigths *= ratio;
				grad.g_output_weights *= ratio;
				// Update the weights and biases of the network
				m_hidden_biases -= grad.g_hidden_biases;
				m_output_biases -= grad.g_output_biases;
				m_hidden_weights -= grad.g_hidden_weigths;
				m_output_weights -= grad.g_output_weights;
			}
			std::cout << "Epoch " << e << " completed." << std::endl;
		}
	}

	// Evalute network; Counts the times the network correctly classifies an image
	void Evaluate(const std::vector<Vector<float, I>>& images, const std::vector<uint8_t>& labels)
	{
		uint32_t correct = 0;
		for (int i = 0; i < labels.size(); i++)
		{
			float max = 0;
			uint8_t max_index = 0;
			Vector<float, O> results = Feedforward(images[i]);
			for (int j = 0; j < O; j++)
			{
				if (results[j] > max)
				{
					max = results[j];
					max_index = j;
				}
			}
			if (max_index == labels[i])
				correct++;
		}
		std::cout << correct << " were classified correctly out of " << labels.size() << '.' << std::endl;
	}

	void Save(std::string name)
	{
		std::ofstream f;
		f.open("networks/" + name, std::ios::binary);
		if (f.is_open())
		{
			uint32_t i = I; uint32_t h = H; uint32_t o = O;
			// Write the sizes of the network to the file
			f.write(reinterpret_cast<char*>(&i), sizeof(size_t));
			f.write(reinterpret_cast<char*>(&h), sizeof(size_t));
			f.write(reinterpret_cast<char*>(&o), sizeof(size_t));
			// Write the weights to the file
			for (int row = 0; row < m_hidden_weights.Rows(); row++)
				for (int col = 0; col < m_hidden_weights.Cols(); col++)
					f.write(reinterpret_cast<char*>(&m_hidden_weights.Get(row, col)), sizeof(float));
			for (int row = 0; row < m_output_weights.Rows(); row++)
				for (int col = 0; col < m_output_weights.Cols(); col++)
					f.write(reinterpret_cast<char*>(&m_output_weights.Get(row, col)), sizeof(float));
			// Write the biases to the file
			for (int b = 0; b < H; b++)
				f.write(reinterpret_cast<char*>(&m_hidden_biases[b]), sizeof(float));
			for (int b = 0; b < O; b++)
				f.write(reinterpret_cast<char*>(&m_output_biases[b]), sizeof(float));
			std::cout << "Network saved successfully." << std::endl;
		}
		else
			std::cout << "Failed to save network." << std::endl;
	}
};