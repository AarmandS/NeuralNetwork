#pragma once
#include <array>
#include <random>

template <typename T, size_t R, size_t C>
class Matrix {
private:
	std::array<T, R * C> m_array;
public:
	const int Rows() const { return R; }
	const int Cols() const { return C; }
	T& Get(int row_index, int col_index) { return m_array[int(C) * row_index + col_index]; }
	const T& Get(int row_index, int col_index) const { return m_array[int(C) * row_index + col_index]; }
	void operator += (const Matrix& other)
	{
		for (size_t i = 0; i < R; i++)
			for (size_t j = 0; j < C; j++)
				this->Get(i, j) += other.Get(i, j);
	}
	void operator -= (const Matrix& other)
	{
		for (size_t i = 0; i < R; i++)
			for (size_t j = 0; j < C; j++)
				this->Get(i, j) -= other.Get(i, j);
	}
	void operator *= (const Matrix& other)
	{
		for (size_t i = 0; i < R; i++)
			for (size_t j = 0; j < C; j++)
				this->Get(i, j) *= other.Get(i, j);
	}
	void operator /= (const Matrix& other)
	{
		for (size_t i = 0; i < R; i++)
			for (size_t j = 0; j < C; j++)
				this->Get(i, j) /= other.Get(i, j);
	}
	void operator += (const T& value)
	{
		for (size_t i = 0; i < R; i++)
			for (size_t j = 0; j < C; j++)
				this->Get(i, j) += value;
	}
	void operator -= (const T& value)
	{
		for (size_t i = 0; i < R; i++)
			for (size_t j = 0; j < C; j++)
				this->Get(i, j) -= value;
	}
	void operator *= (const T& value)
	{
		for (size_t i = 0; i < R; i++)
			for (size_t j = 0; j < C; j++)
				this->Get(i, j) *= value;
	}
	void operator /= (const T& value)
	{
		for (size_t i = 0; i < R; i++)
			for (size_t j = 0; j < C; j++)
				this->Get(i, j) /= value;
	}
	void Random()
	{
		std::random_device rd;
		std::mt19937 e(rd());
		std::uniform_real_distribution<> dist(-1, 1);
		for (size_t i = 0; i < R; i++)
			for (size_t j = 0; j < C; j++)
				this->Get(i, j) = dist(e);
	}
	Matrix<T, C, R> Transposed() const
	{
		Matrix<T, C, R> transposed;
		for (size_t i = 0; i < R; i++)
			for (size_t j = 0; j < C; j++)
				transposed.Get(j, i) = this->Get(i, j);
		return transposed;
	}
};