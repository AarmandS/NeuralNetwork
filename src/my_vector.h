#pragma once
#include <array>
#include <random>

template <typename T, size_t S>
class Vector
{
private:
	std::array<T, S> m_array;
public:
	const int Size() const { return S; }
	T& operator [] (int index) { return m_array[index]; }
	const T& operator [] (int index) const { return m_array[index]; }
	void operator += (const Vector& other)
	{
		for (size_t i = 0; i < S; i++)
			(*this)[i] += other[i];
	}
	void operator -= (const Vector& other)
	{
		for (size_t i = 0; i < S; i++)
			(*this)[i] -= other[i];
	}
	void operator *= (const Vector& other)
	{
		for (size_t i = 0; i < S; i++)
			(*this)[i] *= other[i];
	}
	void operator /= (const Vector& other)
	{
		for (size_t i = 0; i < S; i++)
			(*this)[i] /= other[i];
	}
	void operator += (const T& value)
	{
		for (size_t i = 0; i < S; i++)
			(*this)[i] += value;
	}
	void operator -= (const T& value)
	{
		for (size_t i = 0; i < S; i++)
			(*this)[i] -= value;
	}
	void operator *= (const T& value)
	{
		for (size_t i = 0; i < S; i++)
			(*this)[i] *= value;
	}
	void operator /= (const T& value)
	{
		for (size_t i = 0; i < S; i++)
			(*this)[i] /= value;
	}
	void Random()
	{
		std::random_device rd;
		std::mt19937 e(rd());
		std::uniform_real_distribution<> dist(-1, 1);
		for (size_t i = 0; i < S; i++)
			m_array[i] = dist(e);
	}
};