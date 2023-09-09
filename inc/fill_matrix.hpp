#include <cassert>
#include <random>

template <class T> void fill_randomly(T *mat, int M, int N)
{
    for (int i = 0; i < M * N; ++i)
    {
        mat[i] = rand() % 100 / 10.0;
    }
}

template <class T> void fill_eye(T *mat, int M, int N)
{
    assert(M == N);
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            if (i == j)
                mat[i * N + j] = 1;
            else
                mat[i * N + j] = 0;
        }
    }
}
