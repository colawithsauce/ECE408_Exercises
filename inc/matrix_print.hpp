#include <string>

template <typename T> std::string matrix2str(T *mat, int y, int x)
{
    std::string s;
    s += "[";
    for (int i = 0; i != y; i++)
    {
        s += "[";
        for (int j = 0; j != x; j++)
        {
            if (j != 0)
                s += ", ";
            s += std::to_string(mat[j + i * x]);
        }

        s += "]";
    }

    s += "]";

    return s;
}

#define PRINT_ARR(ARR)                                                                                                 \
    printf("Array %s :\n", #ARR);                                                                                      \
    for (int i = 0; i != SIZE; i++)                                                                                    \
    {                                                                                                                  \
        for (int j = 0; j != SIZE; j++)                                                                                \
        {                                                                                                              \
            if (j != 0)                                                                                                \
                printf(" ");                                                                                           \
            printf("%lf", ARR[i * SIZE + j]);                                                                          \
        }                                                                                                              \
        printf("\n");                                                                                                  \
    }                                                                                                                  \
    printf("\n");
