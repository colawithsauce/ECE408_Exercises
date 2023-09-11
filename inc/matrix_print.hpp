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

#define PRINT_ARR(ARR, HEIGHT, WIDTH)                                                                                  \
    printf("Array %s :\n", #ARR);                                                                                      \
    for (int i = 0; i != HEIGHT; i++)                                                                                  \
    {                                                                                                                  \
        for (int j = 0; j != WIDTH; j++)                                                                               \
        {                                                                                                              \
            if (j != 0)                                                                                                \
                printf(" ");                                                                                           \
            printf("%lf", ARR[i * WIDTH + j]);                                                                         \
        }                                                                                                              \
        printf("\n");                                                                                                  \
    }                                                                                                                  \
    printf("\n");

#define PRINT_MAT(ARR, HEIGHT, WIDTH)                                                                                  \
    printf("Array %s :\n", #ARR);                                                                                      \
    for (int i = 0; i != HEIGHT; i++)                                                                                  \
    {                                                                                                                  \
        for (int j = 0; j != WIDTH; j++)                                                                               \
        {                                                                                                              \
            if (j != 0)                                                                                                \
                printf(" ");                                                                                           \
            printf("%lf", ARR[i][j]);                                                                                  \
        }                                                                                                              \
        printf("\n");                                                                                                  \
    }                                                                                                                  \
    printf("\n");
