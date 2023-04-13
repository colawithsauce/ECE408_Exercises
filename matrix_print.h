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
