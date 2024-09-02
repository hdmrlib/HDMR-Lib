#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// TensorND yapısı, n boyutlu tensorleri temsil eder
typedef struct {
    double *data;
    int *dims;
    int num_dims;
} TensorND;

// TensorND oluşturma
TensorND* create_tensor(int num_dims, int *dims) {
    TensorND *tensor = (TensorND*)malloc(sizeof(TensorND));
    tensor->num_dims = num_dims;
    tensor->dims = (int*)malloc(num_dims * sizeof(int));

    int total_size = 1;
    for (int i = 0; i < num_dims; i++) {
        tensor->dims[i] = dims[i];
        total_size *= dims[i];
    }
    tensor->data = (double*)malloc(total_size * sizeof(double));
    return tensor;
}

// TensorND belleği serbest bırakma
void free_tensor(TensorND *tensor) {
    free(tensor->data);
    free(tensor->dims);
    free(tensor);
}

// TensorND değerini alma
double get_tensor_value(TensorND *tensor, int *indices) {
    int index = 0;
    int stride = 1;
    for (int i = tensor->num_dims - 1; i >= 0; i--) {
        index += indices[i] * stride;
        stride *= tensor->dims[i];
    }
    return tensor->data[index];
}

// TensorND değerini ayarlama
void set_tensor_value(TensorND *tensor, int *indices, double value) {
    int index = 0;
    int stride = 1;
    for (int i = tensor->num_dims - 1; i >= 0; i--) {
        index += indices[i] * stride;
        stride *= tensor->dims[i];
    }
    tensor->data[index] = value;
}

// TensorND üzerinde eksene göre tensor dot işlemi
TensorND* tensor_dot(TensorND *A, TensorND *B, int axis) {
    TensorND *result = create_tensor(A->num_dims, A->dims);

    int *indices = (int*)calloc(A->num_dims, sizeof(int));
    int total_elements = 1;
    for (int i = 0; i < A->num_dims; i++) {
        total_elements *= A->dims[i];
    }

    for (int n = 0; n < total_elements; n++) {
        int temp_n = n;
        for (int i = A->num_dims - 1; i >= 0; i--) {
            indices[i] = temp_n % A->dims[i];
            temp_n /= A->dims[i];
        }

        double sum = get_tensor_value(A, indices) + get_tensor_value(B, indices);
        set_tensor_value(result, indices, sum);
    }

    free(indices);
    return result;
}

// g0 hesaplama fonksiyonu
double calculate_g0(TensorND *G, TensorND **support_vectors, double *weights, int dimensions) {
    TensorND *g0 = G;
    for (int i = 0; i < dimensions; i++) {
        TensorND *temp = tensor_dot(g0, support_vectors[i], 0);
        free_tensor(g0);
        g0 = temp;
    }
    double result = get_tensor_value(g0, (int[]){0});
    free_tensor(g0);
    return result;
}

// Yaklaşım hesaplama fonksiyonu
TensorND* calculate_approximation(TensorND *G, TensorND **support_vectors, double *weights, int dimensions, int order) {
    TensorND *overall_sum = tensor_dot(G, support_vectors[0], 0);
    
    for (int i = 1; i < order; i++) {
        for (int j = 0; j < dimensions; j++) {
            TensorND *temp = tensor_dot(overall_sum, support_vectors[j], 0);
            free_tensor(overall_sum);
            overall_sum = temp;
        }
    }

    return overall_sum;
}

int main() {
    int dims[] = {3, 4, 5};
    int num_dims = sizeof(dims) / sizeof(dims[0]);

    TensorND *G = create_tensor(num_dims, dims);

    // Örnek tensor verileri
    for (int i = 0; i < dims[0]; i++) {
        for (int j = 0; j < dims[1]; j++) {
            for (int k = 0; k < dims[2]; k++) {
                int indices[] = {i, j, k};
                set_tensor_value(G, indices, rand() / (double)RAND_MAX);
            }
        }
    }

    TensorND *support_vectors[3];
    for (int i = 0; i < 3; i++) {
        support_vectors[i] = create_tensor(num_dims, dims);
        // Destek vektörlerini doldur
    }

    double weights[3] = {1.0 / dims[0], 1.0 / dims[1], 1.0 / dims[2]};

    double g0 = calculate_g0(G, support_vectors, weights, num_dims);
    printf("g0: %f\n", g0);

    TensorND *approximation = calculate_approximation(G, support_vectors, weights, num_dims, 3);

    printf("Approximation Tensor:\n");
    for (int i = 0; i < dims[0]; i++) {
        for (int j = 0; j < dims[1]; j++) {
            for (int k = 0; k < dims[2]; k++) {
                int indices[] = {i, j, k};
                printf("%f ", get_tensor_value(approximation, indices));
            }
            printf("\n");
        }
        printf("\n");
    }

    free_tensor(G);
    for (int i = 0; i < 3; i++) {
        free_tensor(support_vectors[i]);
    }
    free_tensor(approximation);

    return 0;
}
