//
// Created by cartercpp on 10/31/26.
//

#pragma once

#include <initializer_list>
#include <string>
#include <vector>
#include <mdspan>
#include <cstddef>

class neural_network
{
public:

    // CONSTRUCTORS

    explicit neural_network(
        std::initializer_list<std::size_t> neuronsPerLayer,
        float learningRate,
        bool randomlyInitWeights = true
    )
        pre(neuronsPerLayer.size() >= 3)
        pre((learningRate > 0.0f) && (learningRate < 1.0f));

    // METHODS

    void load_weights(const std::string& filepath);
    void load_biases(const std::string& filepath);

    [[nodiscard]] std::vector<float> predict(const std::vector<float>& input) const
        pre(input.size() == m_neuronsPerLayer[0]);

    void fit(const std::vector<float>& input, const std::vector<float>& target)
        pre(input.size() == m_neuronsPerLayer[0])
        pre(target.size() == m_neuronsPerLayer.back());

private:

    using Vector = std::mdspan<const float, std::dextents<std::size_t, 1>>;
    using Matrix = std::mdspan<const float, std::dextents<std::size_t, 2>>;

    [[nodiscard]] static std::vector<float> MatrixMatrix(Matrix lArg, Matrix rArg)
        pre(lArg.extent(1) == rArg.extent(0));
    [[nodiscard]] static std::vector<float> MatrixVector(Matrix mat, Vector vec)
        pre(mat.extent(1) == vec.extent(0));
    [[nodiscard]] static std::vector<float> VectorMatrix(Vector vec, Matrix mat)
        pre(vec.extent(0) == mat.extent(0));
    [[nodiscard]] std::vector<float> OuterProduct(Vector lArg, Vector rArg);
    static void ApplySoftmax(std::vector<float>& logits);

    [[nodiscard]] std::vector<std::vector<float>> Forward(const std::vector<float>& input) const
        pre(input.size() == m_neuronsPerLayer[0]);

    std::vector<std::vector<float>> m_weights,
                                    m_biases;
    std::vector<std::size_t> m_neuronsPerLayer;
    float m_learningRate;
};