//
// Created by cartercpp on 10/31/26.
//

#include "neural_network.hpp"
#include <fstream>
#include <initializer_list>
#include <string>
#include <vector>
#include <mdspan>
#include <utility>
#include <random>
#include <ranges>
#include <algorithm>
#include <cmath>
#include <cstddef>

std::vector<float> neural_network::MatrixMatrix(Matrix lArg, Matrix rArg)
{
    const std::size_t lArgRows = lArg.extent(0),
                      rArgColumns = rArg.extent(1),
                      iterCount = lArg.extent(1);

    std::vector<float> output(lArgRows * rArgColumns, 0.0f);
    for (std::size_t row = 0; row < lArgRows; ++row)
        for (std::size_t column = 0; column < rArgColumns; ++column)
            for (std::size_t iter = 0; iter < iterCount; ++iter)
                std::mdspan(output.data(), lArgRows, rArgColumns)[row, column]
                    += lArg[row, iter] * rArg[iter, column];

    return output;
}

std::vector<float> neural_network::MatrixVector(Matrix mat, Vector vec)
{
    const std::size_t rows = mat.extent(0);

    std::vector<float> output(rows, 0.0f);
    for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t i = 0; i < vec.extent(0); ++i)
            output[row] += mat[row, i] * vec[i];

    return output;
}

std::vector<float> neural_network::VectorMatrix(Vector vec, Matrix mat)
{
    const std::size_t columns = mat.extent(1);

    std::vector<float> output(columns, 0.0f);
    for (std::size_t column = 0; column < columns; ++column)
        for (std::size_t i = 0; i < vec.extent(0); ++i)
            output[column] += vec[i] * mat[i, column];

    return output;
}

std::vector<float> neural_network::OuterProduct(Vector lArg, Vector rArg)
{
    const std::size_t lArgSize = lArg.extent(0),
                      rArgSize = rArg.extent(0);
    std::vector<float> output(lArgSize * rArgSize, 0.0f);

    for (std::size_t i = 0; i < lArgSize; ++i)
        for (std::size_t i2 = 0; i2 < rArgSize; ++i2)
            std::mdspan(output.data(), lArgSize, rArgSize)[i, i2] = lArg[i] * rArg[i2];

    return output;
}

std::vector<std::vector<float>> neural_network::Forward(const std::vector<float>& input) const
{
    // `activations` stores the activations of each layer up to the output layer
    // `activations[n]` = activations at layer n
    std::vector<std::vector<float>> activations;
    activations.reserve(m_neuronsPerLayer.size());
    activations.push_back(input);

    // iterate through all weight matrices but the last one:
    for (std::size_t i = 0; i + 1 < m_weights.size(); ++i)
    {
        // weights * input:
        std::vector<float> layerActivations = MatrixVector(
            Matrix{m_weights[i].data(), m_neuronsPerLayer[i + 1], m_neuronsPerLayer[i]},
            Vector{activations[i].data(), m_neuronsPerLayer[i]}
        );

        // add bias to activations, and apply relu:
        for (std::size_t neuron = 0; neuron < m_neuronsPerLayer[i + 1]; ++neuron)
        {
            layerActivations[neuron] += m_biases[i][neuron];
            layerActivations[neuron] = (layerActivations[neuron] > 0.0f) ? layerActivations[neuron] : 0.0f;
        }

        // append to `activations`:
        activations.emplace_back(std::move(layerActivations));
    }

    return activations;
}

void neural_network::ApplySoftmax(std::vector<float>& probabilities)
{
    const float maxLogit = std::ranges::max(probabilities);
    float softmaxSum = 0.0f;

    std::ranges::for_each(probabilities, [&](float& valueRef) {
        valueRef = std::exp(valueRef - maxLogit);
        softmaxSum += valueRef;
    });

    std::ranges::for_each(probabilities, [softmaxSum](float& valueRef){valueRef /= softmaxSum;});
}

std::vector<float> neural_network::predict(const std::vector<float>& input) const
{
    const std::vector<std::vector<float>> activations{Forward(input)};

    // multiply last weight matrix and last hidden activation, and add last bias vector to get unnormalized probabilities:
    std::vector<float> probabilities = MatrixVector(
        Matrix{m_weights.back().data(), m_neuronsPerLayer.back(), m_neuronsPerLayer[m_neuronsPerLayer.size() - 2]},
        Vector{activations.back().data(), m_neuronsPerLayer[m_neuronsPerLayer.size() - 2]}
    );

    for (std::size_t index = 0; index < m_neuronsPerLayer.back(); ++index)
        probabilities[index] += m_biases.back()[index];

    // convert logits to probabilities:
    ApplySoftmax(probabilities);

    // return output:
    return probabilities;
}

void neural_network::fit(const std::vector<float>& input, const std::vector<float>& target)
{
    const std::vector<std::vector<float>> activations{Forward(input)};

    // compute prediction:
    std::vector<float> prediction = MatrixVector(
        Matrix{m_weights.back().data(), m_neuronsPerLayer.back(), m_neuronsPerLayer[m_neuronsPerLayer.size() - 2]},
        Vector{activations.back().data(), m_neuronsPerLayer[m_neuronsPerLayer.size() - 2]}
    );

    for (std::size_t index = 0; index < m_neuronsPerLayer.back(); ++index)
        prediction[index] += m_biases.back()[index];

    ApplySoftmax(prediction);

    // compute loss gradient:
    std::vector<float> incomingGradient(m_neuronsPerLayer.back(), 0.0f);
    for (std::size_t neuron = 0; neuron < m_neuronsPerLayer.back(); ++neuron)
        incomingGradient[neuron] = prediction[neuron] - target[neuron];

    // propagate loss gradient backwards:
    for (std::size_t weightIndex = m_weights.size(); weightIndex-- > 0;)
    {
        const std::size_t layerSize = m_neuronsPerLayer[weightIndex + 1],
                          prevLayerSize = m_neuronsPerLayer[weightIndex];
        const std::vector<float>& prevLayerActivations{activations[weightIndex]};
        std::vector<float>& layerWeightsRef{m_weights[weightIndex]};
        std::vector<float>& layerBiasesRef{m_biases[weightIndex]};

        // incoming gradient is wrt activations, make it wrt pre-activations:
        if (weightIndex < m_weights.size() - 1)
        {
            const std::vector<float>& layerActivations{activations[weightIndex + 1]};

            for (std::size_t neuron = 0; neuron < layerSize; ++neuron)
                incomingGradient[neuron] *= (layerActivations[neuron] > 0.0f) ? 1.0f : 0.0f;
        }

        /*
            compute weight gradient by taking the outer product of this incoming gradient and the activations
            in the previous layer
        */
        const std::vector<float> weightGradients = OuterProduct(
            Vector{incomingGradient.data(), incomingGradient.size()},
            Vector{prevLayerActivations.data(), prevLayerActivations.size()}
        );

        // biasGradient = incomingGradient, so apply SGD like this:
        for (std::size_t index = 0; index < layerSize; ++index)
            layerBiasesRef[index] -= m_learningRate * incomingGradient[index];

        // fit incoming gradient to previous layer:
        if (weightIndex > 0)
            incomingGradient = VectorMatrix(
                Vector{incomingGradient.data(), layerSize},
                Matrix{layerWeightsRef.data(), layerSize, prevLayerSize}
            );

        // use SGD to update weights:
        for (std::size_t index = 0; index < weightGradients.size(); ++index)
            layerWeightsRef[index] -= m_learningRate * weightGradients[index];
    }
}

neural_network::neural_network(
    std::initializer_list<std::size_t> neuronsPerLayer,
    float learningRate,
    bool randomlyInitWeights
)
    : m_neuronsPerLayer{neuronsPerLayer},
    m_learningRate{learningRate}
{
    m_weights.reserve(m_neuronsPerLayer.size() - 1);
    m_biases.reserve(m_neuronsPerLayer.size() - 1);

    for (std::size_t i = 0; i < m_neuronsPerLayer.size() - 1; ++i)
    {
        const std::size_t layerSize = m_neuronsPerLayer[i + 1],
                          prevLayerSize = m_neuronsPerLayer[i];

        m_weights.emplace_back(layerSize * prevLayerSize, 0.0f);
        m_biases.emplace_back(layerSize, 0.0f);

        if (randomlyInitWeights)
        {
            std::mt19937_64 rng{std::random_device{}()};
            std::normal_distribution<float> dist(0, std::sqrtf(2.0f / static_cast<float>(prevLayerSize)));
            std::ranges::for_each(m_weights.back(), [&](float& valueRef){valueRef = dist(rng);});
        }
    }
}

void neural_network::load_weights(const std::string& filepath)
{
    std::ifstream file{filepath};

    for (std::size_t weightIndex = 0; weightIndex < m_weights.size(); ++weightIndex)
        for (std::size_t i = 0; i < m_weights[weightIndex].size(); ++i)
            file >> m_weights[weightIndex][i];
}

void neural_network::load_biases(const std::string& filepath)
{
    std::ifstream file{filepath};

    for (std::size_t biasIndex = 0; biasIndex < m_biases.size(); ++biasIndex)
        for (std::size_t i = 0; i < m_biases[biasIndex].size(); ++i)
            file >> m_biases[biasIndex][i];
}
