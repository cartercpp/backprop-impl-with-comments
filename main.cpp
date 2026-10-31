#include <iostream>
#include <fstream>
#include <string>
#include <set>
#include <map>
#include <ranges>
#include <algorithm>
#include <thread>
#include <stop_token>
#include "neural_network.hpp"

int main()
{
    std::ifstream file{"/home/cartercpp/Documents/C++/BackpropFromScratch/training_text.txt"};
    const std::string fileContent(std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{});
    const std::string uniqueChars = [&fileContent]() {
        std::set<char> uniqueChars(fileContent.begin(), fileContent.end());
        return std::string(uniqueChars.begin(), uniqueChars.end());
    }();
    std::map<char, std::size_t> charToIndex;

    {
        std::size_t index = 0;
        for (const char ch : uniqueChars)
        {
            charToIndex[ch] = index;
            ++index;
        }
    }

    constexpr std::size_t contextLength = 10;
    neural_network nextCharNN(
        {contextLength * uniqueChars.size(), 64, 64, uniqueChars.size()},
        0.001f
    );

    std::cout << "\033[2J"; // clear screen
    std::cout << "\033[?25l"; // turn off mouse flicker

    {
        std::jthread trainingThread{[&](std::stop_token st) {
            std::size_t epoch = 0;
            while (!st.stop_requested())
            {
                // do training:
                std::vector<float> inputEncoding(uniqueChars.size() * contextLength, 0.0f),
                                   outputEncoding(uniqueChars.size(), 0.0f);
                for (std::size_t start = 0; start + contextLength < fileContent.size(); ++start)
                {
                    for (std::size_t i = 0; i < contextLength; ++i)
                    {
                        const char ch = fileContent[start + i];
                        inputEncoding[i * uniqueChars.size() + charToIndex[ch]] = 1.0f;
                    }

                    const char nextChar = fileContent[start + contextLength];
                    outputEncoding[charToIndex[nextChar]] = 1.0f;

                    nextCharNN.fit(inputEncoding, outputEncoding);

                    std::ranges::for_each(inputEncoding, [](float& valueRef){valueRef = 0.0f;});
                    std::ranges::for_each(outputEncoding, [](float& valueRef){valueRef = 0.0f;});
                }

                // display progress:
                if (epoch % 100 == 0)
                {
                    std::cout << "\033[2J"; // clear screen
                    std::cout << "\033[0H"; // go to top left
                    std::cout << "Learning language in action:\n";

                    std::string prediction{fileContent.substr(0, contextLength)};
                    std::cout << prediction << std::flush;

                    while (prediction.size() < fileContent.size())
                    {
                        for (std::size_t i = 0; i < contextLength; ++i)
                        {
                            const char ch = prediction[prediction.size() - (contextLength - i)];
                            inputEncoding[i * uniqueChars.size() + charToIndex[ch]] = 1.0f;
                        }

                        outputEncoding = nextCharNN.predict(inputEncoding);
                        const std::size_t nextCharIndex
                            = std::ranges::max_element(outputEncoding) - outputEncoding.begin();

                        const char nextChar = uniqueChars[nextCharIndex];
                        prediction.push_back(nextChar);
                        std::cout << nextChar << std::flush;

                        std::ranges::for_each(inputEncoding, [](float& valueRef){valueRef = 0.0f;});
                    }
                }

                ++epoch;
            }
        }};
        std::cin.get();
    }
}