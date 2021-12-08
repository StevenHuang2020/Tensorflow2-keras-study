//
// Created by StevenHuang on 8/21/21
// 
// Descrption: c++ inference after model traning by using cppflow and tensorflow c api
// cppflow: https://serizba.github.io/cppflow/index.html
// tensorflow c API: https://www.tensorflow.org/install/lang_c
// Visual studio 2019 project.
//

#include <iostream>

#include "cppflow/ops.h"
#include "cppflow/model.h"


void test1()
{
    auto input = cppflow::fill({ 3, 5 }, 1.0f);

    std::cout << "hello" << std::endl;
    std::cout << input << std::endl;

    cppflow::model model("../../model");
    auto output = model(input);

    std::cout << output << std::endl;

    auto values = output.get_data<float>();

    std::cout << "prediction:" << std::endl;
    for (auto v : values) {
        std::cout << v << std::endl;
    }
}

//check input & output layers names for function model()
//saved_model_cli show --dir ./model_cnn --all
void testCNN() //mnistCNN
{
    auto input = cppflow::fill({ 1, 28, 28, 1 }, 10.0f);

    std::cout << "hello" << std::endl;
    std::cout << input << std::endl;

    cppflow::model model("../../model_cnn");
    auto output = model({ {"serving_default_conv2d_input:0", input} }, { "StatefulPartitionedCall:0" });

    std::cout << "output size:" << output.size() << std::endl;;
    std::cout << "output[0]:" << output[0] << std::endl;

    auto values = output[0].get_data<float>();
    std::cout << "prediction:" << std::endl;
    for (auto v : values) {
        std::cout << v << std::endl;
    }

    auto max = std::max_element(values.begin(), values.end());
    int argmaxVal = std::distance(values.begin(), max);
    std::cout << "argmaxVal = " << argmaxVal << std::endl;
}

int main() 
{
    //test1();
    testCNN();
    return 0;
}
