#include "libLKYDeepNN/LKYDeepNN.hpp"


int main()
{
    int numEachHiddenNodes = 2;
    int numHiddenLayers = 4;
    LKYDeepNN nn(2, vector<int>(numHiddenLayers, numEachHiddenNodes), 2);
    nn.SetActivation(new ReLU());

    vector<double> targetArray{2,-2};
    vector<double> outputArray;

    for(int i=0; i<500 ;i++)
    {
        cout << "訓練, ";
        nn.Training(0.01, targetArray);

        outputArray = nn.ForwardPropagation(targetArray);
        cout << "outputArray: ";
        for (double const output : outputArray)
        {//print
            printf("%lf, ",output);
        }
        cout << "順傳遞測試完成" <<endl;
    }
}