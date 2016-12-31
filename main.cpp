#include "libLKYDeepNN/LKYDeepNN.hpp"
#include "libLKYDeepNN/DataSet.hpp"

void DrawTraining(LKYDeepNN _nn, int maxEpochs, int currentEpochs, const vector<vector<double>>& displayData)
{   //size_t numItems = 80;
    string strPngName = "png/訓練途中" + to_string(currentEpochs) + ".png";
    string strPutText = "Epoch:"+to_string(currentEpochs)+"/"+to_string(maxEpochs)+"  Err:" + to_string(_nn.GetTrainError().back());

    //cv::imwrite(strPngName.c_str(),DrawData("訓練途中", displayData, strPutText));
    Draw2DClassificationData("訓練途中", displayData, _nn, strPutText);
    //fgetc(stdin);
}

int main()
{
    vector<vector<double>> trainData = Make2DBinaryTrainingData();
    int numEachHiddenNodes = 2;
    int numHiddenLayers = 4;
    LKYDeepNN nn(2, vector<int>(numHiddenLayers, numEachHiddenNodes), 2);
    nn.SetActivation(new Tanh());
    //nn.eventInTraining = DrawTraining;//將包有視覺化的事件傳入

    cout << "訓練開始";
    int epochs = 500;
    nn.Training(0.001, epochs, trainData);

    //outputArray = nn.ForwardPropagation(targetArray);
    cout << "outputArray: ";
    // for (double const output : outputArray)
    // {//print
    //     printf("%lf, ",output);
    // }
    cout << "順傳遞測試完成" <<endl;
}