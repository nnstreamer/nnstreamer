-- passthrough.lua

inputTensorsInfo = {
  num = 1,
  dim = {{3, 640, 480, 1}, },
  type = {"uint8", }
}

outputTensorsInfo = {
  num = 1,
  dim = {{3, 640, 480, 1}, },
  type = {"uint8", }
}

iC = inputTensorsInfo["dim"][1][1]
iW = inputTensorsInfo["dim"][1][2]
iH = inputTensorsInfo["dim"][1][3]
iN = inputTensorsInfo["dim"][1][4]

oC = outputTensorsInfo["dim"][1][1]
oW = outputTensorsInfo["dim"][1][2]
oH = outputTensorsInfo["dim"][1][3]
oN = outputTensorsInfo["dim"][1][4]

function nnstreamer_invoke()
  input = input_tensor(1)
  output = output_tensor(1)

  for i=1,oC*oW*oH*oN do
    output[i] = input[i]
  end
end
