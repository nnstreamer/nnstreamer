-- passthrough.lua

inputTensorInfo = {
  dim = {3, 640, 480, 1},
  type = "uint8_t",
}

outputTensorInfo = {
  dim = {3, 640, 480, 1},
  type = "uint8_t",
}

iC = inputTensorInfo["dim"][1]
iW = inputTensorInfo["dim"][2]
iH = inputTensorInfo["dim"][3]
iN = inputTensorInfo["dim"][4]

oC = outputTensorInfo["dim"][1]
oW = outputTensorInfo["dim"][2]
oH = outputTensorInfo["dim"][3]
oN = outputTensorInfo["dim"][4]

function nnstreamer_invoke()
  input = input_tensor()
  output = output_tensor()

  for i=1,oC*oW*oH*oN do
    output[i] = input[i]
  end
end
