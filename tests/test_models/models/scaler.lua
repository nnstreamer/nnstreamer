-- scaler.lua

inputTensorInfo = {
  dim = {3, 640, 480, 1},
  type = "uint8_t",
}

outputTensorInfo = {
  dim = {3, 320, 240, 1},
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

  for n=1,oN do
    for h=1,oH do
      for w=1,oW do
        for c=1,oC do
          outIndex = (n - 1) * oH * oW * oC
          outIndex = outIndex + (h - 1) * oW * oC
          outIndex = outIndex + (w - 1) * oC
          outIndex = outIndex + c

          inputIndex = (n - 1) * iH * iW * iC
          inputIndex = inputIndex + (math.floor((h - 1) * (iH / oH))) * iW * iC
          inputIndex = inputIndex + (math.floor((w - 1) * (iW / oW))) * iC
          inputIndex = inputIndex + c

          output[outIndex] = input[inputIndex]
        end
      end
    end
  end

end
