-- scaler.lua

inputTensorsInfo = {
  num = 1,
  dim = {{3, 640, 480, 1},},
  type = {"uint8", }
}

outputTensorsInfo = {
  num = 1,
  dim = {{3, 320, 240, 1},},
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
