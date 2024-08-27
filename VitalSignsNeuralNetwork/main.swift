import Foundation

let trainFilePath = "/Users/gustavokumasawa/Documents/UTFPR/2024.1/Sistemas Inteligentes/Trabalho 2/treino_sinais_vitais_com_label.txt"
let testFilePath = "/Users/gustavokumasawa/Documents/UTFPR/2024.1/Sistemas Inteligentes/Trabalho 2/treino_sinais_vitais_sem_label.txt"

let neuron11 = Neuron(weightsCount: 4)
let neuron21 = Neuron(weightsCount: 4)
let neuron31 = Neuron(weightsCount: 4)

let neuronOutput = Neuron(weightsCount: 4)

let dataElements = Utilities.getParsedDataElements(from: trainFilePath)
let learningRate: Double = 0.05

for _ in 0..<50 {
    dataElements[0..<1350].forEach { dataElement in
        let inputValues: [Double] = [
            -1,
            dataElement.pressureQuality,
            dataElement.pulse,
            dataElement.breathing
        ]
        
        let y11 = neuron11.activation(inputs: inputValues)
        let y21 = neuron21.activation(inputs: inputValues)
        let y31 = neuron31.activation(inputs: inputValues)
        
        let output = neuronOutput.activation(
            inputs: [
                -1,
                 y11,
                 y21,
                 y31,
            ]
        )
        
        let errorOutput = output * (1 - output) * (dataElement.severity - output)
        neuronOutput.weights[0] -= learningRate * errorOutput * (-1)
        neuronOutput.weights[1] += learningRate * errorOutput * y11
        neuronOutput.weights[2] += learningRate * errorOutput * y21
        neuronOutput.weights[3] += learningRate * errorOutput * y31
        
        let error11 = y11 * (1 - y11) * (neuronOutput.weights[1] * errorOutput)
        neuron11.weights[0] += learningRate * error11 * inputValues[0]
        neuron11.weights[1] += learningRate * error11 * inputValues[1]
        neuron11.weights[2] += learningRate * error11 * inputValues[2]
        neuron11.weights[3] += learningRate * error11 * inputValues[3]
        
        let error21 = y21 * (1 - y21) * (neuronOutput.weights[2] * errorOutput)
        neuron21.weights[0] += learningRate * error21 * inputValues[0]
        neuron21.weights[1] += learningRate * error21 * inputValues[1]
        neuron21.weights[2] += learningRate * error21 * inputValues[2]
        neuron21.weights[3] += learningRate * error21 * inputValues[3]
        
        let error31 = y31 * (1 - y31) * (neuronOutput.weights[3] * errorOutput)
        neuron31.weights[0] += learningRate * error31 * inputValues[0]
        neuron31.weights[1] += learningRate * error31 * inputValues[1]
        neuron31.weights[2] += learningRate * error31 * inputValues[2]
        neuron31.weights[3] += learningRate * error31 * inputValues[3]
    }
}

var totalError: Double = 0
var dataToPrint: [String] = []
dataElements[1350..<1500].forEach { dataElement in
    let inputValues: [Double] = [
        -1,
        dataElement.pressureQuality,
        dataElement.pulse,
        dataElement.breathing
    ]
    
    let y11 = neuron11.activation(inputs: inputValues)
    let y21 = neuron21.activation(inputs: inputValues)
    let y31 = neuron31.activation(inputs: inputValues)
    
    let output = neuronOutput.activation(
        inputs: [
            -1,
             y11,
             y21,
             y31
        ]
    )
    
    dataToPrint.append("\(dataElement.id) | \(dataElement.pressureQuality) | \(dataElement.pulse) | \(dataElement.breathing) | \(Utilities.mapSigmoidOutputToSeverity(dataElement.severity)) | \(Utilities.mapSigmoidOutputToSeverity(output))")
    totalError += pow(dataElement.severity - output, 2)
}

print("id | qPA | pulso | frequência respiratória | gravidade esperada | gravidade obtida")
print(dataToPrint.joined(separator: "\n"))
print("Erro quadrático médio: \(totalError / 150)")
