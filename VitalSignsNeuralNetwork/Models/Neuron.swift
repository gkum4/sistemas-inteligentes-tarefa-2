import Foundation

class Neuron {
    var weights: [Double] = []
    let activationFunctionType: ActivationFunctionType
    
    init(
        weightsCount: Int,
        activation: ActivationFunctionType = .sigmoide
    ) {
        weights = Array(1...weightsCount).map { _ in Double.random(in: (-1)...1) }
        self.activationFunctionType = activation
    }
    
    func activation(inputs: [Double]) -> Double {
        switch activationFunctionType {
        case .step:
            return step(inputs: inputs)
        case .sigmoide:
            return sigmoide(inputs: inputs)
        }
    }
}

extension Neuron {
    private func step(inputs: [Double]) -> Double {
        let sum = getInputsWeightsSum(inputs)
        return sum >= 0 ? 1 : 0
    }
    
    private func sigmoide(inputs: [Double]) -> Double {
        let sum = getInputsWeightsSum(inputs)
        return 1 / (1 + exp(-sum))
    }
    
    private func getInputsWeightsSum(_ inputs: [Double]) -> Double {
        var sum: Double = 0
        inputs.enumerated().forEach { index, value in
            sum += value * weights[index]
        }
        return sum
    }
}
