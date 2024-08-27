import Foundation

class Utilities {
    static func mapInputToSigmoide(_ value: Double, min: Double, max: Double) -> Double {
        let value = (value - min) / (max - min)
        return (value * 0.8) + 0.1
    }
    
    static func mapToSigmoideOutput(_ value: Double, min: Double, max: Double) -> Double {
        return (value - min) / (max - min)
    }
    
    static func mapSigmoidOutputToSeverity(_ value: Double) -> Double {
        let min: Double = 13.222719
        let max: Double = 87
        return value * (max - min) + min
    }
    
    static func getRandomWeight() -> Double {
        return Double.random(in: (-1)...1)
    }
    
    static func getParsedDataElements(from filePath: String) -> [DataElement] {
        let fileURL = URL(fileURLWithPath: filePath)
        
        guard let content = try? String(contentsOf: fileURL, encoding: .utf8) else {
            fatalError("Failed to read contents of file")
        }
        
        let splittedContent = content.split(separator: "\n")
        
        let dataElements: [DataElement] = splittedContent.map { contentLine in
            let params = contentLine.split(separator: ",")
            
            guard
                let id = Int(params[0]),
                let pressureQuality = Double(params[3]),
                let pulse = Double(params[4]),
                let breathing = Double(params[5]),
                let severity = Double(params[6])
            else {
                fatalError("Failed to map data")
            }
            
            return DataElement(
                id: id,
                pressureQuality: pressureQuality,
                pulse: pulse,
                breathing: breathing,
                severity: severity
            )
        }
        
        return dataElements
    }
}
