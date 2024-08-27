import Foundation

struct DataElement: Identifiable {
    let id: Int
    let pressureQuality: Double
    let pulse: Double
    let breathing: Double
    let severity: Double
    
    init(id: Int, pressureQuality: Double, pulse: Double, breathing: Double, severity: Double) {
        self.id = id
        self.pressureQuality = Utilities.mapInputToSigmoide(pressureQuality, min: -10, max: 10)
        self.pulse = Utilities.mapInputToSigmoide(pulse, min: 0, max: 200)
        self.breathing = Utilities.mapInputToSigmoide(breathing, min: 0, max: 22)
        self.severity = Utilities.mapToSigmoideOutput(severity, min: 13.222719, max: 87)
    }
}
