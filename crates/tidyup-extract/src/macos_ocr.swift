import CoreImage
import Foundation
import ImageIO
import Vision

guard CommandLine.arguments.count == 2 else {
    FileHandle.standardError.write(Data("usage: tidyup-ocr <image>\n".utf8))
    exit(2)
}

let url = URL(fileURLWithPath: CommandLine.arguments[1])

func recognize(_ handler: VNImageRequestHandler) -> Result<[String], Error> {
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true
    do {
        try handler.perform([request])
        return .success((request.results ?? []).compactMap { observation in
            observation.topCandidates(1).first?.string
        })
    } catch {
        return .failure(error)
    }
}

var failures: [String] = []
var attempts: [VNImageRequestHandler] = [VNImageRequestHandler(url: url, options: [:])]
if let source = CGImageSourceCreateWithURL(url as CFURL, nil),
   let image = CGImageSourceCreateImageAtIndex(source, 0, nil) {
    attempts.append(VNImageRequestHandler(cgImage: image, options: [:]))
} else {
    failures.append("ImageIO could not decode the image")
}
if let image = CIImage(contentsOf: url) {
    attempts.append(VNImageRequestHandler(ciImage: image, options: [:]))
} else {
    failures.append("CoreImage could not decode the image")
}

var lines: [String]?
for handler in attempts {
    switch recognize(handler) {
    case .success(let recognized):
        lines = recognized
    case .failure(let error):
        failures.append(error.localizedDescription)
    }
    if lines != nil { break }
}

guard let lines else {
    let detail = failures.joined(separator: "; ")
    FileHandle.standardError.write(Data("OCR failed: \(detail)\n".utf8))
    exit(5)
}
for line in lines {
    print(line)
}
