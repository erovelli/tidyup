import CoreImage
import Foundation
import Vision

guard CommandLine.arguments.count == 2 else {
    FileHandle.standardError.write(Data("usage: tidyup-ocr <image>\n".utf8))
    exit(2)
}

let url = URL(fileURLWithPath: CommandLine.arguments[1])

func recognize(_ handler: VNImageRequestHandler) -> [String]? {
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true
    do {
        try handler.perform([request])
        return (request.results ?? []).compactMap { observation in
            observation.topCandidates(1).first?.string
        }
    } catch {
        return nil
    }
}

let direct = recognize(VNImageRequestHandler(url: url, options: [:]))
let lines = direct ?? CIImage(contentsOf: url).flatMap { image in
    recognize(VNImageRequestHandler(ciImage: image, options: [:]))
}
guard let lines else {
    FileHandle.standardError.write(Data("OCR failed for both image decoders\n".utf8))
    exit(5)
}
for line in lines {
    print(line)
}
