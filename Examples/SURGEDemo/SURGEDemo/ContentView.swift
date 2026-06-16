import AppKit
import CoreGraphics
import ImageIO
import MLX
import MLXSurGe
import Observation
import SwiftUI
import UniformTypeIdentifiers

@main struct SURGEDemoApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

// MARK: - View model
//
// MainActor-isolated (the project defaults to MainActor isolation). All heavy
// work — model download, weight load, inference — runs off the main actor in a
// `Task.detached`; only Sendable values (`SurGeFrame`, progress fractions) hop
// back here. `SurGeSession` is `@unchecked Sendable`, so the single detached
// driver can own it while the UI reads snapshots.

@Observable
@MainActor
final class SurGeViewModel {
    enum Phase: Equatable {
        case needsModel
        case downloading(Double)
        case ready
        case inferring
    }

    var phase: Phase = .needsModel
    var status: String = ""
    var inputImage: CGImage?
    var depthImage: CGImage?
    var lastSeconds: Double?

    private let cacheDir = SurGeModelDownloader.defaultCacheDirectory()
    private var session: SurGeSession?

    init() {
        phase = SurGeModelDownloader.isDownloaded(at: cacheDir) ? .ready : .needsModel
    }

    var modelPresent: Bool { SurGeModelDownloader.isDownloaded(at: cacheDir) }

    func downloadModel() {
        phase = .downloading(0)
        status = "Downloading model…"
        let dir = cacheDir
        // Bind the weak ref to an immutable local before the nested Task so we
        // capture a `let` (not `self`) across the concurrency boundary.
        let report: @Sendable (Double) -> Void = { [weak self] frac in
            let vm = self
            Task { @MainActor in vm?.phase = .downloading(frac) }
        }
        // Download is I/O-bound (URLSession runs off-main internally), so a
        // MainActor Task that just awaits it is fine — no detached needed.
        Task { [weak self] in
            do {
                try await SurGeModelDownloader().download(to: dir, progress: report)
                self?.phase = .ready
                self?.status = "Model ready."
            } catch {
                self?.phase = .needsModel
                self?.status = "Download failed: \(error)"
            }
        }
    }

    func runInference(on cgImage: CGImage) {
        // Single-writer invariant for SurGeSession: only fire when `.ready`, and
        // flip to `.inferring` immediately so a second call (and the disabled
        // "Open Image…" button) can't drive the session concurrently.
        guard phase == .ready else { return }
        inputImage = cgImage
        depthImage = nil
        phase = .inferring
        status = "Running inference…"

        // Convert to Sendable host floats on the main actor; reconstruct the
        // MLXArray inside the detached task (CGImage / MLXArray aren't Sendable).
        let (floats, h, w) = Self.nhwcFloats(cgImage)
        let dir = cacheDir.path
        let existing = session

        Task.detached { [weak self] in
            let session: SurGeSession
            if let existing {
                session = existing
            } else {
                do {
                    session = try SurGeSession.load(
                        SurGeSessionConfig(weightsPath: dir, tokens: .min))
                } catch {
                    await MainActor.run { [weak self] in
                        self?.phase = .needsModel
                        self?.status = "Load failed: \(error)"
                    }
                    return
                }
                await MainActor.run { [weak self] in self?.session = session }
            }

            let (rgba, ow, oh, secs): ([UInt8], Int, Int, Double) = autoreleasepool {
                let image = MLXArray(floats, [1, h, w, 3])
                let frame = session.infer(image)
                return (frame.depthRGBA(), frame.width, frame.height, frame.inferenceSeconds)
            }
            // Hand back only Sendable bytes; build the CGImage on the main actor.
            await MainActor.run { [weak self] in
                self?.depthImage = Self.makeCGImage(rgba: rgba, width: ow, height: oh)
                self?.lastSeconds = secs
                self?.phase = .ready
                self?.status = String(format: "Done in %.2f s", secs)
            }
        }
    }

    // MARK: - CGImage <-> bytes (main-actor helpers)

    nonisolated static func nhwcFloats(_ cg: CGImage) -> ([Float], Int, Int) {
        let w = cg.width, h = cg.height
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        let info = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        rgba.withUnsafeMutableBytes { buf in
            CGContext(
                data: buf.baseAddress, width: w, height: h, bitsPerComponent: 8,
                bytesPerRow: w * 4, space: cs, bitmapInfo: info
            )?.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        }
        var floats = [Float](repeating: 0, count: w * h * 3)
        for i in 0..<(w * h) {
            floats[i * 3 + 0] = Float(rgba[i * 4 + 0]) / 255
            floats[i * 3 + 1] = Float(rgba[i * 4 + 1]) / 255
            floats[i * 3 + 2] = Float(rgba[i * 4 + 2]) / 255
        }
        return (floats, h, w)
    }

    nonisolated static func makeCGImage(rgba: [UInt8], width: Int, height: Int) -> CGImage? {
        guard let provider = CGDataProvider(data: Data(rgba) as CFData) else { return nil }
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        return CGImage(
            width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: width * 4, space: cs, bitmapInfo: info,
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    }
}

// MARK: - View

struct ContentView: View {
    @State private var vm = SurGeViewModel()

    var body: some View {
        VStack(spacing: 16) {
            Text("SurGe — monocular geometry")
                .font(.headline)

            switch vm.phase {
            case .needsModel:
                ContentUnavailableView {
                    Label("Model not downloaded", systemImage: "arrow.down.circle")
                } description: {
                    Text("Download karimknaebel/surge-large (~1.4 GB) to run inference.")
                } actions: {
                    Button("Download Model") { vm.downloadModel() }
                        .buttonStyle(.borderedProminent)
                }

            case .downloading(let frac):
                VStack(spacing: 8) {
                    ProgressView(value: frac) { Text("Downloading model…") }
                    Text("\(Int(frac * 100))%").monospacedDigit()
                }
                .frame(maxWidth: 320)

            case .ready, .inferring:
                HStack(spacing: 12) {
                    imagePane(vm.inputImage, label: "Input")
                    imagePane(vm.depthImage, label: "Depth")
                }
                HStack {
                    Button("Open Image…") { openImage() }
                        .disabled(vm.phase == .inferring)
                    if vm.phase == .inferring { ProgressView().controlSize(.small) }
                }
            }

            if !vm.status.isEmpty {
                Text(vm.status).font(.caption).foregroundStyle(.secondary)
            }
        }
        .padding()
        .frame(minWidth: 560, minHeight: 420)
    }

    @ViewBuilder
    private func imagePane(_ image: CGImage?, label: String) -> some View {
        VStack(spacing: 4) {
            if let image {
                Image(decorative: image, scale: 1)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxWidth: 256, maxHeight: 256)
                    .background(Color.black.opacity(0.05))
            } else {
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color.gray.opacity(0.1))
                    .frame(width: 256, height: 256)
                    .overlay(Text("—").foregroundStyle(.tertiary))
            }
            Text(label).font(.caption)
        }
    }

    private func openImage() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.png, .jpeg, .image]
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let url = panel.url,
              let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else { return }
        vm.runInference(on: cg)
    }
}

#Preview {
    ContentView()
}
