// Downloads the SurGe checkpoint from Hugging Face so the GUI can offer a
// "Download Model" button when the weights are missing. Library-side (Foundation
// only, no SwiftUI) so the CLI can reuse it too.

import Foundation

public enum SurGeDownloadError: Error, CustomStringConvertible {
    case badResponse(file: String, code: Int)
    case writeFailed(String)

    public var description: String {
        switch self {
        case .badResponse(let f, let c): return "Download of \(f) failed (HTTP \(c))"
        case .writeFailed(let p): return "Could not write \(p)"
        }
    }
}

/// Fetches `config.json` + `model.safetensors` for a Hugging Face model repo.
public struct SurGeModelDownloader: Sendable {
    public static let defaultRepoID = "karimknaebel/surge-large"

    /// Files that make up the snapshot. `config.json` is tiny; the safetensors
    /// dominates, so overall progress tracks it.
    public static let files = ["config.json", "model.safetensors"]

    public let repoID: String
    public let revision: String

    public init(repoID: String = SurGeModelDownloader.defaultRepoID, revision: String = "main") {
        self.repoID = repoID
        self.revision = revision
    }

    /// Default on-device cache location: `~/Library/Caches/MLXSurGe/<repo>`.
    public static func defaultCacheDirectory(
        repoID: String = SurGeModelDownloader.defaultRepoID
    ) -> URL {
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        return caches.appendingPathComponent("MLXSurGe", isDirectory: true)
            .appendingPathComponent(repoID.replacingOccurrences(of: "/", with: "--"), isDirectory: true)
    }

    /// True when every required file already exists at `dir`.
    public static func isDownloaded(at dir: URL) -> Bool {
        files.allSatisfy {
            FileManager.default.fileExists(atPath: dir.appendingPathComponent($0).path)
        }
    }

    private func url(for file: String) -> URL {
        URL(string: "https://huggingface.co/\(repoID)/resolve/\(revision)/\(file)")!
    }

    /// Download any missing files into `dir`, reporting fractional progress in
    /// `[0, 1]`. The progress handler may be called from a background queue.
    public func download(
        to dir: URL, progress: @escaping @Sendable (Double) -> Void
    ) async throws {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        // config.json first (negligible size, no progress).
        let configDst = dir.appendingPathComponent("config.json")
        if !FileManager.default.fileExists(atPath: configDst.path) {
            try await fetch(url(for: "config.json"), to: configDst, file: "config.json", progress: nil)
        }
        progress(0.01)

        // model.safetensors drives the bar.
        let weightsDst = dir.appendingPathComponent("model.safetensors")
        if !FileManager.default.fileExists(atPath: weightsDst.path) {
            try await fetch(url(for: "model.safetensors"), to: weightsDst, file: "model.safetensors") {
                progress(max(0.01, $0))
            }
        }
        progress(1.0)
    }

    private func fetch(
        _ url: URL, to dst: URL, file: String, progress: (@Sendable (Double) -> Void)?
    ) async throws {
        let (tempURL, response) = try await URLSession.shared.downloadWithProgress(
            from: url, progress: progress)
        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            throw SurGeDownloadError.badResponse(file: file, code: http.statusCode)
        }
        do {
            if FileManager.default.fileExists(atPath: dst.path) {
                try FileManager.default.removeItem(at: dst)
            }
            try FileManager.default.moveItem(at: tempURL, to: dst)
        } catch {
            throw SurGeDownloadError.writeFailed(dst.path)
        }
    }
}

// MARK: - Progress-reporting download via URLSession delegate

extension URLSession {
    /// Like `download(from:)` but reports fractional progress. Bridges a
    /// `URLSessionDownloadDelegate` to async/await.
    fileprivate func downloadWithProgress(
        from url: URL, progress: (@Sendable (Double) -> Void)?
    ) async throws -> (URL, URLResponse) {
        let delegate = DownloadProgressDelegate(progress: progress)
        let session = URLSession(configuration: .default, delegate: delegate, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }
        return try await withCheckedThrowingContinuation { continuation in
            delegate.continuation = continuation
            let task = session.downloadTask(with: url)
            task.resume()
        }
    }
}

/// `@unchecked Sendable` invariant: URLSession delivers all delegate callbacks
/// for a task serially on the session's delegate queue, and `continuation` is
/// assigned before `task.resume()` (happens-before the first callback). So the
/// mutable `continuation` / `resumed` are never touched concurrently.
private final class DownloadProgressDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    let progress: (@Sendable (Double) -> Void)?
    var continuation: CheckedContinuation<(URL, URLResponse), Error>?
    // The downloaded temp file is deleted when the delegate callback returns, so
    // move it to our own temp location synchronously inside the callback.
    private var resumed = false

    init(progress: (@Sendable (Double) -> Void)?) {
        self.progress = progress
    }

    func urlSession(
        _ session: URLSession, downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64, totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        guard totalBytesExpectedToWrite > 0 else { return }
        progress?(Double(totalBytesWritten) / Double(totalBytesExpectedToWrite))
    }

    func urlSession(
        _ session: URLSession, downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        let dst = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        do {
            try FileManager.default.moveItem(at: location, to: dst)
            let response = downloadTask.response ?? URLResponse()
            resumeOnce(.success((dst, response)))
        } catch {
            resumeOnce(.failure(error))
        }
    }

    func urlSession(
        _ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?
    ) {
        if let error { resumeOnce(.failure(error)) }
    }

    private func resumeOnce(_ result: Result<(URL, URLResponse), Error>) {
        guard !resumed else { return }
        resumed = true
        continuation?.resume(with: result)
        continuation = nil
    }
}
