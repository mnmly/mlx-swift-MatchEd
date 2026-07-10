// ContentView — preview surface + run controls. Presentation only.
//
// Apple-design notes: response lives on the native controls (press feedback is
// instant); the primary action is prominent; results *materialize* with a
// critically-damped spring (damping 1.0, response ~0.35) rather than a flat
// fade, and that motion collapses to a plain cross-fade under Reduce Motion.
// Type is the system font; filenames are monospaced for scannability.

import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @State private var vm = DetectionViewModel()
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        VStack(spacing: 14) {
            controls

            HStack(spacing: 12) {
                imagePane("Input", vm.inputImage, systemImage: "photo")
                imagePane("Edge map", vm.edgeImage, systemImage: "scribble.variable")
                imagePane("Crisp", vm.thinImage, systemImage: "wand.and.rays")
            }
            .frame(maxHeight: .infinity)

            HStack(spacing: 6) {
                if vm.isRunning {
                    ProgressView().controlSize(.small)
                }
                Text(vm.status)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .contentTransition(.opacity)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(16)
    }

    // MARK: Controls

    private var controls: some View {
        HStack(spacing: 10) {
            Button { pickFile(types: [.data]) { vm.weightsURL = $0; refreshStatus() } } label: {
                Label("Weights", systemImage: "shippingbox")
            }
            filenameLabel(vm.weightsURL, placeholder: "no .safetensors")

            Divider().frame(height: 20)

            Button { pickFile(types: [.image]) { vm.imageURL = $0; refreshStatus() } } label: {
                Label("Image", systemImage: "photo.badge.plus")
            }
            filenameLabel(vm.imageURL, placeholder: "no image")

            Spacer()

            Button("Detect", systemImage: "sparkle.magnifyingglass") { vm.detect() }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.return, modifiers: [])
                .disabled(!vm.canRun)
        }
        .padding(10)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(.separator.opacity(0.6), lineWidth: 1)
        )
    }

    // MARK: Panes

    private func imagePane(_ title: String, _ image: CGImage?, systemImage: String) -> some View {
        VStack(spacing: 8) {
            Text(title)
                .font(.headline)
                .foregroundStyle(.primary)
            ZStack {
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(.quaternary.opacity(0.4))
                    .overlay(
                        RoundedRectangle(cornerRadius: 10, style: .continuous)
                            .strokeBorder(.separator.opacity(0.5), lineWidth: 1)
                    )

                if let image {
                    Image(decorative: image, scale: 1, orientation: .up)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .padding(8)
                        .transition(revealTransition)
                } else {
                    Image(systemName: systemImage)
                        .font(.system(size: 34, weight: .light))
                        .foregroundStyle(.tertiary)
                }
            }
            // Results "materialize" — spring scale+fade, or a plain cross-fade
            // when the user prefers reduced motion.
            .animation(revealAnimation, value: image.map { ObjectIdentifier($0) })
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var revealTransition: AnyTransition {
        reduceMotion ? .opacity
                     : .scale(scale: 0.97).combined(with: .opacity)
    }

    private var revealAnimation: Animation {
        reduceMotion ? .easeOut(duration: 0.2)
                     : .spring(response: 0.35, dampingFraction: 1.0)  // critically damped
    }

    // MARK: Bits

    private func filenameLabel(_ url: URL?, placeholder: String) -> some View {
        Text(url?.lastPathComponent ?? placeholder)
            .font(.caption.monospaced())
            .foregroundStyle(url == nil ? .tertiary : .secondary)
            .lineLimit(1)
            .truncationMode(.middle)
            .frame(maxWidth: 150, alignment: .leading)
    }

    private func refreshStatus() {
        if vm.canRun { vm.status = "Ready. Press Detect (↩)." }
    }

    private func pickFile(types: [UTType], _ completion: (URL) -> Void) {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = types
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        if panel.runModal() == .OK, let url = panel.url { completion(url) }
    }
}

#Preview {
    ContentView().frame(width: 820, height: 580)
}
