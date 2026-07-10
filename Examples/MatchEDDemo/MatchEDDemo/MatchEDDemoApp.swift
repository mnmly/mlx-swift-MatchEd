// MatchEDDemo — a minimal SwiftUI frontend for the shared MatchEDKit.MatchED
// pipeline.
//
// The Xcode project depends on the repo root as a local SwiftPM package
// (product "MatchEDKit"). The app contains no model / conv / weight code — it
// only drives `MatchED` and presents its edge maps (swift-cli-gui-shared-driver).

import SwiftUI

@main
struct MatchEDDemoApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .frame(minWidth: 760, minHeight: 560)
        }
    }
}
